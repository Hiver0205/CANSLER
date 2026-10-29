"""
03_preprocess.py ─ 치료 전 DCE 전처리
====================================
각 환자의 치료 전(vis1) DCE에서 **조영전(pre)**과 **후기(post_late)** 시점을 뽑아
spacing 복원 → 마스크 주변 crop → percent enhancement(PE) → 등방 리샘플 하여 저장한다.

  PE = 100 · (late − pre) / max(pre, floor)     [% 단위, voxel별]
    · 곱셈형 bias field·스캐너 이득이 비율에서 상쇄 → N4·z-score 불필요
    · 종양 ROI z-score(구버전)는 조영 강도 자체를 지워 firstorder 특징을 무의미하게 만들었음

입력 (02 해제 결과): Data/Interim/extracted/.../{pid}_spy2_vis1_dce_aqc_{t}.nii.gz
                     Data/Interim/extracted/.../{pid}_{pid}_spy2_vis1_mask.nii.gz
      시점 인덱스·spacing : metadata 의 pre / post_late / slice_thick / xy_spacing
출력 (Data/Interim/<pid>/): pe.nii.gz, mask.nii.gz
                            pe_early.nii.gz (초기 PE), ser.nii.gz (SER, ROI 안만) — 동역학 지도 질감용
QC   : Evaluate/03_preprocess_qc.csv  (환자별 상태·사용 파일·PE 분포)

실행:
    python 03_preprocess.py --limit 5   # 소량 검증
    python 03_preprocess.py             # 전체

주의: NIfTI 헤더에 물리 정보가 없음(affine=단위행렬, pixdim=1) → spacing은 metadata에서 복원.
      파일명은 vis1까지 정확히 일치해야 하며, 없거나 중복이면 건너뛰고 QC에 기록(조용한 대체 없음).
"""
from __future__ import annotations
import argparse
import math
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
EXTRACT_DIR = cfg.DATA_INTERIM / "extracted"
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


def _sitk():
    try:
        import SimpleITK as sitk  # noqa: N813
        return sitk
    except ImportError:
        print("[!] SimpleITK 필요:  pip install SimpleITK"); raise


def _index(root):
    """파일명 → 경로 목록. (환자마다 rglob 하지 않고 한 번만 훑음, 중복 탐지용)"""
    idx = {}
    for p in root.rglob("*.nii.gz"):
        if p.is_file():
            idx.setdefault(p.name, []).append(p)
    return idx


def _lookup(idx, name):
    hits = idx.get(name, [])
    if not hits:
        return None, f"없음:{name}"
    if len(hits) > 1:
        return None, f"중복:{name}({len(hits)})"
    return hits[0], None


def _read(path, spacing, sitk, pixel):
    img = sitk.ReadImage(str(path), pixel)
    img.SetSpacing(spacing); img.SetOrigin((0.0, 0.0, 0.0)); img.SetDirection(IDENTITY)
    return img


def _crop_box(mask, spacing, sitk):
    f = sitk.LabelShapeStatisticsImageFilter(); f.Execute(mask)
    if not f.HasLabel(1):
        return None
    bb = f.GetBoundingBox(1)                       # (x0, y0, z0, sx, sy, sz) — sitk 인덱스 순서
    size_img = mask.GetSize()
    idx, size = [], []
    for d in range(3):
        m = int(math.ceil(cfg.CROP_MARGIN_MM / spacing[d]))
        lo = max(0, bb[d] - m); hi = min(size_img[d], bb[d] + bb[d + 3] + m)
        idx.append(int(lo)); size.append(int(hi - lo))
    return idx, size


def _resample(img, spacing, sitk, interp, pixel):
    if spacing is None:
        return img
    osz, ospc = img.GetSize(), img.GetSpacing()
    nsz = [max(1, int(round(s * o / n))) for s, o, n in zip(osz, ospc, spacing)]
    return sitk.Resample(img, nsz, sitk.Transform(), interp, img.GetOrigin(),
                         tuple(float(s) for s in spacing), img.GetDirection(), 0.0, pixel)


def process_one(pid, t_pre, t_late, spacing, idx, sitk, t_early=None):
    """반환: QC dict (status == "OK" 이면 저장 완료)."""
    q = {"pid": pid, "t_pre": t_pre, "t_late": t_late,
         "spacing": "x".join(f"{s:.4g}" for s in spacing)}

    mask_f, err = _lookup(idx, cfg.MASK_PATTERN.format(pid=pid))
    if err:
        q["status"] = "마스크" + err; return q
    pre_f, err = _lookup(idx, cfg.DCE_TP_PATTERN.format(pid=pid, t=int(t_pre)))
    if err:
        q["status"] = "pre" + err; return q
    late_f, err = _lookup(idx, cfg.DCE_TP_PATTERN.format(pid=pid, t=int(t_late)))
    if err:
        q["status"] = "late" + err; return q
    early_f = None
    if t_early is not None:
        early_f, err = _lookup(idx, cfg.DCE_TP_PATTERN.format(pid=pid, t=int(t_early)))
        if err:
            q["status"] = "early" + err; return q
    q.update(pre_file=pre_f.name, late_file=late_f.name, mask_file=mask_f.name,
             early_file=early_f.name if early_f else None)

    pre  = _read(pre_f,  spacing, sitk, sitk.sitkFloat32)
    late = _read(late_f, spacing, sitk, sitk.sitkFloat32)
    early = _read(early_f, spacing, sitk, sitk.sitkFloat32) if early_f else None
    mask = sitk.Cast(sitk.ReadImage(str(mask_f), sitk.sitkUInt8) > 0, sitk.sitkUInt8)
    mask.SetSpacing(spacing); mask.SetOrigin((0.0, 0.0, 0.0)); mask.SetDirection(IDENTITY)
    q["size"] = "x".join(map(str, pre.GetSize()))
    if not (pre.GetSize() == late.GetSize() == mask.GetSize()) or \
            (early is not None and early.GetSize() != pre.GetSize()):
        q["status"] = (f"크기불일치 pre{pre.GetSize()} late{late.GetSize()} mask{mask.GetSize()}")
        return q

    box = _crop_box(mask, spacing, sitk)
    if box is None:
        q["status"] = "빈마스크"; return q
    pre, late, mask = (sitk.RegionOfInterest(im, box[1], box[0]) for im in (pre, late, mask))
    if early is not None:
        early = sitk.RegionOfInterest(early, box[1], box[0])

    if cfg.DO_N4:   # 기본 꺼짐: PE 비율에서 bias field가 상쇄됨
        f = sitk.N4BiasFieldCorrectionImageFilter()
        f.Execute(pre, sitk.OtsuThreshold(pre, 0, 1, 200))
        log_bias = f.GetLogBiasFieldAsImage(pre)            # pre의 bias field 하나를
        pre  = sitk.Divide(pre,  sitk.Exp(log_bias))         # pre·late 양쪽에 동일 적용
        late = sitk.Divide(late, sitk.Exp(log_bias))

    a_pre, a_late = sitk.GetArrayFromImage(pre), sitk.GetArrayFromImage(late)
    m = sitk.GetArrayFromImage(mask).astype(bool)
    q["mask_vox"] = int(m.sum())
    ref = float(np.median(a_pre[m]))
    if not ref > 0:
        q["status"] = f"pre신호이상(median={ref:.3g})"; return q
    floor = cfg.PE_FLOOR_FRAC * ref
    q["floor_frac"] = float((a_pre[m] < floor).mean())     # 분모 하한에 걸린 ROI 비율
    den = np.maximum(a_pre, floor)
    pe_a = (100.0 * (a_late - a_pre) / den).astype(np.float32)
    pe = sitk.GetImageFromArray(pe_a); pe.CopyInformation(pre)

    # 추가 동역학 지도 (질감 추출용): 초기 PE, SER=(S_e−S_0)/(S_l−S_0) [0,3] (ROI 밖·분모≤0 → 0)
    extra = {}
    if early is not None:
        a_e = sitk.GetArrayFromImage(early)
        extra["pe_early"] = (100.0 * (a_e - a_pre) / den).astype(np.float32)
        d_l = a_late - a_pre; ok = m & (d_l > 0)
        ser = np.zeros_like(pe_a)
        ser[ok] = np.clip((a_e - a_pre)[ok] / d_l[ok], 0.0, 3.0)
        extra["ser"] = ser

    pe   = _resample(pe, cfg.RESAMPLE_MM, sitk, sitk.sitkLinear, sitk.sitkFloat32)
    mask = sitk.Resample(mask, pe, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    v = sitk.GetArrayFromImage(pe)[sitk.GetArrayFromImage(mask).astype(bool)]
    q["roi_vox_resampled"] = int(v.size)
    if v.size < 10:
        q["status"] = "리샘플후ROI과소"; return q
    q.update(pe_median=float(np.median(v)), pe_p1=float(np.percentile(v, 1)),
             pe_p99=float(np.percentile(v, 99)))

    out = cfg.DATA_INTERIM / pid; out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pe,   str(out / "pe.nii.gz"))
    sitk.WriteImage(mask, str(out / "mask.nii.gz"))
    for name, arr in extra.items():
        im = sitk.GetImageFromArray(arr); im.CopyInformation(pre)
        im = sitk.Resample(im, pe, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)  # pe와 같은 격자
        sitk.WriteImage(im, str(out / f"{name}.nii.gz"))
    q["status"] = "OK"
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg.ensure_dirs()

    if not EXTRACT_DIR.exists():
        print(f"[!] 해제 폴더 없음: {EXTRACT_DIR}. 먼저 02_extract_images.py 실행.")
        return 1
    meta = pd.read_csv(cfg.METADATA_CSV)
    sitk = _sitk()

    cols = [cfg.COL_PID, cfg.COL_T_PRE, cfg.COL_T_LATE, cfg.COL_T_EARLY, cfg.COL_SLICE_THICK, cfg.COL_XY_SPACING]
    rows = meta[cols].to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    print(f"[i] 파일 색인 중: {EXTRACT_DIR} ...", flush=True)
    idx = _index(EXTRACT_DIR)
    print(f"[i] NIfTI {sum(len(v) for v in idx.values())}개 · 전처리 대상 {len(rows)}명")

    qc = []
    for i, r in enumerate(rows, 1):
        pid = str(r[cfg.COL_PID])
        st, xy = r[cfg.COL_SLICE_THICK], r[cfg.COL_XY_SPACING]
        if pd.isna(st) or pd.isna(xy) or pd.isna(r[cfg.COL_T_PRE]) or pd.isna(r[cfg.COL_T_LATE]):
            q = {"pid": pid, "status": "metadata결측(spacing/시점)"}
        else:
            spacing = (float(st), float(xy), float(xy))   # (slice, row, col)
            try:
                t_e = None if pd.isna(r[cfg.COL_T_EARLY]) else r[cfg.COL_T_EARLY]
                q = process_one(pid, r[cfg.COL_T_PRE], r[cfg.COL_T_LATE], spacing, idx, sitk, t_early=t_e)
            except Exception as e:  # noqa: BLE001
                q = {"pid": pid, "status": f"오류:{e}"}
        qc.append(q)
        if q["status"] == "OK":
            if i % 25 == 0 or i <= 5:
                print(f"  [{i}/{len(rows)}] {pid} ✓ PE중앙값 {q['pe_median']:.0f}%")
        else:
            print(f"  [{i}/{len(rows)}] {pid} {q['status']}")

    qdf = pd.DataFrame(qc)
    qdf.to_csv(cfg.EVAL_DIR / "03_preprocess_qc.csv", index=False)
    vc = qdf["status"].str.split(":").str[0].value_counts()
    print("\n[상태 요약]\n" + vc.to_string())
    print(f"[✓] → {cfg.DATA_INTERIM} · QC: Evaluate/03_preprocess_qc.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
