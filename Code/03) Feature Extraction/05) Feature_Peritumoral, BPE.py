"""
03c_peritumoral.py ─ 종양 주변·배경 실질 영상 특징 (G1 주변 질감 / G2 BPE 근사)
==============================================================================
종양 안만 보던 기존 특징과 달리 "종양 바깥 조직"의 조영을 본다. 모든 계산은 환자 내부(라벨 미사용).

  거리 지도 : 종양 경계로부터의 실제 거리(mm, SignedMaurerDistanceMap)
  고리      : 1=(0,5]  2=(5,10]  3=(10,20]  4=(20,40] mm   ∩ 조직(공기 제외)
  조직 마스크: 조영 전 영상 Otsu 임계 이상 (공기·배경 제외)
  고리 4(20–40mm) 증강 = 배경 실질 증강(BPE) 근사 → TCIA BPE_5slice_mean_T0(384명)와 상관으로 검증
    ※ 종양이 흉벽에 가까우면 고리에 흉벽·심장 조영이 섞일 수 있음 (한계)

출력 (Data/Interim/<pid>/): pe_early_wide.nii.gz, shells.nii.gz (라벨 1–4)   ← 04 질감 추출용
       Data/Features/peri_features.csv  (pid × peri_*: 고리별 1차 통계)
QC:    Evaluate/03c_peri_qc.csv

실행:  python 03c_peritumoral.py --limit 5 / (전체)
"""
from __future__ import annotations
import argparse
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
pp = import_module("03_preprocess")

SHELLS = [(0, 5), (5, 10), (10, 20), (20, 40)]   # mm
MARGIN_MM = 45.0


def _box(mask, spacing, sitk, margin_mm):
    f = sitk.LabelShapeStatisticsImageFilter(); f.Execute(mask)
    if not f.HasLabel(1):
        return None
    bb = f.GetBoundingBox(1); size_img = mask.GetSize(); idx, size = [], []
    for d in range(3):
        m = int(np.ceil(margin_mm / spacing[d]))
        lo = max(0, bb[d] - m); hi = min(size_img[d], bb[d] + bb[d + 3] + m)
        idx.append(int(lo)); size.append(int(hi - lo))
    return idx, size


def _stats(prefix, v, o):
    if v.size < 20:
        return
    o[f"{prefix}_mean"] = float(v.mean()); o[f"{prefix}_p50"] = float(np.median(v))
    o[f"{prefix}_p90"] = float(np.percentile(v, 90)); o[f"{prefix}_std"] = float(v.std())


def peri_one(pid, r, spacing, idx, sitk):
    t_pre, t_e, t_l = (int(r[c]) for c in (cfg.COL_T_PRE, cfg.COL_T_EARLY, cfg.COL_T_LATE))
    files = {}
    for k, name in (("mask", cfg.MASK_PATTERN.format(pid=pid)),
                    ("pre", cfg.DCE_TP_PATTERN.format(pid=pid, t=t_pre)),
                    ("early", cfg.DCE_TP_PATTERN.format(pid=pid, t=t_e)),
                    ("late", cfg.DCE_TP_PATTERN.format(pid=pid, t=t_l))):
        f, err = pp._lookup(idx, name)
        if err:
            return None, f"{k}{err}"
        files[k] = f
    mask = sitk.Cast(sitk.ReadImage(str(files["mask"]), sitk.sitkUInt8) > 0, sitk.sitkUInt8)
    mask.SetSpacing(spacing); mask.SetOrigin((0.0, 0.0, 0.0)); mask.SetDirection(pp.IDENTITY)
    box = _box(mask, spacing, sitk, MARGIN_MM)
    if box is None:
        return None, "빈마스크"
    mask = sitk.RegionOfInterest(mask, box[1], box[0])
    im = {k: sitk.RegionOfInterest(pp._read(files[k], spacing, sitk, sitk.sitkFloat32), box[1], box[0])
          for k in ("pre", "early", "late")}

    a = {k: sitk.GetArrayFromImage(v) for k, v in im.items()}
    m = sitk.GetArrayFromImage(mask).astype(bool)
    ref = float(np.median(a["pre"][m]))
    if not ref > 0:
        return None, "pre신호이상"
    den = np.maximum(a["pre"], cfg.PE_FLOOR_FRAC * ref)
    pe_e = (100.0 * (a["early"] - a["pre"]) / den).astype(np.float32)
    pe_l = (100.0 * (a["late"] - a["pre"]) / den).astype(np.float32)
    tissue_img = sitk.OtsuThreshold(im["pre"], 0, 1, 200)                    # 조직=1, 공기=0
    tissue = sitk.GetArrayFromImage(tissue_img).astype(bool)

    # 등방 1mm 격자로 (질감 추출과 거리 계산을 mm 단위로 일관되게)
    def to_img(arr):
        x = sitk.GetImageFromArray(arr); x.CopyInformation(im["pre"]); return x
    pe_e_i = pp._resample(to_img(pe_e), cfg.RESAMPLE_MM, sitk, sitk.sitkLinear, sitk.sitkFloat32)
    pe_l_i = sitk.Resample(to_img(pe_l), pe_e_i, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    mask_i = sitk.Resample(mask, pe_e_i, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    tis_i = sitk.Resample(sitk.Cast(tissue_img, sitk.sitkUInt8), pe_e_i, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    # FTV 마스크는 증강 voxel의 모음이라 점처럼 흩어져 있음 → 닫힘(3mm)+구멍 채우기로 종양 외곽선을 만든 뒤
    # 그 외곽선 기준 거리로 고리를 정의 (흩어진 조각마다 고리가 생기는 것 방지)
    env = sitk.BinaryFillhole(sitk.BinaryMorphologicalClosing(mask_i, [3, 3, 3]))
    dist = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        env, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
    tis = sitk.GetArrayFromImage(tis_i).astype(bool)
    vE, vL = sitk.GetArrayFromImage(pe_e_i), sitk.GetArrayFromImage(pe_l_i)

    lab = np.zeros(dist.shape, dtype=np.uint8)
    o, q = {}, {}
    for k, (lo, hi) in enumerate(SHELLS, 1):
        sh = (dist > lo) & (dist <= hi) & tis
        lab[sh] = k
        tag = f"peri_r{lo}_{hi}"
        _stats(f"{tag}_pe_early", vE[sh], o); _stats(f"{tag}_pe_late", vL[sh], o)
        if sh.sum() >= 20:
            o[f"{tag}_frac_enh50"] = float((vE[sh] > 50).mean())     # 50% 이상 조기 증강 비율
            o[f"{tag}_washout_ratio"] = float(np.median(vL[sh]) - np.median(vE[sh]))
        q[f"vox_r{lo}_{hi}"] = int(sh.sum())
    # 종양 대비 주변 대조 (경계가 뚜렷한가)
    tv = vE[sitk.GetArrayFromImage(mask_i).astype(bool)]
    for lo, hi in SHELLS[:2]:
        key = f"peri_r{lo}_{hi}_pe_early_mean"
        if key in o and tv.size:
            o[f"peri_contrast_r{lo}_{hi}"] = float(tv.mean() - o[key])
    q["tissue_frac"] = float(tissue.mean())
    q["env_over_mask"] = float(sitk.GetArrayFromImage(env).sum() / max(1, int(tv.size)))  # 외곽선/FTV 부피비

    out = cfg.DATA_INTERIM / pid; out.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pe_e_i, str(out / "pe_early_wide.nii.gz"))
    shells = sitk.GetImageFromArray(lab); shells.CopyInformation(pe_e_i)
    sitk.WriteImage(shells, str(out / "shells.nii.gz"))
    return (o, q), "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg.ensure_dirs()
    sitk = pp._sitk()
    meta = pd.read_csv(cfg.METADATA_CSV)
    rows = meta.to_dict("records")[: args.limit or None]
    print("[i] 파일 색인 중 ...", flush=True)
    idx = pp._index(pp.EXTRACT_DIR)
    print(f"[i] 주변·배경 특징 대상 {len(rows)}명")

    recs, qc = [], []
    for i, r in enumerate(rows, 1):
        pid = str(r[cfg.COL_PID])
        try:
            sp = (float(r[cfg.COL_SLICE_THICK]), float(r[cfg.COL_XY_SPACING]), float(r[cfg.COL_XY_SPACING]))
            res, st = peri_one(pid, r, sp, idx, sitk)
        except Exception as e:  # noqa: BLE001
            res, st = None, f"오류:{e}"
        qc.append({"pid": pid, "status": st, **(res[1] if res else {})})
        if res:
            recs.append({cfg.COL_PID: pid, **res[0]})
            if i % 100 == 0 or i <= 3:
                o = res[0]
                print(f"  [{i}/{len(rows)}] {pid} ✓ 0–5mm PE {o.get('peri_r0_5_pe_early_mean', float('nan')):.0f}%"
                      f" · 20–40mm PE {o.get('peri_r20_40_pe_early_mean', float('nan')):.0f}%", flush=True)
        else:
            print(f"  [{i}/{len(rows)}] {pid} {st}")

    pd.DataFrame(qc).to_csv(cfg.EVAL_DIR / "03c_peri_qc.csv", index=False)
    if not recs:
        print("[!] 추출 0건"); return 2
    df = pd.DataFrame(recs)
    df.to_csv(cfg.DATA_FEATURES / "peri_features.csv", index=False)
    print(f"\n[✓] peri_features.csv ({df.shape[0]}명 × {df.shape[1]-1}특징) · 결측 있는 특징 {int(df.isna().any().sum())}")

    # G2 검증: 20–40mm 고리 증강 vs TCIA BPE (384명 하위집합)
    mf = cfg.DATA_ORIGINAL / "tcia_clinical" / "Multi-feature-MRI-NACT-Data.xlsx"
    if mf.exists() and "peri_r20_40_pe_early_mean" in df:
        t = pd.read_excel(mf)[["CLINICAL-TRIAL-SUBJECT-ID", "BPE_5slice_mean_T0"]]
        t["num"] = t.iloc[:, 0].astype(int)
        df["num"] = df[cfg.COL_PID].str.split("-").str[-1].astype(int)
        j = df.merge(t, on="num")
        for c in ("peri_r20_40_pe_early_mean", "peri_r20_40_pe_early_p50", "peri_r10_20_pe_early_mean"):
            if c in j:
                print(f"[검증] {c} vs TCIA BPE_T0: Spearman r = "
                      f"{j[c].corr(j['BPE_5slice_mean_T0'], method='spearman'):.2f} (n={j[c].notna().sum()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
