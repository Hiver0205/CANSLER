"""
03e_timecurve.py ─ J1: 전체 시간-증강 곡선 지도 (모든 조영 후 시점 사용)
=======================================================================
기존 특징은 3개 시점(조영 전·초기·후기)만 썼다. 여기서는 환자당 모든 시점(4~11개)으로
voxel별 증강 곡선 PE_t (t = 1..T) 을 만들고 곡선 모양을 지도로 요약한다. 환자 내부 계산(라벨 미사용).

  washin   = PE_peak / TTP                    최고점까지의 평균 유입 속도 (%/시점)
  washout  = (PE_last − PE_peak) / (T − TTP)  최고점 이후 유출 속도 (%/시점, 최고점이 마지막이면 0)
  auc      = mean_t PE_t                      곡선 아래 면적을 시점 수로 정규화 (시점 수가 달라도 비교 가능)
  early_frac = PE_1 / PE_peak                  첫 조영 후 시점에 이미 최고점의 몇 %인가
  ※ 촬영 시각(초)이 없어 시간은 시점 번호. 시간 해상도 차이는 한계로 남음.

지도는 03의 pe.nii.gz와 같은 격자(같은 crop·1mm 리샘플)로 저장 → 같은 mask.nii.gz로 질감 추출 가능.
출력 (Data/Interim/<pid>/): washin.nii.gz, washout.nii.gz, auc.nii.gz, early_frac.nii.gz
       Data/Features/timecurve_features.csv (tc_*: 종양 내 지도별 분포 통계 + 곡선 유형 비율)
실행:  python 03e_timecurve.py [--limit N]
"""
from __future__ import annotations
import argparse
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
pp = import_module("03_preprocess")
MAPS = ("washin", "washout", "auc", "early_frac")


def _stats(prefix, v, o):
    o[f"{prefix}_mean"] = float(v.mean()); o[f"{prefix}_std"] = float(v.std())
    for q in (10, 50, 90):
        o[f"{prefix}_p{q}"] = float(np.percentile(v, q))


def tc_one(pid, r, spacing, idx, sitk):
    t_pre, n_t = int(r[cfg.COL_T_PRE]), int(r["n_times"])
    mask_f, err = pp._lookup(idx, cfg.MASK_PATTERN.format(pid=pid))
    if err:
        return None, "마스크" + err
    ref_pe = cfg.DATA_INTERIM / pid / "pe.nii.gz"
    if not ref_pe.exists():
        return None, "03 결과없음"
    files = {}
    for t in range(n_t):
        f, _ = pp._lookup(idx, cfg.DCE_TP_PATTERN.format(pid=pid, t=t))
        if f is not None:
            files[t] = f
    post = sorted(t for t in files if t > t_pre)
    if t_pre not in files or len(post) < 2:
        return None, f"시점부족({len(post)})"

    mask = sitk.Cast(sitk.ReadImage(str(mask_f), sitk.sitkUInt8) > 0, sitk.sitkUInt8)
    mask.SetSpacing(spacing); mask.SetOrigin((0.0, 0.0, 0.0)); mask.SetDirection(pp.IDENTITY)
    box = pp._crop_box(mask, spacing, sitk)                       # 03과 같은 crop
    crop = lambda im: sitk.RegionOfInterest(im, box[1], box[0])  # noqa: E731
    S = {t: crop(pp._read(files[t], spacing, sitk, sitk.sitkFloat32)) for t in [t_pre] + post}
    a0 = sitk.GetArrayFromImage(S[t_pre]); m = sitk.GetArrayFromImage(crop(mask)).astype(bool)
    ref = float(np.median(a0[m]))
    if not ref > 0:
        return None, "pre신호이상"
    den = np.maximum(a0, cfg.PE_FLOOR_FRAC * ref)
    PE = np.stack([100.0 * (sitk.GetArrayFromImage(S[t]) - a0) / den for t in post])    # (T, …)
    T = PE.shape[0]
    k = PE.argmax(0); peak = PE.max(0); ttp = (k + 1).astype(np.float32)               # 시점 번호 1..T
    last = PE[-1]
    maps = {
        "washin": peak / ttp,
        "washout": np.where(k < T - 1, (last - peak) / np.maximum(T - ttp, 1), 0.0),
        "auc": PE.mean(0),
        "early_frac": np.clip(PE[0] / np.maximum(peak, 1.0), -1.0, 1.5),
    }
    o = {}
    for name, a in maps.items():
        _stats(f"tc_{name}", a[m], o)
    # 곡선 유형 비율 (BI-RADS 지연기 유형을 전체 곡선으로): 유출 / 평탄 / 지속 상승
    rel = (last - peak) / np.maximum(peak, 1.0)
    o["tc_frac_washout"] = float((rel[m] < -0.10).mean())
    o["tc_frac_plateau"] = float(((rel[m] >= -0.10) & (k[m] < T - 1)).mean())
    o["tc_frac_persistent"] = float((k[m] == T - 1).mean())
    o["tc_ttp_mean_frac"] = float((ttp[m] / T).mean())          # 최고점 도달 시점의 상대 위치
    o["tc_n_post"] = T                                          # (QC용, 모델에선 제외)

    ref_img = sitk.ReadImage(str(ref_pe))                       # 03 pe 격자에 맞춤
    out = cfg.DATA_INTERIM / pid
    for name, a in maps.items():
        x = sitk.GetImageFromArray(a.astype(np.float32)); x.CopyInformation(S[t_pre])
        sitk.WriteImage(sitk.Resample(x, ref_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32),
                        str(out / f"{name}.nii.gz"))
    return o, "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sitk = pp._sitk()
    meta = pd.read_csv(cfg.METADATA_CSV)
    rows = meta.to_dict("records")[: args.limit or None]
    print("[i] 파일 색인 중 ...", flush=True)
    idx = pp._index(pp.EXTRACT_DIR)
    recs, qc = [], []
    for i, r in enumerate(rows, 1):
        pid = str(r[cfg.COL_PID])
        try:
            sp = (float(r[cfg.COL_SLICE_THICK]), float(r[cfg.COL_XY_SPACING]), float(r[cfg.COL_XY_SPACING]))
            o, st = tc_one(pid, r, sp, idx, sitk)
        except Exception as e:  # noqa: BLE001
            o, st = None, f"오류:{e}"
        qc.append({"pid": pid, "status": st, "n_post": (o or {}).get("tc_n_post")})
        if o:
            o.pop("tc_n_post", None)
            recs.append({cfg.COL_PID: pid, **o})
            if i % 100 == 0 or i <= 3:
                print(f"  [{i}/{len(rows)}] {pid} ✓ washin {o['tc_washin_mean']:.0f}%/시점 · "
                      f"유출형 {o['tc_frac_washout']:.0%}", flush=True)
        else:
            print(f"  [{i}/{len(rows)}] {pid} {st}")
    pd.DataFrame(qc).to_csv(cfg.EVAL_DIR / "03e_timecurve_qc.csv", index=False)
    df = pd.DataFrame(recs)
    df.to_csv(cfg.DATA_FEATURES / "timecurve_features.csv", index=False)
    print(f"\n[✓] timecurve_features.csv ({df.shape[0]}명 × {df.shape[1]-1}특징) · 결측 특징 {int(df.isna().any().sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
