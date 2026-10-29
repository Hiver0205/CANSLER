"""
03b_kinetic_features.py ─ DCE 동역학(kinetic) 특징 추출
=====================================================
radiomics(04)는 후기 PE 한 장만 본다. DCE의 핵심인 "조영제가 들어오고 빠지는 시간 패턴"을
I-SPY에서 검증된 지표로 요약해 추가한다. 모든 계산은 환자 내부(누수 없음), 원 해상도(mm 단위).

  PE_t   = 100·(S_t − S_0) / max(S_0, floor)        시점별 증강률 (%)
  FTV    = 종양 내 PE_early ≥ 70% 인 부피            (I-SPY functional tumor volume)
           ※ 이 데이터셋 마스크 자체가 FTV 기준으로 그려져 있음 → FTV = tum_vol, 특징에서 제외
  SER    = (S_early − S_0) / (S_late − S_0)          FTV 안에서만, [0, 3]로 제한
           >1.1 washout · 0.9–1.1 plateau · <0.9 persistent
  peak / TTP = 모든 조영후 시점 중 최대 PE와 그 시점 번호
  ring   = 종양 바깥 0–RING_MM (peritumoral)

입력:  Data/Interim/extracted (02), metadata (pre / post_early / post_late / n_times / spacing)
출력:  Data/Features/kinetic_features.csv  (pid × kin_*)
QC:    Evaluate/03b_kinetic_qc.csv

실행:  python 03b_kinetic_features.py --limit 5 / (전체)
"""
from __future__ import annotations
import argparse
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
pp = import_module("03_preprocess")          # 파일 색인·읽기·crop 헬퍼 재사용

RING_MM = 5.0
FTV_PE  = 70.0        # I-SPY FTV 임계 (%)


def _stats(prefix, v, out, qs=(10, 50, 90)):
    if v.size == 0:
        return
    out[f"{prefix}_mean"] = float(v.mean()); out[f"{prefix}_std"] = float(v.std())
    for q in qs:
        out[f"{prefix}_p{q}"] = float(np.percentile(v, q))


def kinetic_one(pid, r, spacing, idx, sitk):
    t_pre, t_e, t_l, n_t = (int(r[c]) for c in (cfg.COL_T_PRE, "post_early", cfg.COL_T_LATE, "n_times"))
    mask_f, err = pp._lookup(idx, cfg.MASK_PATTERN.format(pid=pid))
    if err:
        return None, "마스크" + err
    files = {}
    for t in range(n_t):
        f, err = pp._lookup(idx, cfg.DCE_TP_PATTERN.format(pid=pid, t=t))
        if f is not None:
            files[t] = f
        elif t in (t_pre, t_e, t_l):
            return None, f"t{t}" + err
    post_ts = sorted(t for t in files if t != t_pre)

    mask = sitk.Cast(sitk.ReadImage(str(mask_f), sitk.sitkUInt8) > 0, sitk.sitkUInt8)
    mask.SetSpacing(spacing); mask.SetOrigin((0.0, 0.0, 0.0)); mask.SetDirection(pp.IDENTITY)
    box = pp._crop_box(mask, spacing, sitk)
    if box is None:
        return None, "빈마스크"
    mask = sitk.RegionOfInterest(mask, box[1], box[0])
    rad = [max(1, int(round(RING_MM / s))) for s in spacing]
    ring_img = sitk.BinaryDilate(mask, rad, sitk.sitkBall)
    m = sitk.GetArrayFromImage(mask).astype(bool)
    ring = sitk.GetArrayFromImage(ring_img).astype(bool) & ~m

    S = {}
    for t, f in files.items():
        im = pp._read(f, spacing, sitk, sitk.sitkFloat32)
        S[t] =sitk.GetArrayFromImage(sitk.RegionOfInterest(im, box[1], box[0]))
    if any(a.shape != m.shape for a in S.values()):
        return None, "크기불일치"

    S0 = S[t_pre]; ref = float(np.median(S0[m]))
    if not ref > 0:
        return None, "pre신호이상"
    den = np.maximum(S0, cfg.PE_FLOOR_FRAC * ref)
    PE = {t: 100.0 * (S[t] - S0) / den for t in post_ts}
    pe_e, pe_l = PE[t_e], PE[t_l]
    stack = np.stack([PE[t] for t in post_ts])
    peak = stack.max(0); ttp = np.asarray(post_ts)[stack.argmax(0)]
    vox_cm3 = float(np.prod(spacing)) / 1000.0

    o = {}
    _stats("kin_pe_early", pe_e[m], o); _stats("kin_pe_late", pe_l[m], o)
    _stats("kin_peak", peak[m], o, qs=(50, 90))
    o["kin_pe_early_cv"] = float(pe_e[m].std() / (abs(pe_e[m].mean()) + 1e-6))
    o["kin_ttp_mean"] = float(ttp[m].mean())
    o["kin_frac_ttp_le_early"] = float((ttp[m] <= t_e).mean())          # 빠른 도달 비율

    # 주의: 이 데이터셋의 마스크는 PE_early ≥ 70% 로 만든 FTV 자체(5명 검사에서 ftv_frac=1.0)
    #   → FTV 부피 = tum_vol(05의 vol__) 이므로 별도 특징으로 넣지 않음. QC용으로만 기록.
    ftv = m & (pe_e >= FTV_PE)
    o["_qc_ftv_frac"] = float(ftv.sum() / m.sum())
    d_l = S[t_l] - S0
    valid = ftv & (d_l > 0)
    if valid.sum() >= 10:
        ser = np.clip((S[t_e] - S0)[valid] / d_l[valid], 0.0, 3.0)
        _stats("kin_ser", ser, o, qs=(50, 90))
        o["kin_frac_washout"] = float((ser > 1.1).mean())
        o["kin_frac_plateau"] = float(((ser >= 0.9) & (ser <= 1.1)).mean())
        o["kin_frac_persistent"] = float((ser < 0.9).mean())
        o["kin_washout_log_cm3"] = float(np.log1p((ser > 1.1).sum() * vox_cm3))

    if ring.sum() >= 10:
        _stats("kin_ring_pe_early", pe_e[ring], o, qs=(50, 90))
        o["kin_ring_pe_late_mean"] = float(pe_l[ring].mean())
        o["kin_ring_ftv_frac"] = float((pe_e[ring] >= FTV_PE).mean())
        o["kin_tumor_ring_ratio"] = float(pe_e[m].mean() / (abs(pe_e[ring].mean()) + 1.0))
    return o, "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg.ensure_dirs()
    sitk = pp._sitk()
    meta = pd.read_csv(cfg.METADATA_CSV)
    rows = meta.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    print(f"[i] 파일 색인 중 ...", flush=True)
    idx = pp._index(pp.EXTRACT_DIR)
    print(f"[i] 동역학 특징 대상 {len(rows)}명")

    recs, qc = [], []
    for i, r in enumerate(rows, 1):
        pid = str(r[cfg.COL_PID])
        try:
            spacing = (float(r[cfg.COL_SLICE_THICK]), float(r[cfg.COL_XY_SPACING]), float(r[cfg.COL_XY_SPACING]))
            o, st = kinetic_one(pid, r, spacing, idx, sitk)
        except Exception as e:  # noqa: BLE001
            o, st = None, f"오류:{e}"
        qc.append({"pid": pid, "status": st, "ftv_frac": (o or {}).get("_qc_ftv_frac")})
        if o is not None:
            o.pop("_qc_ftv_frac", None)
            recs.append({cfg.COL_PID: pid, **o})
            if i % 50 == 0 or i <= 3:
                print(f"  [{i}/{len(rows)}] {pid} ✓ PE_early {o['kin_pe_early_mean']:.0f}%"
                      f" · washout {o.get('kin_frac_washout', float('nan')):.0%}", flush=True)
        else:
            print(f"  [{i}/{len(rows)}] {pid} {st}")

    pd.DataFrame(qc).to_csv(cfg.EVAL_DIR / "03b_kinetic_qc.csv", index=False)
    if not recs:
        print("[!] 추출 0건"); return 2
    df = pd.DataFrame(recs)
    out = cfg.DATA_FEATURES / "kinetic_features.csv"
    df.to_csv(out, index=False)
    print(f"\n[✓] {out} ({df.shape[0]}명 × {df.shape[1]-1}특징) · 결측 있는 특징 "
          f"{int(df.isna().any().sum())} · QC: Evaluate/03b_kinetic_qc.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
