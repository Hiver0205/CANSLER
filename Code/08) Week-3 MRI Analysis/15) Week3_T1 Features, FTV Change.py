"""
10_t1_features.py ─ 실험 3: 치료 3주 차(T1) 영상 특징 + T0→T1 변화량
====================================================================
09가 받은 T1 DICOM(같은 I-SPY2 환자)을 읽어, T0와 **같은 정의**로 특징을 만든다. 환자 내부 계산(라벨 미사용).

  DCE   : TemporalPositionIdentifier로 시점 분리, 슬라이스 법선 방향 위치로 정렬, spacing은 DICOM에서
  마스크: TCIA "Analysis Mask"(DICOM SEG, 비트 부호화 역마스크) → 값 0 = FTV voxel (T0 마스크와 같은 정의)
          (1 PE임계 · 2 연결성 · 8 배경 · 32 VOI 밖 · 64 수동 제외)
  시점  : pre = 첫 시점. 초기·후기 = 이 환자 T0 metadata의 post_early/post_late를 T1 시점 수에 비례 환산
  특징 (T1, 원 해상도에서 종양 voxel 통계):
    t1v__ftv_log, t1v__ftv_pch (T0 대비 FTV 변화율 %), t1v__vanish (T1 FTV ≈ 0 → 조기 완전 반응 신호)
    t1c__* : 전체 곡선 통계 (03e와 같은 정의: washin·washout·auc·early_frac, 곡선 유형 비율)
    t1h__* : habitat 비율 (03f가 T0 개발셋으로 학습한 표준화·중심에 배정 → T0와 같은 유형)
    t1d__* : T1 − T0 변화량 (곡선 평균·유형 비율·habitat 비율)

입력:  Data/Interim/T1_dicom/<pid>/{dce,mask}/, metadata, Data/Features/{timecurve_features,habitat_features}.csv,
       Data/Features/habitat_model.npz
출력:  Data/Features/t1_features.csv, Evaluate/10_t1_qc.csv
실행:  python 10_t1_features.py [--limit N] [--workers 6]
"""
from __future__ import annotations
import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
T1DIR = cfg.DATA_INTERIM / "T1_dicom"
FEATS = ("washin", "washout", "auc", "early_frac")
MIN_VOX = 20


def _normal(ds):
    o = np.array(ds.ImageOrientationPatient, float)
    return np.cross(o[:3], o[3:])


def read_dce(d):
    import pydicom
    hs = [pydicom.dcmread(f) for f in glob.glob(str(d / "*.dcm"))]
    nrm = _normal(hs[0])
    pos = np.array([float(np.dot(np.array(h.ImagePositionPatient, float), nrm)) for h in hs])
    if all(hasattr(h, "TemporalPositionIdentifier") for h in hs):
        tp = np.array([int(h.TemporalPositionIdentifier) for h in hs])
    else:   # 대체: 같은 위치에서 InstanceNumber 순서 = 시점
        inst = np.array([int(h.InstanceNumber) for h in hs]); tp = np.zeros(len(hs), int)
        for p in np.unique(np.round(pos, 2)):
            k = np.where(np.round(pos, 2) == p)[0]; tp[k[np.argsort(inst[k])]] = np.arange(len(k))
    tps = np.unique(tp)
    vols, zs = [], None
    for t in tps:
        k = np.where(tp == t)[0]; k = k[np.argsort(pos[k])]
        vols.append(np.stack([hs[i].pixel_array.astype(np.float32) * float(getattr(hs[i], "RescaleSlope", 1))
                              + float(getattr(hs[i], "RescaleIntercept", 0)) for i in k]))
        if zs is None:
            zs = pos[k]
    n = min(v.shape[0] for v in vols)
    arr = np.stack([v[:n] for v in vols]); zs = zs[:n]
    dz = float(np.median(np.diff(zs))) if n > 1 else float(hs[0].SliceThickness)
    ps = [float(x) for x in hs[0].PixelSpacing]
    return arr, zs, (abs(dz), ps[0], ps[1]), nrm                      # 배열 (T, Z, Y, X), spacing (z, y, x)


def read_mask(d, zs, nrm, shape, dz):
    import pydicom
    s = pydicom.dcmread(glob.glob(str(d / "*.dcm"))[0])
    px = s.pixel_array
    if px.ndim == 2:
        px = px[None]
    fpos = np.array([float(np.dot(np.array(f.PlanePositionSequence[0].ImagePositionPatient, float), nrm))
                     for f in s.PerFrameFunctionalGroupsSequence])
    m = np.zeros(shape, bool); matched = 0
    for fi, p in enumerate(fpos):
        k = int(np.argmin(np.abs(zs - p)))
        if abs(zs[k] - p) <= 0.5 * dz + 1e-3 and px[fi].shape == shape[1:]:
            m[k] = px[fi] == 0                                          # 역마스크: 0 = FTV
            matched += 1
    return m, matched, len(fpos)


def _stats(prefix, v, o):
    o[f"{prefix}_mean"] = float(v.mean()); o[f"{prefix}_std"] = float(v.std())
    for q in (10, 50, 90):
        o[f"{prefix}_p{q}"] = float(np.percentile(v, q))


def one(pid, t0row, hab_model):
    q = {"pid": pid}
    try:
        arr, zs, sp, nrm = read_dce(T1DIR / pid / "dce")
        m, matched, nf = read_mask(T1DIR / pid / "mask", zs, nrm, arr.shape[1:], sp[0])
        T = arr.shape[0]; q.update(n_t=T, shape="x".join(map(str, arr.shape[1:])),
                                    spacing="x".join(f"{v:.3g}" for v in sp), mask_frames=f"{matched}/{nf}")
        if T < 3:
            q["status"] = f"시점부족({T})"; return None, q
        vox = float(np.prod(sp)) / 1000.0
        o = {}
        ftv1 = float(m.sum() * vox); ftv0 = float(t0row["tum_vol"]) / 1000.0
        o["t1v__ftv_log"] = float(np.log1p(ftv1))
        o["t1v__ftv_pch"] = float(np.clip((ftv1 - ftv0) / max(ftv0, 1e-3) * 100.0, -100.0, 300.0))
        o["t1v__vanish"] = float(m.sum() < MIN_VOX)
        q.update(ftv0_cm3=round(ftv0, 2), ftv1_cm3=round(ftv1, 2))
        if m.sum() < MIN_VOX:
            q["status"] = "OK(소실)"; return o, q
        # 초기·후기 시점: T0 인덱스를 시점 수 비례로 환산
        n0 = max(int(t0row["n_times"]), 2)
        e_i = int(np.clip(round(t0row["post_early"] * (T - 1) / (n0 - 1)), 1, T - 1))
        l_i = int(np.clip(round(t0row["post_late"] * (T - 1) / (n0 - 1)), e_i, T - 1))
        S0 = arr[0]; ref = float(np.median(S0[m]))
        if not ref > 0:
            q["status"] = "pre신호이상"; return None, q
        den = np.maximum(S0, cfg.PE_FLOOR_FRAC * ref)
        PE = np.stack([100.0 * (arr[t][m] - S0[m]) / den[m] for t in range(1, T)])   # (Tpost, Nvox)
        Tp = PE.shape[0]; k = PE.argmax(0); peak = PE.max(0); ttp = (k + 1).astype(np.float32); last = PE[-1]
        maps = {"washin": peak / ttp,
                "washout": np.where(k < Tp - 1, (last - peak) / np.maximum(Tp - ttp, 1), 0.0),
                "auc": PE.mean(0),
                "early_frac": np.clip(PE[0] / np.maximum(peak, 1.0), -1.0, 1.5)}
        for nm, a in maps.items():
            _stats(f"t1c__tc_{nm}", a, o)
        rel = (last - peak) / np.maximum(peak, 1.0)
        o["t1c__tc_frac_washout"] = float((rel < -0.10).mean())
        o["t1c__tc_frac_plateau"] = float(((rel >= -0.10) & (k < Tp - 1)).mean())
        o["t1c__tc_frac_persistent"] = float((k == Tp - 1).mean())
        o["t1c__tc_ttp_mean_frac"] = float((ttp / Tp).mean())
        o["t1c__pe_early_mean"] = float(PE[e_i - 1].mean()); o["t1c__pe_late_mean"] = float(PE[l_i - 1].mean())
        # habitat: T0 개발셋으로 학습한 표준화·중심에 배정
        X = np.stack([maps[f] for f in FEATS], 1)
        Z = (np.clip(X, hab_model["lo"], hab_model["hi"]) - hab_model["mu"]) / hab_model["sd"]
        for K in (3, 4):
            c = hab_model[f"centers{K}"]
            lab = np.argmin(((Z[:, None, :] - c[None]) ** 2).sum(-1), 1)
            for j, v in enumerate(np.bincount(lab, minlength=K) / len(lab)):
                o[f"t1h__hab{K}_{j}_frac"] = float(v)
        q["status"] = "OK"; q.update(early_idx=e_i, late_idx=l_i)
        return o, q
    except Exception as e:  # noqa: BLE001
        q["status"] = f"오류:{e}"; return None, q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_csv(cfg.METADATA_CSV); meta[cfg.COL_PID] = meta[cfg.COL_PID].astype(str)
    meta = meta.set_index(cfg.COL_PID)
    hm = dict(np.load(cfg.DATA_FEATURES / "habitat_model.npz"))
    pts = sorted(p.name for p in T1DIR.iterdir()
                 if (p / "dce" / ".done").exists() and (p / "mask" / ".done").exists() and p.name in meta.index)
    pts = pts[: args.limit or None]
    print(f"[i] T1 처리 대상 {len(pts)}명 · 작업자 {args.workers}", flush=True)
    recs, qc = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, p, meta.loc[p].to_dict(), hm): p for p in pts}
        for i, f in enumerate(as_completed(futs), 1):
            o, q = f.result(); qc.append(q)
            if o is not None:
                recs.append({cfg.COL_PID: futs[f], **o})
            if i % 50 == 0 or not q["status"].startswith("OK"):
                print(f"  [{i}/{len(pts)}] {q['pid']} {q['status']} · FTV {q.get('ftv0_cm3')}→{q.get('ftv1_cm3')}cm³"
                      f" · 마스크 프레임 {q.get('mask_frames')}", flush=True)
    df = pd.DataFrame(recs)
    # 변화량: T1 − T0 (곡선 평균·유형 비율, habitat 비율)
    tc0 = pd.read_csv(cfg.DATA_FEATURES / "timecurve_features.csv"); hb0 = pd.read_csv(cfg.DATA_FEATURES / "habitat_features.csv")
    base = tc0.merge(hb0, on=cfg.COL_PID); base[cfg.COL_PID] = base[cfg.COL_PID].astype(str)
    df = df.merge(base, on=cfg.COL_PID, how="left", suffixes=("", "_t0"))
    for c in [c for c in df if c.startswith("t1c__tc_") and (c.endswith("_mean") or "frac" in c)]:
        c0 = c[5:]
        if c0 in df:
            df[f"t1d__{c0}"] = df[c] - df[c0]
    for c in [c for c in df if c.startswith("t1h__")]:
        c0 = c[5:]
        if c0 in df:
            df[f"t1d__{c0}"] = df[c] - df[c0]
    keep = [cfg.COL_PID] + [c for c in df if c.startswith(("t1v__", "t1c__", "t1h__", "t1d__"))]
    df[keep].to_csv(cfg.DATA_FEATURES / "t1_features.csv", index=False)
    qd = pd.DataFrame(qc); qd.to_csv(cfg.EVAL_DIR / "10_t1_qc.csv", index=False)
    print("\n" + qd.status.str.split(":").str[0].value_counts().to_string())
    print(f"[✓] t1_features.csv ({len(df)}명 × {len(keep)-1}특징)")
    if "ftv0_cm3" in qd:
        ok = qd[qd.status.str.startswith("OK")]
        print(f"    FTV 변화율 중앙값 {np.median((ok.ftv1_cm3-ok.ftv0_cm3)/ok.ftv0_cm3*100):.0f}% · T1 소실 {int((ok.status=='OK(소실)').sum())}명")
    return 0


if __name__ == "__main__":
    sys.exit(main())
