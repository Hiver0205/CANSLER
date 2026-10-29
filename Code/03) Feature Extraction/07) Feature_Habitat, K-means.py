"""
03f_habitat.py ─ J4: 시간-증강 곡선 기반 종양 내부 하위 영역(habitat)
=====================================================================
종양 voxel을 곡선 모양 4특징(유입·유출·면적·초기비율, 03e)으로 K개 유형으로 군집 →
유형별 비율·부피·위치(가장자리 vs 중심)·조각 수를 환자 특징으로.

누수 관리: 군집은 라벨을 쓰지 않는 비지도 학습이고, 표준화·군집 중심은 **개발셋 환자 voxel로만** 학습.
          홀드아웃 환자는 학습된 중심에 배정만 됨.

입력:  Data/Interim/<pid>/{washin,washout,auc,early_frac,mask}.nii.gz  (03e·03, 1mm 등방, 같은 격자)
출력:  Data/Features/habitat_features.csv  (hab3_*, hab4_* / J5 확장: hbx5_*, hbx6_*, hbx{K}_mixing·adj_*)
       Data/Interim/<pid>/hab3.nii.gz        (K=3 유형 지도 → 04 --mask hab3 --label j 로 유형별 질감)
       Evaluate/03f_habitat_centers.csv     (유형별 중심 — 해석용)
실행:  python 03f_habitat.py
"""
from __future__ import annotations
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
FEATS = ("washin", "washout", "auc", "early_frac")
KS = (3, 4, 5, 6)
RIM_MM = 3.0
SAMPLE_PER_PT = 2000
# 접두어: K=3,4 기본 특징은 기존 그대로 "hab{K}_" (기존 모델 정의 불변),
#         J5 확장(K=5,6 기본 특징 + 모든 K의 경계·섞임)은 "hbx{K}_"
BASE_PREFIX = lambda K: "hab" if K in (3, 4) else "hbx"  # noqa: E731
SAVE_LABEL_K = 3          # 이 K의 유형 지도를 저장 → 04로 유형별 질감 (hab3.nii.gz)


def adjacency(lab_vol, K):
    """종양 내부 6-이웃 voxel 쌍 중 서로 다른 유형이 맞닿은 비율. 반환: (섞임 지수, {(i,j): 비율})."""
    P = np.zeros((K + 1, K + 1))
    for ax in range(3):
        a = np.moveaxis(lab_vol, ax, 0)
        x, y = a[:-1].ravel(), a[1:].ravel()
        ok = (x > 0) & (y > 0)
        np.add.at(P, (np.minimum(x[ok], y[ok]), np.maximum(x[ok], y[ok])), 1)
    total = P.sum()
    if total == 0:
        return np.nan, {}
    pairs = {(i - 1, j - 1): P[i, j] / total for i in range(1, K + 1) for j in range(i + 1, K + 1)}
    return float(sum(pairs.values())), pairs


def load_pt(pid, sitk):
    d = cfg.DATA_INTERIM / pid
    m_img = sitk.ReadImage(str(d / "mask.nii.gz"), sitk.sitkUInt8) > 0
    m = sitk.GetArrayFromImage(m_img).astype(bool)
    X = np.stack([sitk.GetArrayFromImage(sitk.ReadImage(str(d / f"{f}.nii.gz")))[m] for f in FEATS], 1)
    env = sitk.BinaryFillhole(sitk.BinaryMorphologicalClosing(sitk.Cast(m_img, sitk.sitkUInt8), [3, 3, 3]))
    depth = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        env, insideIsPositive=True, squaredDistance=False, useImageSpacing=True))[m]   # 외곽선까지 깊이(mm)
    vox_cm3 = float(np.prod(m_img.GetSpacing())) / 1000.0
    return X.astype(np.float32), depth, m, m_img, vox_cm3


def main() -> int:
    import SimpleITK as sitk  # noqa: N813
    from sklearn.cluster import KMeans
    meta = pd.read_csv(cfg.METADATA_CSV)
    meta["pid"] = meta[cfg.COL_PID].astype(str)
    hold = set(meta.loc[meta[cfg.COL_SPLIT].isin(cfg.HOLDOUT_SPLITS), "pid"])
    pts = [p for p in meta.pid if all((cfg.DATA_INTERIM / p / f"{f}.nii.gz").exists() for f in FEATS + ("mask",))]
    print(f"[i] 대상 {len(pts)}명 (군집 학습은 개발셋 {sum(p not in hold for p in pts)}명 voxel만)", flush=True)

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    data, samp = {}, []
    for i, pid in enumerate(pts, 1):
        data[pid] = load_pt(pid, sitk)
        if pid not in hold:
            X = data[pid][0]
            samp.append(X[rng.choice(len(X), min(SAMPLE_PER_PT, len(X)), replace=False)])
        if i % 200 == 0:
            print(f"  적재 {i}/{len(pts)}", flush=True)
    S = np.concatenate(samp)
    lo, hi = np.percentile(S, 1, 0), np.percentile(S, 99, 0)           # 극단값 클리핑 범위 (개발셋)
    Sc = np.clip(S, lo, hi); mu, sd = Sc.mean(0), Sc.std(0) + 1e-6
    norm = lambda X: (np.clip(X, lo, hi) - mu) / sd                    # noqa: E731

    rows = {p: {cfg.COL_PID: p} for p in pts}
    centers = []
    saved = {"lo": lo, "hi": hi, "mu": mu, "sd": sd}                      # 다른 방문(T1) 배정용 모델 저장
    for K in KS:
        km = KMeans(K, n_init=10, random_state=cfg.RANDOM_STATE).fit(norm(S))
        order = np.argsort(km.cluster_centers_[:, 0])                     # 유입 속도 낮은 순으로 유형 번호
        remap = np.empty(K, int); remap[order] = np.arange(K)
        saved[f"centers{K}"] = km.cluster_centers_[order]                 # 표준화 공간, 유형 번호 순
        cen = km.cluster_centers_[order] * sd + mu
        for j, c in enumerate(cen):
            centers.append({"K": K, "habitat": j, **{f: float(v) for f, v in zip(FEATS, c)}})
        for pid in pts:
            X, depth, m, m_img, vox = data[pid]
            lab = remap[km.predict(norm(X))]
            o = rows[pid]; n = len(lab)
            lab_vol = np.zeros(m.shape, np.uint8); lab_vol[m] = lab + 1
            limg = sitk.GetImageFromArray(lab_vol); limg.CopyInformation(m_img)
            if K == SAVE_LABEL_K:
                sitk.WriteImage(limg, str(cfg.DATA_INTERIM / pid / f"hab{K}.nii.gz"))
            mix, pairs = adjacency(lab_vol, K)                              # J5: 유형 간 경계
            o[f"hbx{K}_mixing"] = mix
            for (i, j), v in pairs.items():
                o[f"hbx{K}_adj_{i}_{j}"] = float(v)
            pf = BASE_PREFIX(K)
            for j in range(K):
                k = lab == j; t = f"{pf}{K}_{j}"
                o[f"{t}_frac"] = float(k.mean())
                o[f"{t}_log_cm3"] = float(np.log1p(k.sum() * vox))
                o[f"{t}_rim_frac"] = float((depth[k] <= RIM_MM).mean()) if k.any() else np.nan   # 가장자리 비율
                o[f"{t}_depth_mm"] = float(depth[k].mean()) if k.any() else np.nan
                if k.sum() >= 10:
                    cc = sitk.RelabelComponent(sitk.ConnectedComponent(limg == (j + 1)), minimumObjectSize=10)
                    st = sitk.LabelShapeStatisticsImageFilter(); st.Execute(cc)
                    o[f"{t}_n_comp"] = len(st.GetLabels())
                else:
                    o[f"{t}_n_comp"] = 0
            p = np.bincount(lab, minlength=K) / n                         # 유형 다양성(엔트로피)
            o[f"{pf}{K}_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        print(f"[i] K={K} 완료", flush=True)

    np.savez(cfg.DATA_FEATURES / "habitat_model.npz", feats=np.array(FEATS), **saved)
    df = pd.DataFrame(rows.values())
    df.to_csv(cfg.DATA_FEATURES / "habitat_features.csv", index=False)
    cdf = pd.DataFrame(centers); cdf.to_csv(cfg.EVAL_DIR / "03f_habitat_centers.csv", index=False)
    print(f"\n[✓] habitat_features.csv ({df.shape[0]}명 × {df.shape[1]-1}특징)")
    print("[유형 중심 (원 단위)]\n" + cdf.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
