"""
21_infer.py ─ 환자 1명 추론 (웹 도구용, 치료 전 모드 = ens_main)
==================================================================
입력: 환자의 치료 전 DCE 시점별 NIfTI, 종양 마스크, 시점 번호(pre/early/late), spacing, 임상 정보.
처리: 학습 때와 같은 함수(03·03b·03c·03e·03g·04)를 그대로 호출해 작업 폴더에 지도·특징을 만들고,
      habitat은 03f가 저장한 모델(habitat_model.npz)로 배정만 한다 (환자 간 학습 단계 없음).
예측: Models/ens_main_lr.joblib(로지스틱 회귀 14개) + Models/cnn_t6_tta_s3, cnn_t6_3d(CNN 2개) 확률 평균.

  from importlib import import_module
  inf = import_module("21_infer")
  res = inf.predict_pre(pid, dce={0: p0, 1: p1, ...}, mask=pm, t_pre=0, t_early=2, t_late=6,
                        spacing=(slice_thick, xy, xy), clinical={...}, work=Path("작업폴더"))

치료 3주 차 모드(ens_t3): predict_t3(... , w3_dce_dir, w3_mask_dir) — 3주 차 DCE DICOM 폴더와
      Analysis Mask(DICOM SEG) 폴더를 더 받는다. 10·10b의 함수로 T1 특징·지도를 만들고 로지스틱 회귀 11개 + T1 CNN.

확인:  python 21_infer.py --check 3            # 홀드아웃 환자 3명: 특징·확률이 기존 결과와 같은지
       python 21_infer.py --check 3 --mode t3  # 3주 차 모드
"""
from __future__ import annotations
import argparse
import sys
import tempfile
from importlib import import_module
from pathlib import Path
import torch  # noqa: F401  — Windows: SimpleITK·pyradiomics보다 먼저 불러와야 DLL 충돌(WinError 127)이 없음
import numpy as np
import pandas as pd

cfg = import_module("00_config")
# Windows: 저장 모델이 참조하는 06(sklearn)을 먼저 불러와야 OpenMP 런타임 충돌(libomp vs libiomp5md, OMP #15)이 없음.
#          웹(Streamlit) 등 다른 프로그램에서 이 모듈을 불러와도 같은 순서가 지켜지게 여기서 적재.
tr6 = import_module("06_train_cv")
pp = import_module("03_preprocess")
kb = import_module("03b_kinetic_features")
pc = import_module("03c_peritumoral")
te = import_module("03e_timecurve")
tg = import_module("03g_timestack")
rad = import_module("04_extract_radiomics")
hb = import_module("03f_habitat")
MODEL_DIR = cfg.PROJECT_ROOT / "Models"
MAIN_CNN = ("cnn_t6_tta_s3", "cnn_t6_3d")

# (지도, 접두어, binWidth, shape 포함) — 구현문서의 04 실행 명령과 같은 설정
RADIOMICS = [("pe", "dce", None, True), ("pe_early", "dceE", None, False), ("ser", "dceS", 0.05, False),
             ("washin", "tcW", 5, False), ("washout", "tcO", 2, False), ("auc", "tcA", 10, False),
             ("early_frac", "tcF", 0.05, False)]


class InferError(RuntimeError):
    """입력 영상이 처리 조건을 만족하지 않음 (웹에서 사용자 안내 문구로 보여 줌)."""


def _work_root(work):
    """학습 코드는 cfg.DATA_INTERIM/<pid>/ 를 읽고 쓰므로, 호출 동안 작업 폴더로 바꿔 둔다."""
    class _Swap:
        def __enter__(self):
            self.old = cfg.DATA_INTERIM; cfg.DATA_INTERIM = Path(work); return self

        def __exit__(self, *a):
            cfg.DATA_INTERIM = self.old
    return _Swap()


def _habitat(pid, sitk):
    """03f와 같은 특징을, 저장된 표준화 값·군집 중심에 배정만 해서 계산 (K=3, 4 = hab3_/hab4_)."""
    mdl = np.load(cfg.DATA_FEATURES / "habitat_model.npz")
    norm = lambda X: (np.clip(X, mdl["lo"], mdl["hi"]) - mdl["mu"]) / mdl["sd"]  # noqa: E731
    X, depth, m, m_img, vox = hb.load_pt(pid, sitk)
    o = {}
    for K in (3, 4):
        C = mdl[f"centers{K}"]                                 # 유형 번호 순으로 저장됨
        Z = norm(X)
        lab = np.argmin(((Z[:, None, :] - C[None]) ** 2).sum(-1), 1)
        lab_vol = np.zeros(m.shape, np.uint8); lab_vol[m] = lab + 1
        limg = sitk.GetImageFromArray(lab_vol); limg.CopyInformation(m_img)
        if K == hb.SAVE_LABEL_K:
            sitk.WriteImage(limg, str(cfg.DATA_INTERIM / pid / f"hab{K}.nii.gz"))
        for j in range(K):
            k = lab == j; t = f"hab{K}_{j}"
            o[f"{t}_frac"] = float(k.mean())
            o[f"{t}_log_cm3"] = float(np.log1p(k.sum() * vox))
            o[f"{t}_rim_frac"] = float((depth[k] <= hb.RIM_MM).mean()) if k.any() else np.nan
            o[f"{t}_depth_mm"] = float(depth[k].mean()) if k.any() else np.nan
            if k.sum() >= 10:
                cc = sitk.RelabelComponent(sitk.ConnectedComponent(limg == (j + 1)), minimumObjectSize=10)
                st = sitk.LabelShapeStatisticsImageFilter(); st.Execute(cc)
                o[f"{t}_n_comp"] = len(st.GetLabels())
            else:
                o[f"{t}_n_comp"] = 0
        p = np.bincount(lab, minlength=K) / len(lab)
        o[f"hab{K}_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return o


def _radiomics(d):
    o = {}
    for image, prefix, bw, shape in RADIOMICS:
        ext = rad._extractor(bw, no_shape=not shape); ext.settings["label"] = 1
        res = ext.execute(str(d / f"{image}.nii.gz"), str(d / "mask.nii.gz"), label=1)
        o.update({f"{prefix}_{k}": float(v) for k, v in res.items() if not k.startswith("diagnostics_")})
    return o


def _clinical(c, tum_vol_mm3):
    """05와 같은 열: clin__HR·HER2·age·race_white·race_black·menopause (0/1, 결측 NaN), vol__log_tum_vol."""
    o = {f"clin__{k}": (np.nan if c.get(k) is None else float(c[k])) for k in cfg.CLINICAL_COLS}
    o["vol__log_tum_vol"] = float(np.log1p(tum_vol_mm3 / 1000.0))
    return o


def _step(progress, frac, text):
    """진행률 알림 (웹 진행률 막대용). progress = f(0~1 비율, 단계 이름) 또는 None."""
    if progress is not None:
        progress(frac, text)


def extract_pre(pid, dce, mask, t_pre, t_early, t_late, spacing, clinical, work, progress=None):
    """치료 전 영상 → (특징 1행 dict, 작업 폴더). 실패하면 InferError."""
    sitk = pp._sitk()
    idx = {cfg.MASK_PATTERN.format(pid=pid): [Path(mask)]}
    idx.update({cfg.DCE_TP_PATTERN.format(pid=pid, t=int(t)): [Path(p)] for t, p in dce.items()})
    r = {cfg.COL_T_PRE: t_pre, cfg.COL_T_EARLY: t_early, cfg.COL_T_LATE: t_late, "n_times": max(dce) + 1}
    row = {}
    with _work_root(work):
        _step(progress, 0.05, "영상 정렬·전처리")
        q = pp.process_one(pid, t_pre, t_late, spacing, idx, sitk, t_early=t_early)
        if q["status"] != "OK":
            raise InferError(f"전처리 실패: {q['status']}")
        for i, (name, fn) in enumerate((("동역학", kb.kinetic_one), ("종양 주변부", pc.peri_one),
                                        ("시간 곡선", te.tc_one))):
            _step(progress, 0.15 + 0.12 * i, f"{name} 특징 계산")
            o, st = fn(pid, r, spacing, idx, sitk)
            if o is None:
                raise InferError(f"{name} 특징 실패: {st}")
            row.update(o[0] if isinstance(o, tuple) else o)
        row.pop("_qc_ftv_frac", None)
        _step(progress, 0.5, "CNN 입력 영상 생성")
        st = tg.stack_one(pid, r, spacing, idx, sitk)
        if st != "OK":
            raise InferError(f"CNN 입력 생성 실패: {st}")
        d = cfg.DATA_INTERIM / pid
        _step(progress, 0.6, "종양 내부 질감 분석")
        row.update(_radiomics(d))
        _step(progress, 0.72, "종양 내부 구성(habitat) 분류")
        row.update(_habitat(pid, sitk))
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(mask), sitk.sitkUInt8)) > 0
    row.update(_clinical(clinical, float(m.sum()) * float(np.prod(spacing))))
    return row, Path(work) / pid


_CNN = {}   # 체크포인트 경로 → (설정, 모델) — 한 번 읽은 모델은 메모리에 두고 다시 씀 (웹 속도)


def _cnn_logit(name, pid, work):
    """저장된 CNN(시드별)로 환자 logit — 04c train_predict의 예측 부분과 같은 계산 (시드 평균)."""
    import torch
    cn = import_module("04c_cnn_finetune")
    import SimpleITK as sitk  # noqa: N813
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logits = []
    for ck in sorted((MODEL_DIR / name).glob("seed*.pt")):
        if str(ck) not in _CNN:
            s = torch.load(ck, map_location=dev, weights_only=False)
            cn.CH_MODE, cn.DIM3, cn.MEAN, cn.STD = s["ch_mode"], s["dim3"], s["mean"], s["std"]
            mdl = cn.make_model(torch).to(dev); mdl.load_state_dict(s["state_dict"]); mdl.eval()
            _CNN[str(ck)] = ({k: v for k, v in s.items() if k != "state_dict"}, mdl)
        s, model = _CNN[str(ck)]
        cn.CH_MODE, cn.DIM3, cn.MEAN, cn.STD = s["ch_mode"], s["dim3"], s["mean"], s["std"]
        X, _ = cn.load_all([pid], s["size"], s["max_slices"], sitk, torch, root=Path(work))
        mean, std = torch.tensor(s["mean"], device=dev), torch.tensor(s["std"], device=dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            xb = X.to(dev).float()
            if s["tta"]:
                views = [torch.rot90(v, k, dims=(-2, -1)) for v in (xb, xb.flip(-1)) for k in range(4)]
                lg = torch.stack([model((v - mean) / std).squeeze(1).float() for v in views]).mean(0)
            else:
                lg = model((xb - mean) / std).squeeze(1).float()
        logits.append(float(lg.mean()))
    return float(np.mean(logits))


_BUNDLES = {}


def _bundle(name):
    import joblib
    if name not in _BUNDLES:
        _BUNDLES[name] = joblib.load(MODEL_DIR / f"{name}.joblib")
    return _BUNDLES[name]


def explain(row, bundle="ens_main_lr", n_models=16):
    """특징별 기여도(logit 단위) = 계수 × 표준화 값, 구성 모델 합 ÷ 앙상블 모델 수.
    로지스틱 회귀는 이 값이 근사가 아니라 정확한 분해. 양수 = pCR 가능성을 높임. L1이 고른 특징만 남는다."""
    tot = {}
    for m in _bundle(bundle)["members"].values():
        est = m["est"]; pipe = getattr(est, "best_estimator_", est)
        x = pd.DataFrame([{c: row.get(c, np.nan) for c in m["cols"]}])
        z = pipe.named_steps["prep"].transform(x)[0]
        names = pipe.named_steps["prep"].get_feature_names_out()
        for n, v, w in zip(names, z, pipe.named_steps["clf"].coef_[0]):
            k = n.split("__", 1)[1]
            tot[k] = tot.get(k, 0.0) + float(w * v)
    return pd.Series(tot) / n_models


def predict_row(row, bundle="ens_main_lr"):
    """특징 1행 → 구성 모델별 확률 (로지스틱 회귀)."""
    b = _bundle(bundle)
    out = {}
    for name, m in b["members"].items():
        x = pd.DataFrame([{c: row.get(c, np.nan) for c in m["cols"]}])
        out[name] = float(m["est"].predict_proba(x)[:, 1][0])
    return out


def predict_pre(pid, dce, mask, t_pre, t_early, t_late, spacing, clinical, work=None, progress=None):
    """치료 전 모드 최종 확률. 반환: {"prob", "members", "cnn", "features", "work"}."""
    work = Path(work or tempfile.mkdtemp(prefix="bpr_"))
    row, d = extract_pre(pid, dce, mask, t_pre, t_early, t_late, spacing, clinical, work, progress)
    _step(progress, 0.8, "예측 모델 계산 (로지스틱 회귀 14개 + CNN 2개)")
    members = predict_row(row)
    cnn = {n: float(1 / (1 + np.exp(-_cnn_logit(n, pid, work)))) for n in MAIN_CNN}
    _step(progress, 1.0, "완료")
    prob = float(np.mean(list(members.values()) + list(cnn.values())))
    return {"prob": prob, "members": members, "cnn": cnn, "features": row, "work": d}


T3_CNN = "cnn_t6_T1"


def extract_t1(pid, row0, t0row, dce_dir, mask_dir, work):
    """3주 차 DICOM(DCE 폴더, Analysis Mask SEG 폴더) → T1 특징(t1v/t1c/t1h/t1d/t1R_)과 CNN용 지도.
    10_t1_features.one, 10b_t1_maps.one을 그대로 호출 (읽는 위치만 작업 폴더로 바꿈)."""
    import shutil
    t1m, t1b = import_module("10_t1_features"), import_module("10b_t1_maps")
    dce_dir, mask_dir = Path(dce_dir), Path(mask_dir)
    if dce_dir.name == "dce" and dce_dir.parent.name == pid and mask_dir == dce_dir.parent / "mask":
        root = dce_dir.parent.parent            # 이미 <root>/<pid>/dce·mask 배치 → 복사 없이 그대로 읽음 (읽기만 함)
    else:
        root = Path(work) / "T1_dicom"
        for k, d in (("dce", dce_dir), ("mask", mask_dir)):
            shutil.copytree(d, root / pid / k, dirs_exist_ok=True)
    old = (t1m.T1DIR, t1b.OUT, t1m.read_dce)
    memo, rd = {}, t1m.read_dce

    def read_once(d):                           # 10·10b가 같은 DICOM을 두 번 읽지 않게
        if str(d) not in memo:
            memo[str(d)] = rd(d)
        return memo[str(d)]
    t1m.T1DIR, t1b.OUT, t1m.read_dce = root, Path(work) / "T1", read_once
    try:
        hm = dict(np.load(cfg.DATA_FEATURES / "habitat_model.npz"))
        o, q = t1m.one(pid, t0row, hm)
        if o is None:
            raise InferError(f"3주 차 특징 실패: {q['status']}")
        _, st = t1b.one(pid, t0row)
        if not st.startswith("OK"):
            raise InferError(f"3주 차 지도 생성 실패: {st}")
    finally:
        t1m.T1DIR, t1b.OUT, t1m.read_dce = old
    # 변화량 t1d__ = T1 − T0 (10_t1_features.main과 같은 규칙)
    for c in [c for c in o if c.startswith("t1c__tc_") and (c.endswith("_mean") or "frac" in c)]:
        if c[5:] in row0:
            o[f"t1d__{c[5:]}"] = o[c] - row0[c[5:]]
    for c in [c for c in o if c.startswith("t1h__")]:
        if c[5:] in row0:
            o[f"t1d__{c[5:]}"] = o[c] - row0[c[5:]]
    d = Path(work) / "T1" / pid
    ext = rad._extractor(None, no_shape=False); ext.settings["label"] = 1
    res = ext.execute(str(d / "pe.nii.gz"), str(d / "mask.nii.gz"), label=1)
    o.update({f"t1R_{k}": float(v) for k, v in res.items() if not k.startswith("diagnostics_")})
    return o, q


def predict_t3(pid, dce, mask, t_pre, t_early, t_late, spacing, clinical, w3_dce_dir, w3_mask_dir, work=None,
               progress=None, pre=None):
    """치료 3주 차 모드 최종 확률 (ens_t3 = 로지스틱 회귀 11개 + T1 CNN 1개).
    치료 전 확률(ens_main)과 3주 차 부피 변화도 함께 반환한다.
    pre = 같은 환자의 치료 전 결과(predict_pre 반환값)가 이미 있으면 다시 계산하지 않음."""
    work = Path(work or tempfile.mkdtemp(prefix="bpr_"))
    sub = (lambda f, t: progress(0.55 * f, "치료 전 MRI: " + t)) if progress else None
    if pre is None:
        pre = predict_pre(pid, dce, mask, t_pre, t_early, t_late, spacing, clinical, work, sub)
    _step(progress, 0.6, "3주 차 MRI 읽기·특징 계산")
    row0 = pre["features"]
    t0row = {"tum_vol": float(np.expm1(row0["vol__log_tum_vol"]) * 1000.0), "n_times": max(dce) + 1,
             "post_early": t_early, "post_late": t_late}
    o, q = extract_t1(pid, row0, t0row, w3_dce_dir, w3_mask_dir, work)
    row = {**row0, **o}
    _step(progress, 0.9, "예측 모델 계산 (로지스틱 회귀 11개 + CNN 1개)")
    members = predict_row(row, "ens_t3_lr")
    cnn = float(1 / (1 + np.exp(-_cnn_logit(T3_CNN, pid, work / "T1"))))
    _step(progress, 1.0, "완료")
    prob = float(np.mean(list(members.values()) + [cnn]))
    return {"prob": prob, "prob_pre": pre["prob"], "members": members, "cnn": cnn, "features": row,
            "ftv_pch": o.get("t1v__ftv_pch"), "ftv_cm3": (q.get("ftv0_cm3"), q.get("ftv1_cm3")), "work": work,
            "work_pre": pre["work"]}


def check_t3(n):
    """홀드아웃 중 T1 있는 환자 n명: 3주 차 모드 특징·확률이 08·04c 결과와 같은지."""
    import joblib
    meta = pd.read_csv(cfg.METADATA_CSV); meta[cfg.COL_PID] = meta[cfg.COL_PID].astype(str)
    ex3 = import_module("08_exp3_early_response")
    _, hold = ex3.load_full(); hold = hold.set_index(cfg.COL_PID)
    ref = pd.read_csv(cfg.EVAL_DIR / "08_exp3_holdout_predictions.csv", dtype={cfg.COL_PID: str}).set_index(cfg.COL_PID)
    cref = pd.read_csv(MODEL_DIR / T3_CNN / "holdout_logit.csv", dtype={cfg.COL_PID: str}).set_index(cfg.COL_PID)
    cols = sorted({c for m in joblib.load(MODEL_DIR / "ens_t3_lr.joblib")["members"].values() for c in m["cols"]})
    t1dir = cfg.DATA_INTERIM / "T1_dicom"
    print("[i] 원본 파일 색인 중 ...", flush=True)
    idx = pp._index(pp.EXTRACT_DIR)
    pids = [p for p in ref.index if (t1dir / p / "dce" / ".done").exists()][:n]
    for pid in pids:
        r = meta[meta[cfg.COL_PID] == pid].iloc[0]
        dce = {t: idx[cfg.DCE_TP_PATTERN.format(pid=pid, t=t)][0] for t in range(int(r["n_times"]))
               if cfg.DCE_TP_PATTERN.format(pid=pid, t=t) in idx}
        clin = {k: (None if pd.isna(r[k]) else r[k]) for k in cfg.CLINICAL_COLS}
        sp = (float(r[cfg.COL_SLICE_THICK]), float(r[cfg.COL_XY_SPACING]), float(r[cfg.COL_XY_SPACING]))
        with tempfile.TemporaryDirectory(prefix="bpr_chk_") as w:
            res = predict_t3(pid, dce, idx[cfg.MASK_PATTERN.format(pid=pid)][0], r[cfg.COL_T_PRE], r[cfg.COL_T_EARLY],
                             r[cfg.COL_T_LATE], sp, clin, t1dir / pid / "dce", t1dir / pid / "mask", w)
        f = res["features"]
        a = np.array([f.get(c, np.nan) for c in cols], float); b = hold.loc[pid, cols].to_numpy(float)
        both = ~(np.isnan(a) | np.isnan(b))
        rel = np.abs(a[both] - b[both]) / (np.abs(b[both]) + 1e-9)
        worst = cols[int(np.flatnonzero(both)[np.argmax(rel)])]
        dl = max(abs(res["members"][m] - ref.loc[pid, f"prob__{m}"]) for m in res["members"])
        dc = abs(np.log(res["cnn"] / (1 - res["cnn"])) - cref.loc[pid, "cnn_logit"])
        print(f"  {pid}: 특징 {len(cols)}개 최대 상대차 {rel.max():.1e} ({worst}) · 결측 불일치 "
              f"{int((np.isnan(a) != np.isnan(b)).sum())} · LR 최대차 {dl:.1e} · CNN logit 차 {dc:.1e}"
              f" · 치료 전 {res['prob_pre']:.3f} → 3주 차 {res['prob']:.3f} · 부피 변화 {res['ftv_pch']:+.0f}%", flush=True)
    return 0


def check(n):
    """홀드아웃 환자 n명을 원본 파일에서 처음부터 다시 계산 → 기존 특징·확률과 비교."""
    import joblib
    meta = pd.read_csv(cfg.METADATA_CSV); meta[cfg.COL_PID] = meta[cfg.COL_PID].astype(str)
    tab = pd.read_csv(cfg.DATA_FEATURES / "analysis_table.csv"); tab[cfg.COL_PID] = tab[cfg.COL_PID].astype(str)
    tab = tab.set_index(cfg.COL_PID)
    ref = pd.read_csv(cfg.EVAL_DIR / "11_holdout_predictions_main.csv", dtype={cfg.COL_PID: str}).set_index(cfg.COL_PID)
    cnn_ref = {k: pd.read_csv(MODEL_DIR / k / "holdout_logit.csv", dtype={cfg.COL_PID: str}).set_index(cfg.COL_PID)
               for k in MAIN_CNN}
    cols = sorted({c for m in joblib.load(MODEL_DIR / "ens_main_lr.joblib")["members"].values() for c in m["cols"]})
    print("[i] 원본 파일 색인 중 ...", flush=True)
    idx = pp._index(pp.EXTRACT_DIR)
    for pid in list(ref.index[:n]):
        r = meta[meta[cfg.COL_PID] == pid].iloc[0]
        nt = int(r["n_times"])
        dce = {t: idx[cfg.DCE_TP_PATTERN.format(pid=pid, t=t)][0] for t in range(nt)
               if cfg.DCE_TP_PATTERN.format(pid=pid, t=t) in idx}
        mask = idx[cfg.MASK_PATTERN.format(pid=pid)][0]
        clin = {k: (None if pd.isna(r[k]) else r[k]) for k in cfg.CLINICAL_COLS}
        sp = (float(r[cfg.COL_SLICE_THICK]), float(r[cfg.COL_XY_SPACING]), float(r[cfg.COL_XY_SPACING]))
        with tempfile.TemporaryDirectory(prefix="bpr_chk_") as w:
            res = predict_pre(pid, dce, mask, r[cfg.COL_T_PRE], r[cfg.COL_T_EARLY], r[cfg.COL_T_LATE], sp, clin, w)
        f = res["features"]
        a = np.array([f.get(c, np.nan) for c in cols], float); b = tab.loc[pid, cols].to_numpy(float)
        both = ~(np.isnan(a) | np.isnan(b))
        rel = np.abs(a[both] - b[both]) / (np.abs(b[both]) + 1e-9)
        nan_mis = int((np.isnan(a) != np.isnan(b)).sum())
        worst = cols[int(np.flatnonzero(both)[np.argmax(rel)])] if rel.size else "-"
        dl = max(abs(res["members"][m] - ref.loc[pid, f"prob__{m}"]) for m in res["members"])
        dc = max(abs(np.log(p / (1 - p)) - cnn_ref[k].loc[pid, "cnn_logit"]) for k, p in res["cnn"].items())
        print(f"  {pid}: 특징 {len(cols)}개 최대 상대차 {rel.max():.1e} ({worst}) · 결측 불일치 {nan_mis}"
              f" · LR 확률 최대차 {dl:.1e} · CNN logit 최대차 {dc:.1e} · 최종 {res['prob']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=3)
    ap.add_argument("--mode", choices=["pre", "t3"], default="pre")
    a = ap.parse_args()
    sys.exit(check(a.check) if a.mode == "pre" else check_t3(a.check))
