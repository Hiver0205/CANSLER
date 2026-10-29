"""
08_exp3_early_response.py ─ 실험 3: 치료 초기(3주) 반응 영상을 더하면 어디까지 오르나
=====================================================================================
주 연구(치료 전 영상, 실험 1·2)와 별개의 **추가 분석**. 연구 질문이 "치료 전 예측" → "치료 초기 반응 예측"으로
바뀌므로 보고서에서 반드시 구분한다.

데이터: TCIA I-SPY2 "Multi-feature MRI NACT Data.xlsx" (384명, T0~T3 영상이 모두 있는 환자; CC BY 4.0)
  사용: T0·T1 값과 T0→T1 변화율만 (FTV, 최장직경 LD, 배경실질증강 BPE, 구형도)
  제외: T2(치료 중간)·T3(수술 직전) — 결과를 거의 직접 보여주므로 예측이라 보기 어려움
  컬럼 규약(파일): VOLUME_TUM_BLU_V10 = T0 FTV, V20 = T1 FTV, *_pch_T0_T1 = T0→T1 변화율(%)
대상: 384명 ∩ 개발셋 (홀드아웃은 사용하지 않음)
방법: 06과 같은 파이프라인 (임상·T1 변수는 항상 유지, 치료 전 영상 특징은 L1 선택), 5-fold × 20회, 같은 분할 공유

입력:  Data/Features/analysis_table.csv (05), Data/Original/tcia_clinical/Multi-feature-MRI-NACT-Data.xlsx
출력:  Evaluate/08_exp3_auc.csv, 08_exp3_delta.csv
실행:  python 08_exp3_early_response.py
"""
from __future__ import annotations
import sys
import warnings
from importlib import import_module
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
cfg = import_module("00_config")
tr6 = import_module("06_train_cv")
ev = import_module("07_evaluate")

MF = cfg.DATA_ORIGINAL / "tcia_clinical" / "Multi-feature-MRI-NACT-Data.xlsx"
T1_COLS = {   # 파일 열 → 분석 변수 (T0·T1만)
    "VOLUME_TUM_BLU_V10": "t1__ftv_T0_log", "VOLUME_TUM_BLU_V20": "t1__ftv_T1_log",
    "SPHERICITY_T0": "t1__sph_T0", "SPHERICITY_T1": "t1__sph_T1",
    "LD_T0": "t1__ld_T0", "LD_T1": "t1__ld_T1",
    "BPE_5slice_mean_T0": "t1__bpe_T0", "BPE_5slice_mean_T1": "t1__bpe_T1",
    "FTV_pch_T0_T1": "t1__ftv_pch", "LD_pch_T0_T1": "t1__ld_pch",
    "BPE_pch_T0_T1": "t1__bpe_pch", "Sphericity_pch_T0_T1": "t1__sph_pch",
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["summary", "full"], default="summary",
                    help="summary = TCIA 요약 파일(384명) / full = T1 원본에서 직접 계산(10_t1_features, ~942명)")
    ap.add_argument("--final", action="store_true", help="full 전용: 개발셋 전체 학습 → 홀드아웃 1회 예측 (CV 없음)")
    args = ap.parse_args()
    if args.source == "full":
        return main_full(final=args.final)
    df = pd.read_csv(cfg.DATA_FEATURES / "analysis_table.csv"); df[cfg.COL_PID] = df[cfg.COL_PID].astype(str)
    mf = pd.read_excel(MF)
    mf["num"] = mf["CLINICAL-TRIAL-SUBJECT-ID"].astype(int)
    t1 = mf[["num"] + list(T1_COLS)].rename(columns=T1_COLS)
    for c in ("t1__ftv_T0_log", "t1__ftv_T1_log"):
        t1[c] = np.log1p(t1[c].clip(lower=0))
    df["num"] = df[cfg.COL_PID].str.split("-").str[-1].astype(int)
    d = df.merge(t1, on="num", how="inner")
    n_all = len(d)
    d = d[~d["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)      # 홀드아웃 제외
    y, groups = d["pCR"].to_numpy().astype(int), d[cfg.COL_PID].to_numpy()
    print(f"[i] T1 자료 매칭 {n_all}명 → 개발셋 {len(d)}명 (pCR {y.mean():.1%}) · 홀드아웃 {n_all-len(d)}명은 미사용")
    print("[i] T1 변수 결측: " + ", ".join(f"{c[4:]} {int(d[c].isna().sum())}" for c in T1_COLS.values()
                                          if d[c].isna().any()) or "없음")

    clin, vol, t1c = tr6.cols_of(d, ["clin__"]), tr6.cols_of(d, ["vol__"]), list(T1_COLS.values())
    pre_img = tr6.cols_of(d, [tr6.TC, tr6.HAB])                              # 치료 전 최고 단일 조합
    specs = {
        "a_clinical":         (clin, []),
        "b_clin_preimg":      (clin + vol, pre_img),
        "c_clin_t1":          (clin + t1c, []),
        "d_clin_preimg_t1":   (clin + vol + t1c, pre_img),
        "e_t1_only":          (t1c, []),
    }
    rows, frows = [], []
    for r in range(cfg.CV_REPEATS):
        cv = StratifiedGroupKFold(cfg.CV_FOLDS, shuffle=True, random_state=r)
        for f, (tr, te) in enumerate(cv.split(d, y, groups)):
            for name, (base, im) in specs.items():
                est = tr6.build(base, im, inner_seed=1000 * r + f)
                est.fit(d.iloc[tr][base + im], y[tr])
                p = est.predict_proba(d.iloc[te][base + im])[:, 1]
                rows += [{"pid": pid, "repeat": r, "model": name, "prob": v} for pid, v in zip(groups[te], p)]
                frows.append({"repeat": r, "fold": f, "model": name, "n_train": len(tr), "n_test": len(te),
                              "auc": roc_auc_score(y[te], p)})
        print(f"  반복 {r+1}/{cfg.CV_REPEATS}", flush=True)

    L = pd.DataFrame(rows); F = pd.DataFrame(frows); ymap = dict(zip(groups, y))
    per = L.groupby(["model", "repeat"]).apply(lambda g: roc_auc_score(g.pid.map(ymap), g.prob),
                                               include_groups=False).unstack(0)
    print(f"\n=== 실험 3: 치료 초기 반응 (개발셋 {len(d)}명, 5-fold × {cfg.CV_REPEATS}회) ===")
    out = []
    for m in specs:
        a = per[m].to_numpy()
        out.append({"model": m, "auc_mean": a.mean(), "auc_p2.5": np.percentile(a, 2.5),
                    "auc_p97.5": np.percentile(a, 97.5)})
        print(f"  {m:20s} AUC {a.mean():.3f} [{np.percentile(a, 2.5):.3f}–{np.percentile(a, 97.5):.3f}]")
    pd.DataFrame(out).to_csv(cfg.EVAL_DIR / "08_exp3_auc.csv", index=False)

    wide = F.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    dl = []
    for a, b, desc in [("a_clinical", "b_clin_preimg", "임상 → +치료 전 영상"),
                       ("a_clinical", "c_clin_t1", "임상 → +T1 반응"),
                       ("b_clin_preimg", "d_clin_preimg_t1", "임상+치료 전 영상 → +T1 반응"),
                       ("a_clinical", "d_clin_preimg_t1", "임상 → +치료 전 영상+T1 반응")]:
        dr = (per[b] - per[a]).to_numpy()
        pv = ev.corrected_ttest((wide[b] - wide[a]).to_numpy(), F.n_train.mean(), F.n_test.mean())
        dl.append({"desc": desc, "delta_mean": dr.mean(), "frac_pos": (dr > 0).mean(), "p": pv})
        print(f"  {desc:28s} ΔAUC {dr.mean():+.3f} · 양수 반복 {(dr > 0).mean():.0%} · p={pv:.4f} {ev._star(pv)}")
    pd.DataFrame(dl).to_csv(cfg.EVAL_DIR / "08_exp3_delta.csv", index=False)

    sub = dict(zip(groups, d["subtype"])); L["st"] = L.pid.map(sub); L["y"] = L.pid.map(ymap)
    print("\n[아형별 AUC (반복 평균)]")
    for m in specs:
        g = L[L.model == m]
        txt = " · ".join(f"{cfg.SUBTYPE_LABEL.get(s, s)} "
                         f"{np.mean([roc_auc_score(x.y, x.prob) for _, x in gs.groupby('repeat') if x.y.nunique()==2]):.3f}"
                         f"(n={gs.pid.nunique()})" for s, gs in g.groupby("st"))
        print(f"  {m:20s} {txt}")
    return 0


def run_cv(d, specs, tag):
    """06과 같은 반복 nested CV로 specs 비교 → AUC 요약·ΔAUC·아형별 출력, 저장 접미사 tag."""
    y, groups = d["pCR"].to_numpy().astype(int), d[cfg.COL_PID].to_numpy()
    rows, frows = [], []
    for r in range(cfg.CV_REPEATS):
        cv = StratifiedGroupKFold(cfg.CV_FOLDS, shuffle=True, random_state=r)
        for f, (tr, te) in enumerate(cv.split(d, y, groups)):
            for name, (base, im) in specs.items():
                est = tr6.build(base, im, inner_seed=1000 * r + f)
                est.fit(d.iloc[tr][base + im], y[tr])
                p = est.predict_proba(d.iloc[te][base + im])[:, 1]
                rows += [{"pid": pid, "repeat": r, "fold": f, "model": name, "prob": v}
                         for pid, v in zip(groups[te], p)]
                frows.append({"repeat": r, "fold": f, "model": name, "n_train": len(tr), "n_test": len(te),
                              "auc": roc_auc_score(y[te], p)})
        print(f"  반복 {r+1}/{cfg.CV_REPEATS}", flush=True)
    L = pd.DataFrame(rows); F = pd.DataFrame(frows); ymap = dict(zip(groups, y))
    L.to_csv(cfg.EVAL_DIR / f"08_exp3_oof_{tag}.csv", index=False)
    per = L.groupby(["model", "repeat"]).apply(lambda g: roc_auc_score(g.pid.map(ymap), g.prob),
                                               include_groups=False).unstack(0)
    out = []
    for m in specs:
        a = per[m].to_numpy()
        out.append({"model": m, "auc_mean": a.mean(), "auc_p2.5": np.percentile(a, 2.5),
                    "auc_p97.5": np.percentile(a, 97.5)})
        print(f"  {m:24s} AUC {a.mean():.3f} [{np.percentile(a, 2.5):.3f}–{np.percentile(a, 97.5):.3f}]")
    pd.DataFrame(out).to_csv(cfg.EVAL_DIR / f"08_exp3_auc_{tag}.csv", index=False)
    return L, F, per, ymap


def main_full(final=False) -> int:
    """T1 원본 기반 (10_t1_features): 개발셋 중 T1 있는 환자 전원. final=True면 홀드아웃 1회 예측만."""
    df = pd.read_csv(cfg.DATA_FEATURES / "analysis_table.csv"); df[cfg.COL_PID] = df[cfg.COL_PID].astype(str)
    t1 = pd.read_csv(cfg.DATA_FEATURES / "t1_features.csv"); t1[cfg.COL_PID] = t1[cfg.COL_PID].astype(str)
    d = df.merge(t1, on=cfg.COL_PID, how="inner")
    rp = cfg.DATA_FEATURES / "radiomics_t1.csv"                              # L3: T1 지도 질감 (04 --root T1)
    if rp.exists():
        r1 = pd.read_csv(rp); r1[cfg.COL_PID] = r1[cfg.COL_PID].astype(str)
        d = d.merge(r1, on=cfg.COL_PID, how="left")
    n_all = len(d)
    hold = d[d["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)
    d = d[~d["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)       # 홀드아웃 제외
    print(f"[i] T1 원본 특징 {n_all}명 → 개발셋 {len(d)}명 (pCR {d.pCR.mean():.1%}) · 홀드아웃 {n_all-len(d)}명 미사용")
    clin, vol = tr6.cols_of(d, ["clin__"]), tr6.cols_of(d, ["vol__"])
    pre = tr6.cols_of(d, [tr6.TC, tr6.HAB])                                   # 치료 전 최고 단일 조합
    t1v = tr6.cols_of(d, ["t1v__"])                                            # FTV 변화 (항상 유지)
    t1x = tr6.cols_of(d, ["t1c__", "t1h__", "t1d__"])                          # T1 곡선·habitat·변화량 (L1)
    specs = {
        "a_clinical":          (clin, []),
        "b_clin_preimg":       (clin + vol, pre),
        "c_clin_ftvchange":    (clin + vol + t1v, []),
        "d_clin_t1full":       (clin + vol + t1v, t1x),
        "e_clin_preimg_t1full": (clin + vol + t1v, pre + t1x),
        "f_t1_only":           (vol + t1v, t1x),
    }
    # L1: T1 관점 앙상블 — "임상+FTV변화+T1 특징" 위에 치료 전 관점을 하나씩 붙인 모델들 (평균은 아래)
    t1rad = tr6.cols_of(d, ["t1R_"])                                          # L3: T1 지도 질감 (있으면)
    views = {"v_kin": [tr6.KIN], "v_tc": [tr6.TC], "v_hab": [tr6.HAB], "v_rad": [tr6.RAD], "v_peri": [tr6.PERI],
             "v_bpe": [tr6.BPE], "v_tctex": tr6.TCX, "v_es": [tr6.RAD_E, tr6.RAD_S]}
    for k, pf in views.items():
        specs[f"l1_{k}"] = (clin + vol + t1v, t1x + tr6.cols_of(d, pf))
    if t1rad:
        specs["l3_t1rad"] = (clin + vol + t1v, t1x + t1rad)
    ENS = {"ens_t1": ["d_clin_t1full", "e_clin_preimg_t1full"] + [f"l1_{k}" for k in views],
           "ens_t1_rad": ["d_clin_t1full", "e_clin_preimg_t1full"] + [f"l1_{k}" for k in views]
                         + (["l3_t1rad"] if t1rad else [])}
    print(f"[i] 특징: 치료 전 {len(pre)} · T1 FTV {len(t1v)} · T1 곡선/habitat/변화 {len(t1x)}")
    if final:   # 개발셋 전체 학습 → 홀드아웃 1회 예측 (안쪽 CV 튜닝은 build()가 개발셋 안에서 수행)
        y = d["pCR"].to_numpy().astype(int)
        hp = hold[[cfg.COL_PID, "pCR", "subtype"]].copy()
        for name, (base, im) in specs.items():
            est = tr6.build(base, im, inner_seed=cfg.RANDOM_STATE).fit(d[base + im], y)
            hp[f"prob__{name}"] = est.predict_proba(hold[base + im])[:, 1]
        for ens, mem in ENS.items():
            hp[f"prob__{ens}"] = hp[[f"prob__{m}" for m in mem]].mean(axis=1)
        hp.to_csv(cfg.EVAL_DIR / "08_exp3_holdout_predictions.csv", index=False)
        print(f"[✓] 홀드아웃 {len(hp)}명 예측 → Evaluate/08_exp3_holdout_predictions.csv (평가는 11에서)")
        return 0
    L, F, per, ymap = run_cv(d, specs, "full")
    # 앙상블: 같은 반복·폴드 확률 평균 (추가 학습 없음)
    for ens, mem in ENS.items():
        e = L[L.model.isin(mem)].groupby(["pid", "repeat", "fold"], as_index=False)["prob"].mean(); e["model"] = ens
        L = pd.concat([L, e[L.columns]], ignore_index=True)
        for (r_, f_), g in e.groupby(["repeat", "fold"]):
            F = pd.concat([F, pd.DataFrame([{"repeat": r_, "fold": f_, "model": ens, "n_train": len(d) - len(g),
                                              "n_test": len(g), "auc": roc_auc_score(g.pid.map(ymap), g.prob)}])])
        per[ens] = e.groupby("repeat").apply(lambda g: roc_auc_score(g.pid.map(ymap), g.prob), include_groups=False)
        a = per[ens].to_numpy(); print(f"  {ens:24s} AUC {a.mean():.3f} [{np.percentile(a,2.5):.3f}–{np.percentile(a,97.5):.3f}]")
        specs[ens] = ([], [])
    L.to_csv(cfg.EVAL_DIR / "08_exp3_oof_full.csv", index=False)
    wide = F.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    print()
    dl = []
    for a, b, desc in [("a_clinical", "b_clin_preimg", "임상 → +치료 전 영상"),
                       ("a_clinical", "c_clin_ftvchange", "임상 → +FTV 변화만"),
                       ("c_clin_ftvchange", "d_clin_t1full", "FTV 변화 → +T1 곡선·habitat"),
                       ("b_clin_preimg", "e_clin_preimg_t1full", "치료 전 영상 → +T1 전체"),
                       ("a_clinical", "e_clin_preimg_t1full", "임상 → +치료 전+T1 전체"),
                       ("d_clin_t1full", "ens_t1", "L1 T1 단일 → T1 관점 앙상블"),
                       ("d_clin_t1full", "ens_t1_rad", "L1+L3 → T1 앙상블(+T1 질감)"),
                       ("a_clinical", "ens_t1_rad", "임상 → T1 앙상블 최종")]:
        if a not in per or b not in per:
            continue
        dr = (per[b] - per[a]).to_numpy()
        pv = ev.corrected_ttest((wide[b] - wide[a]).to_numpy(), F.n_train.mean(), F.n_test.mean())
        dl.append({"desc": desc, "delta_mean": dr.mean(), "frac_pos": (dr > 0).mean(), "p": pv})
        print(f"  {desc:28s} ΔAUC {dr.mean():+.3f} · 양수 반복 {(dr > 0).mean():.0%} · p={pv:.4f} {ev._star(pv)}")
    pd.DataFrame(dl).to_csv(cfg.EVAL_DIR / "08_exp3_delta_full.csv", index=False)
    sub = dict(zip(d[cfg.COL_PID], d["subtype"])); L["st"] = L.pid.map(sub); L["y"] = L.pid.map(ymap)
    print("\n[아형별 AUC (반복 평균)]")
    for m in specs:
        g = L[L.model == m]
        txt = " · ".join(f"{cfg.SUBTYPE_LABEL.get(s, s)} "
                         f"{np.mean([roc_auc_score(x.y, x.prob) for _, x in gs.groupby('repeat') if x.y.nunique()==2]):.3f}"
                         f"(n={gs.pid.nunique()})" for s, gs in g.groupby("st"))
        print(f"  {m:24s} {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
