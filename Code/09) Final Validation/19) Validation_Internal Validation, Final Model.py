"""
11_final_holdout.py ─ 최종 홀드아웃 평가 (미리 고정한 모델만, 1회)
==================================================================
개발 과정에서 홀드아웃(split 1·2)은 모델 선택에 쓰지 않았다. 여기서 미리 고정한 구성만
개발셋 전체로 학습해 홀드아웃을 1회 예측·평가한다. 결과를 보고 모델을 바꾸지 않는다.

  주 연구 (치료 전, 홀드아웃 198명)
    · 기준: clinical
    · 비교: clinical_vol, combined_vol_kin, cv_tc_hab (단일, 해석용)
    · 주 모델: ens_main = 14개 관점 모델 평균 + CNN 2개(2.5D 6채널 TTA·3시드, 3D 6채널) 확률
  실험 1 (영상만): img_vol_tc_hab — 아형별 AUC
  실험 3 (치료 3주 차, 홀드아웃 중 T1 있는 197명)
    · 기준: a_clinical · 비교: c_clin_ftvchange, d_clin_t1full
    · 주 모델: ens_t3 = T1 관점 앙상블(+T1 질감) 구성 모델 평균 + T1 CNN 확률
지표: AUC(부트스트랩 95% CI), 임상 대비 ΔAUC(짝지은 부트스트랩 CI + DeLong), PR-AUC, Brier, 아형별 AUC(CI)

입력:  analysis_table.csv, Evaluate/04c_final_t6_tta_s3.csv, 04c_final_t6_3d.csv, 04c_final_t6_T1.csv,
       Evaluate/08_exp3_holdout_predictions.csv (08 --source full --final)
출력:  Evaluate/11_holdout_predictions_main.csv, 11_final_report.csv, 11_final_subtype.csv
"""
from __future__ import annotations
import sys
import warnings
from importlib import import_module
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")
cfg = import_module("00_config")
tr6 = import_module("06_train_cv")
ev = import_module("07_evaluate")

MAIN_MEMBERS = ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES", "cvk_tc",
                "cvk_tc_tex", "cv_tc", "cv_tc_peri", "cv_tc_bpe", "cv_tc_rad", "cv_tc_ES", "cv_tctex", "cv_tc_hab"]
MAIN_SINGLE = ["clinical", "clinical_vol", "combined_vol_kin", "img_vol_tc_hab"]
sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731


def fit_predict_main(df):
    dev = df[~df["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)
    hold = df[df["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)
    assert not set(dev[cfg.COL_PID]) & set(hold[cfg.COL_PID])
    y = dev["pCR"].to_numpy().astype(int)
    hp = hold[[cfg.COL_PID, "pCR", "subtype"]].copy()
    for name in dict.fromkeys(MAIN_SINGLE + MAIN_MEMBERS):
        spec = tr6.MODELS[name]; opts = dict(spec[2]) if len(spec) > 2 else {}
        pca_pfx = opts.pop("pca", [])
        base, im = tr6.cols_of(dev, spec[0]), tr6.cols_of(dev, spec[1])
        if pca_pfx:
            opts["pca_cols"] = tr6.cols_of(dev, pca_pfx)
        cols = base + im + opts.get("pca_cols", []) + (["batch__id"] if opts.get("combat") else [])
        est = tr6.build(base, im, inner_seed=cfg.RANDOM_STATE, **opts).fit(dev[cols], y)
        hp[f"prob__{name}"] = est.predict_proba(hold[cols])[:, 1]
        print(f"  학습·예측: {name}", flush=True)
    c2 = pd.read_csv(cfg.EVAL_DIR / "04c_final_t6_tta_s3.csv", dtype={cfg.COL_PID: str})
    c3 = pd.read_csv(cfg.EVAL_DIR / "04c_final_t6_3d.csv", dtype={cfg.COL_PID: str})
    hp["prob__cnn2d"] = sig(hp[cfg.COL_PID].map(c2.set_index(cfg.COL_PID).cnn_logit))
    hp["prob__cnn3d"] = sig(hp[cfg.COL_PID].map(c3.set_index(cfg.COL_PID).cnn_logit))
    hp["prob__ens_main"] = hp[[f"prob__{m}" for m in MAIN_MEMBERS] + ["prob__cnn2d", "prob__cnn3d"]].mean(axis=1)
    hp.to_csv(cfg.EVAL_DIR / "11_holdout_predictions_main.csv", index=False)
    return hp


def metrics(y, p):
    a, lo, hi = ev.auc_ci(y, p)
    return {"auc": a, "auc_lo": lo, "auc_hi": hi, "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p)}


def report(study, hp, ref, models, desc):
    y = hp["pCR"].to_numpy().astype(int); prev = y.mean(); rows = []
    print(f"\n=== {study} · 홀드아웃 n={len(y)} (pCR {int(y.sum())}명, {prev:.1%}) · PR-AUC 기준선 {prev:.3f}"
          f" · null Brier {prev*(1-prev):.3f} ===")
    for m in [ref] + models:
        p = hp[f"prob__{m}"].to_numpy(); r = {"study": study, "model": m, "desc": desc.get(m, m), **metrics(y, p)}
        if m != ref:
            p0 = hp[f"prob__{ref}"].to_numpy()
            _, _, pv = ev.delong(y, p0, p); lo, hi = ev.delta_auc_ci(y, p0, p)
            r.update(delta=r["auc"] - roc_auc_score(y, p0), delta_lo=lo, delta_hi=hi, p_delong=pv)
        rows.append(r)
        extra = (f" · ΔAUC {r['delta']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}] p={r['p_delong']:.4f} "
                 f"{ev._star(r['p_delong'])}") if m != ref else ""
        print(f"  {r['desc']:34s} AUC {r['auc']:.3f} [{r['auc_lo']:.3f}–{r['auc_hi']:.3f}] · PR-AUC {r['pr_auc']:.3f}"
              f" · Brier {r['brier']:.3f}{extra}")
    return rows


def subtype_rows(study, hp, models, desc):
    out = []
    for m in models:
        for st, g in hp.groupby("subtype"):
            yy = g["pCR"].to_numpy().astype(int)
            if yy.sum() < 3 or (1 - yy).sum() < 3:
                continue
            a, lo, hi = ev.auc_ci(yy, g[f"prob__{m}"].to_numpy())
            out.append({"study": study, "model": m, "desc": desc.get(m, m), "subtype": cfg.SUBTYPE_LABEL.get(st, st),
                        "n": len(yy), "n_pcr": int(yy.sum()), "auc": a, "auc_lo": lo, "auc_hi": hi})
    return out


def main() -> int:
    df = pd.read_csv(cfg.DATA_FEATURES / "analysis_table.csv"); df[cfg.COL_PID] = df[cfg.COL_PID].astype(str)
    print("[i] 주 연구 모델: 개발셋 전체 학습 → 홀드아웃 예측", flush=True)
    hp = fit_predict_main(df)
    D = {"clinical": "임상", "clinical_vol": "임상+부피", "combined_vol_kin": "임상+부피+동역학",
         "cv_tc_hab": "임상+부피+곡선+habitat (단일)", "ens_main": "★ 임상+영상 (14관점+CNN 앙상블)",
         "img_vol_tc_hab": "영상만 (부피+곡선+habitat)",
         "a_clinical": "임상", "c_clin_ftvchange": "임상+부피+FTV 변화", "d_clin_t1full": "임상+T1 (단일)",
         "ens_t3": "★ 임상+치료전·T1 영상 (앙상블+CNN)"}
    rows = report("주 연구 (치료 전)", hp, "clinical", ["clinical_vol", "combined_vol_kin", "cv_tc_hab", "ens_main"], D)
    rows += report("실험 1 (영상만)", hp, "clinical", ["img_vol_tc_hab"], D)
    sub = subtype_rows("주 연구", hp, ["clinical", "ens_main", "img_vol_tc_hab"], D)

    t3p = cfg.EVAL_DIR / "08_exp3_holdout_predictions.csv"
    if t3p.exists():
        h3 = pd.read_csv(t3p, dtype={cfg.COL_PID: str})
        c1 = pd.read_csv(cfg.EVAL_DIR / "04c_final_t6_T1.csv", dtype={cfg.COL_PID: str})
        h3["prob__cnn_t1"] = sig(h3[cfg.COL_PID].map(c1.set_index(cfg.COL_PID).cnn_logit))
        mem = [c for c in h3 if c.startswith("prob__") and c[6:].startswith(("d_", "e_", "l1_", "l3_"))]
        h3["prob__ens_t3"] = h3[mem + ["prob__cnn_t1"]].mean(axis=1)
        h3.to_csv(t3p, index=False)
        rows += report("실험 3 (치료 3주 차)", h3, "a_clinical", ["c_clin_ftvchange", "d_clin_t1full", "ens_t3"], D)
        sub += subtype_rows("실험 3", h3, ["a_clinical", "ens_t3"], D)
    pd.DataFrame(rows).to_csv(cfg.EVAL_DIR / "11_final_report.csv", index=False)
    sdf = pd.DataFrame(sub); sdf.to_csv(cfg.EVAL_DIR / "11_final_subtype.csv", index=False)
    print("\n[아형별 AUC (홀드아웃, 95% CI — 표본이 작아 구간이 넓음)]")
    for r in sdf.itertuples():
        print(f"  {r.study:6s} {r.desc:34s} {r.subtype:9s} n={r.n:3d}(pCR {r.n_pcr:2d})  "
              f"AUC {r.auc:.3f} [{r.auc_lo:.3f}–{r.auc_hi:.3f}]")
    print(f"\n[✓] Evaluate/11_final_report.csv, 11_final_subtype.csv, 11_holdout_predictions_main.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
