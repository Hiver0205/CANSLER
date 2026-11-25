"""
20_export_models.py ─ 웹 추론용 최종 모델 저장 (로지스틱 회귀: ens_main 14개 + ens_t3 11개)
==================================================================
11·08과 같은 방식(개발셋 전체 학습)으로 학습해 파일로 저장한다. CNN 가중치는 04c --final --save.
저장 후 다시 불러 홀드아웃을 예측하고, 기존 최종 예측값과 같은지 확인한다 (웹 예측 = 파이프라인 예측).

출력:  Models/ens_main_lr.joblib, Models/ens_t3_lr.joblib  {"members": {이름: {"est", "cols"}}, "created"}
확인:  Evaluate/11_holdout_predictions_main.csv, 08_exp3_holdout_predictions.csv 의 prob__<모델> 과 최대 차이
"""
from __future__ import annotations
import sys
import warnings
from datetime import datetime
from importlib import import_module
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
cfg = import_module("00_config")
tr6 = import_module("06_train_cv")
fh = import_module("11_final_holdout")
MODEL_DIR = cfg.PROJECT_ROOT / "Models"


def main() -> int:
    w = max(export_main(), export_t3())
    return 0 if w < 1e-6 else 1


def export_main():
    df = pd.read_csv(cfg.DATA_FEATURES / "analysis_table.csv"); df[cfg.COL_PID] = df[cfg.COL_PID].astype(str)
    dev = df[~df["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)
    hold = df[df["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)
    y = dev["pCR"].to_numpy().astype(int)
    members = {}
    for name in fh.MAIN_MEMBERS:   # 11의 fit_predict_main과 같은 학습 절차
        spec = tr6.MODELS[name]; opts = dict(spec[2]) if len(spec) > 2 else {}
        pca_pfx = opts.pop("pca", [])
        base, im = tr6.cols_of(dev, spec[0]), tr6.cols_of(dev, spec[1])
        if pca_pfx:
            opts["pca_cols"] = tr6.cols_of(dev, pca_pfx)
        cols = base + im + opts.get("pca_cols", []) + (["batch__id"] if opts.get("combat") else [])
        est = tr6.build(base, im, inner_seed=cfg.RANDOM_STATE, **opts).fit(dev[cols], y)
        members[name] = {"est": est, "cols": cols}
        print(f"  학습: {name} ({len(cols)}열)", flush=True)

    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / "ens_main_lr.joblib"
    joblib.dump({"members": members, "created": datetime.now().isoformat(timespec="seconds")}, out)
    print(f"[✓] {out}")

    # 다시 불러 홀드아웃 예측 → 11 결과와 비교
    bundle = joblib.load(out)
    ref = pd.read_csv(cfg.EVAL_DIR / "11_holdout_predictions_main.csv", dtype={cfg.COL_PID: str}).set_index(cfg.COL_PID)
    worst = 0.0
    for name, m in bundle["members"].items():
        p = m["est"].predict_proba(hold[m["cols"]])[:, 1]
        d = float(np.max(np.abs(p - ref.loc[hold[cfg.COL_PID], f"prob__{name}"].to_numpy())))
        worst = max(worst, d)
        print(f"  {name:22s} 최대 차이 {d:.2e}")
    print(f"[{'✓' if worst < 1e-6 else '!'}] 14개 모델 최대 차이 {worst:.2e}")
    return worst


T3_PREFIX = ("d_", "e_", "l1_", "l3_")   # 11이 ens_t3에 쓰는 구성 모델 (CNN 제외 11개)


def export_t3():
    """치료 3주 차 ens_t3의 로지스틱 회귀 11개 (08 --source full --final과 같은 학습)."""
    ex3 = import_module("08_exp3_early_response")
    d, hold = ex3.load_full()
    specs, _ = ex3.full_specs(d)
    y = d["pCR"].to_numpy().astype(int)
    members = {}
    for name, (base, im) in specs.items():
        if not name.startswith(T3_PREFIX):
            continue
        est = tr6.build(base, im, inner_seed=cfg.RANDOM_STATE).fit(d[base + im], y)
        members[name] = {"est": est, "cols": base + im}
        print(f"  학습: {name} ({len(base + im)}열)", flush=True)
    out = MODEL_DIR / "ens_t3_lr.joblib"
    joblib.dump({"members": members, "created": datetime.now().isoformat(timespec="seconds")}, out)
    print(f"[✓] {out} ({len(members)}개)")

    bundle = joblib.load(out)
    ref = pd.read_csv(cfg.EVAL_DIR / "08_exp3_holdout_predictions.csv", dtype={cfg.COL_PID: str}).set_index(cfg.COL_PID)
    worst = 0.0
    for name, m in bundle["members"].items():
        p = m["est"].predict_proba(hold[m["cols"]])[:, 1]
        d_ = float(np.max(np.abs(p - ref.loc[hold[cfg.COL_PID], f"prob__{name}"].to_numpy())))
        worst = max(worst, d_)
        print(f"  {name:22s} 최대 차이 {d_:.2e}")
    print(f"[{'✓' if worst < 1e-6 else '!'}] {len(members)}개 모델 최대 차이 {worst:.2e}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
