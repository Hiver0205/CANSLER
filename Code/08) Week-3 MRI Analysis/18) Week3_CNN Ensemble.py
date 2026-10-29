"""
08b_exp3_cnn_ensemble.py ─ 실험 3 L2: T1 CNN 점수를 T1 관점 앙상블에 사후 결합
==========================================================================
08(--source full)과 04c(--root T1)는 같은 코호트·같은 순서·같은 random_state → 반복 0..R-1의 폴드가 동일.
겹치는 반복에서 앙상블 확률 평균에 CNN 확률(sigmoid)을 추가해 비교한다 (추가 학습 없음 → 누수 없음).

입력:  Evaluate/08_exp3_oof_full.csv, Evaluate/04c_cnn_oof_long_t6_T1.csv
실행:  python 08b_exp3_cnn_ensemble.py
"""
from __future__ import annotations
import sys
from importlib import import_module
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

cfg = import_module("00_config")


def main() -> int:
    L = pd.read_csv(cfg.EVAL_DIR / "08_exp3_oof_full.csv", dtype={"pid": str})
    C = pd.read_csv(cfg.EVAL_DIR / "04c_cnn_oof_long_t6_T1.csv", dtype={"pid": str})
    R = sorted(set(C.repeat) & set(L.repeat)); L = L[L.repeat.isin(R)]
    chk = L[L.model == "a_clinical"][["pid", "repeat", "fold"]].merge(
        C[C.model == "clinical"][["pid", "repeat", "fold"]], on=["pid", "repeat"])
    print(f"[i] 반복 {R} · 폴드 일치율 {(chk.fold_x == chk.fold_y).mean():.3f} (1.000이어야 함)")
    ymap = (pd.read_csv(cfg.DATA_FEATURES / "analysis_table.csv", dtype={cfg.COL_PID: str})
            .set_index(cfg.COL_PID)["pCR"])
    cnn = C[C.model == "cnn_only"].copy(); cnn["prob"] = 1 / (1 + np.exp(-cnn.prob)); cnn["model"] = "cnn_t1"
    A = pd.concat([L, cnn[L.columns]])

    def auc(members, w_cnn=1):
        """구성 모델 확률의 (가중) 평균 → 반복별 AUC. CNN은 가중치 w_cnn."""
        sub = A[A.model.isin(members)].copy()
        sub["w"] = np.where(sub.model == "cnn_t1", float(w_cnn), 1.0)
        sub["wp"] = sub.w * sub.prob
        g = sub.groupby(["pid", "repeat"])[["wp", "w"]].sum()
        p = (g.wp / g.w).reset_index(name="prob")
        return np.array([roc_auc_score(ymap[x.pid], x.prob) for _, x in p.groupby("repeat")])

    ens = [m for m in L.model.unique() if m.startswith(("d_", "e_", "l1_", "l3_"))]
    rows = {"임상": auc(["a_clinical"]), "T1 단일 (d)": auc(["d_clin_t1full"]),
            "CNN(T1) 단독": auc(["cnn_t1"]), "T1 관점 앙상블": auc(ens),
            "T1 앙상블 + CNN": auc(ens + ["cnn_t1"]), "T1 앙상블 + CNN×2": auc(ens + ["cnn_t1"], 2)}
    print(f"\n=== 실험 3 최종 후보 (개발셋, 반복 {len(R)}회) ===")
    for k, v in rows.items():
        print(f"  {k:18s} AUC {v.mean():.3f}  (반복 범위 {v.min():.3f}–{v.max():.3f})")
    pd.DataFrame({k: v for k, v in rows.items()}).to_csv(cfg.EVAL_DIR / "08b_exp3_cnn_ensemble.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
