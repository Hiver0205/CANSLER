"""
07_evaluate.py ─ 2단계 평가 (영상의 증분 가치)
=============================================
헤드라인: **임상+부피 vs 임상+부피+영상** — 종양 크기를 통제한 뒤에도 영상이 더하는가.

A. 개발셋 반복 CV (06_oof_long / 06_fold_auc)
   · 모델별 반복별 pooled AUC·PR-AUC·Brier → 평균과 2.5~97.5% 범위 (학습 변동 포함)
   · 비교쌍별 ΔAUC 반복 분포 + corrected resampled t-test (Nadeau–Bengio / Bouckaert–Frank)
     ※ 반복 평균 OOF에 DeLong을 쓰면 폴드 간 상관·앙상블 효과로 과신 → 쓰지 않음
B. 홀드아웃 (06_holdout_predictions) — 개발에 한 번도 안 쓴 환자, 1회 평가
   · AUC + 부트스트랩 95%CI, PR-AUC(유병률 기준선), Brier(null Brier 기준선)
   · 비교쌍별 DeLong p + 짝지은 부트스트랩 ΔAUC 95%CI
C. 보조: 아형별 AUC (개발셋 반복별 → 평균과 범위)

입력:  Evaluate/06_oof_long.csv, 06_fold_auc.csv, 06_oof_predictions.csv, [06_holdout_predictions.csv]
출력:  Evaluate/07_cv_metrics.csv, 07_cv_delta.csv, 07_holdout_metrics.csv, 07_holdout_delta.csv,
       07_subtype_auc.csv, 07_roc.png, 07_calibration.png

실행:  python 07_evaluate.py
"""
from __future__ import annotations
import sys
import warnings
from importlib import import_module
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score, roc_curve)

warnings.filterwarnings("ignore", category=FutureWarning)
cfg = import_module("00_config")

# (기준 모델, 비교 모델, 설명) — 첫 줄이 헤드라인
COMPARISONS = [
    ("clinical_vol", "combined_vol",     "헤드라인: 임상+부피 → +radiomics"),
    ("clinical_vol", "combined_vol_kin", "임상+부피 → +동역학"),
    ("clinical_vol", "combined_vol_all", "임상+부피 → +radiomics+동역학"),
    ("clinical_vol", "combined_vol_kinx", "임상+부피 → +아형별 동역학"),
    ("clinical_vol", "combined_vol_kin_kinx", "임상+부피 → +동역학(공통+아형별)"),
    ("combined_vol_kin", "combined_vol_kin_kinx", "동역학 → +아형별 상호작용"),
    # 개선 라운드 2: 현재 최고(combined_vol_kin) 대비
    ("clinical_vol",     "clinical_vol_fe",      "A 임상+부피 → 임상 다듬기"),
    ("combined_vol_kin", "combined_vol_kin_fe",  "A 동역학모델 → 임상 다듬기"),
    ("combined_vol_kin", "combined_vol_kin_gbm", "B 로지스틱 → 얕은 GBM"),
    ("combined_vol_kin", "combined_vol_kin_rf",  "B 로지스틱 → 랜덤포레스트"),
    ("combined_vol_kin", "combined_vol_kin_ens", "B 로지스틱 → 로지스틱+GBM 앙상블"),
    ("combined_vol_kin", "combined_vol_kin_E",   "C 동역학 → +초기PE 질감"),
    ("combined_vol_kin", "combined_vol_kin_S",   "C 동역학 → +SER 질감"),
    ("combined_vol_kin", "combined_vol_kin_ES",  "C 동역학 → +초기PE·SER 질감"),
    # 개선 라운드 3: 사전학습 CNN 특징
    ("clinical_vol",     "combined_vol_deepR",     "D 임상+부피 → +CNN(ResNet50)"),
    ("combined_vol_kin", "combined_vol_kin_deepR", "D 동역학 → +CNN(ResNet50)"),
    ("clinical_vol",     "combined_vol_deepD",     "D 임상+부피 → +CNN(DINOv2)"),
    ("combined_vol_kin", "combined_vol_kin_deepD", "D 동역학 → +CNN(DINOv2)"),
    # 개선 라운드 4: TCIA 임상
    ("clinical_vol",       "clinical_vol_mp",       "E 임상+부피 → +MammaPrint"),
    ("clinical_vol",       "clinical_vol_arm",      "E 임상+부피 → +배정약제"),
    ("clinical_vol",       "clinical_vol_tcia",     "E 임상+부피 → +MP+약제"),
    ("combined_vol_kin",   "combined_vol_kin_tcia", "E 최고모델 → +MP+약제"),
    ("clinical_vol_tcia",  "combined_vol_kin_tcia", "E 임상+부피+MP+약제 → +동역학"),
    # 영상 라운드 (초기 계획): 실험 2 = 임상 → 임상+영상
    ("clinical",         "combined_vol_kin", "실험2 임상 → +영상(부피+동역학)"),
    ("clinical",         "cvk_peri",         "실험2 임상 → +영상(부피+동역학+주변)"),
    ("combined_vol_kin", "cvk_peri",         "G1 +종양주변 통계"),
    ("combined_vol_kin", "cvk_peri_tex",     "G1 +종양주변 통계+질감"),
    ("combined_vol_kin", "cvk_bpe",          "G2 +배경실질(BPE)"),
    ("combined_vol_kin", "cvk_cb",           "G3 동역학 보정"),
    ("cvk_peri",         "cvk_peri_cb",      "G3 동역학+주변 보정"),
    ("combined_vol_kin", "cvk_rad_cb",       "G3 +radiomics(보정)"),
    ("img_vol_kin",      "img_vol_kin_peri", "실험1 영상만: +주변"),
    # 라운드 H (기준 = 임상+부피+동역학)
    ("combined_vol_kin", "cvk_morph",        "H1 +형태·다발성"),
    ("combined_vol_kin", "cvk_l2",           "H2 L1선택 → L2축소"),
    ("combined_vol_kin", "cvk_morph_l2",     "H1+H2"),
    ("combined_vol_kin", "ens_views",        "H3 관점 앙상블(5)"),
    ("combined_vol_kin", "ens_views_m",      "H3 관점 앙상블(7)"),
    ("combined_vol_kin", "ens_kin_l1l2",     "H3 동역학 앙상블(3)"),
    ("clinical",         "ens_views_m",      "실험2 임상 → 앙상블"),
    ("img_vol_kin",      "img_vol_kin_morph", "실험1 영상만: +형태"),
    # 라운드 J1: 전체 시간-증강 곡선
    ("combined_vol_kin", "cv_tc",            "J1 3시점 동역학 → 전체 곡선"),
    ("combined_vol_kin", "cvk_tc",           "J1 +전체 곡선 통계"),
    ("combined_vol_kin", "cvk_tc_tex",       "J1 +곡선 통계+지도 질감"),
    ("combined_vol_kin", "cv_tctex",         "J1 동역학 → 곡선 지도 질감"),
    ("ens_views",        "ens_views_tc",     "J1 앙상블 +곡선 모델"),
    ("clinical",         "ens_views_tc",     "실험2 임상 → 임상+영상(곡선 포함 앙상블)"),
    ("img_vol_kin",      "img_vol_kin_tc",   "실험1 영상만: +전체 곡선"),
    # 라운드 J4 (기준: 단일 = cv_tc 0.704, 앙상블 = ens_views_tc 0.711)
    ("cv_tc",            "cv_tc_hab",        "J4 전체곡선 → +habitat"),
    ("combined_vol_kin", "cv_hab",           "J4 동역학 → habitat"),
    ("ens_views_tc",     "ens_views_tc_hab", "J4 앙상블 +habitat 모델"),
    ("ens_views_tc",     "ens_tc",           "재구성 앙상블(곡선 관점 7)"),
    ("ens_views_tc",     "ens_tc_all",       "재구성 앙상블(전체 14)"),
    ("clinical",         "ens_tc_all",       "실험2 임상 → 임상+영상(전체 14 앙상블)"),
    ("img_vol_kin_tc",   "img_vol_tc_hab",   "실험1 영상만: 곡선+habitat"),
    # 라운드 J5 (기준: 단일 cv_tc_hab, 앙상블 ens_tc_all)
    ("cv_tc_hab",        "cv_tc_hab_x",      "J5 +K5·6·경계/섞임"),
    ("cv_tc_hab",        "cv_tc_hab_tex",    "J5 +유형별 질감"),
    ("cv_tc_hab",        "cv_tc_hab_all",    "J5 +전부"),
    ("ens_tc_all",       "ens_j5",           "J5 앙상블(17)"),
    ("clinical",         "ens_j5",           "실험2 임상 → 임상+영상(J5 앙상블)"),
    ("img_vol_tc_hab",   "img_vol_tc_hab_x", "실험1 영상만: +habitat 확장"),
    # 라운드 K (기준: 단일 cv_tc 0.704 / cv_tc_hab 0.707, 앙상블 ens_tc_all 0.715)
    ("cv_tc",            "cv_tc_bag16",      "K1 곡선 +단어주머니16"),
    ("cv_tc",            "cv_tc_bag32",      "K1 곡선 +단어주머니32"),
    ("cv_tc_hab",        "cv_tc_hab_bag",    "K1 곡선+habitat +단어주머니16"),
    ("cv_tc",            "cv_tc_vtc",        "K2 곡선 +부피×곡선"),
    ("cv_tc_hab",        "cv_tc_hab_vtc",    "K2 곡선+habitat +부피×곡선"),
    ("ens_tc_all",       "ens_k",            "K 앙상블(17)"),
    ("clinical",         "ens_k",            "실험2 임상 → 임상+영상(K 앙상블)"),
    ("img_vol_tc_hab",   "img_vol_tc_bag",   "실험1 영상만: habitat → 단어주머니"),
    # 라운드 I (기준 = 앙상블 ens_views 0.707)
    ("combined_vol_kin", "bag_cvk",          "I1 동역학모델 배깅"),
    ("ens_views",        "ens_bag",          "I1 앙상블 → 배깅 앙상블"),
    ("ens_views",        "ens_diverse",      "I2 앙상블 → 다양성 앙상블"),
    ("ens_views",        "ens_all",          "I1+I2 전체 앙상블"),
    ("clinical",         "ens_all",          "실험2 임상 → 전체 앙상블"),
    # 라운드 5: 기준 = 임상+부피+MP
    ("clinical_vol_mp",    "clin_vol_mp_x",            "F1 +MP×아형"),
    ("clinical_vol_mp",    "clin_vol_mp_drug",         "F2 +약제계열"),
    ("clinical_vol_mp",    "clin_vol_mp_x_drug",       "F1+F2"),
    ("clinical_vol_mp",    "clin_vol_mp_tuned",        "F3 규제강도 튜닝"),
    ("clinical_vol_mp",    "clin_vol_mp_x_drug_tuned", "F1+F2+F3"),
    ("clin_vol_mp_x_drug", "clin_vol_mp_x_drug_kin",   "F4 F1+F2 → +동역학(영상)"),
    ("clinical",     "combined",         "임상 → +radiomics (크기 미통제)"),
    ("clinical",     "clinical_vol",     "임상 → +부피 (크기 자체의 기여)"),
    ("imaging",      "combined",         "radiomics → +임상"),
]


# ── AUC 부트스트랩 CI ──────────────────────────────────────────────
def auc_ci(y, p, n=None, seed=0):
    n = n or cfg.N_BOOTSTRAP
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); st = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) == 2:
            st.append(roc_auc_score(y[s], p[s]))
    return roc_auc_score(y, p), np.percentile(st, 2.5), np.percentile(st, 97.5)


def delta_auc_ci(y, p1, p2, n=None, seed=0):
    """짝지은 부트스트랩: 같은 재표본에서 AUC(p2) − AUC(p1)."""
    n = n or cfg.N_BOOTSTRAP
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); st = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) == 2:
            st.append(roc_auc_score(y[s], p2[s]) - roc_auc_score(y[s], p1[s]))
    return np.percentile(st, 2.5), np.percentile(st, 97.5)


# ── DeLong (상관 ROC AUC 차이 검정) — 독립 홀드아웃에서만 사용 ─────
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def _fastdelong(preds, m):
    n = preds.shape[1]-m; k = preds.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m+n])
    for r in range(k):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    v01 = (tz[:, :m]-tx)/n; v10 = 1.0-(tz[:, m:]-ty)/m
    cov = np.cov(v01)/m + np.cov(v10)/n
    return aucs, cov


def delong(y, p1, p2):
    y = np.asarray(y); order = (-y).argsort(kind="mergesort")
    m = int(y[order].sum())
    preds = np.vstack((np.asarray(p1)[order], np.asarray(p2)[order]))
    aucs, cov = _fastdelong(preds, m)
    var = cov[0, 0]+cov[1, 1]-2*cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], float("nan")
    z = (aucs[0]-aucs[1])/np.sqrt(var)
    return aucs[0], aucs[1], 2*(1-stats.norm.cdf(abs(z)))


# ── 반복 CV용 corrected resampled t-test ───────────────────────────
def corrected_ttest(d, n_train, n_test):
    """폴드별 차이 d (k×r개). 폴드 간 학습셋 중첩을 보정한 분산 사용."""
    d = np.asarray(d, dtype=float); J = len(d); v = d.var(ddof=1)
    if J < 2 or v <= 0:
        return float("nan")
    t = d.mean() / np.sqrt((1.0/J + n_test/n_train) * v)
    return float(2 * stats.t.sf(abs(t), J - 1))


def _star(p):
    if not np.isfinite(p):
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _rng(a):
    return np.mean(a), np.percentile(a, 2.5), np.percentile(a, 97.5)


def main() -> int:
    cfg.ensure_dirs()
    E = cfg.EVAL_DIR
    if not (E / "06_oof_long.csv").exists():
        print(f"[!] {E / '06_oof_long.csv'} 없음. 먼저 06 실행."); return 1
    long = pd.read_csv(E / "06_oof_long.csv", dtype={"pid": str})
    folds = pd.read_csv(E / "06_fold_auc.csv")
    oof = pd.read_csv(E / "06_oof_predictions.csv", dtype={cfg.COL_PID: str})
    lab = oof.set_index(cfg.COL_PID)[["pCR", "subtype"]]
    long = long.join(lab, on="pid")
    models = list(dict.fromkeys(long["model"]))
    comps = [c for c in COMPARISONS if c[0] in models and c[1] in models]
    prev = oof["pCR"].mean()
    print(f"[i] 모델 {models} · 개발 n={len(oof)} · pCR {prev:.1%} · 반복 {long['repeat'].nunique()}")

    # ── A. 개발셋 반복 CV ──────────────────────────────────────────
    per_rep = []
    for (m, r), d in long.groupby(["model", "repeat"]):
        per_rep.append({"model": m, "repeat": r, "auc": roc_auc_score(d.pCR, d.prob),
                        "pr_auc": average_precision_score(d.pCR, d.prob),
                        "brier": brier_score_loss(d.pCR, d.prob)})
    per_rep = pd.DataFrame(per_rep)

    print(f"\n=== A. 개발셋 반복 CV (평균 [2.5–97.5% 반복 범위]) · PR-AUC 기준선 {prev:.3f}"
          f" · null Brier {prev*(1-prev):.3f} ===")
    rows = []
    for m in models:
        g = per_rep[per_rep.model == m]
        a, alo, ahi = _rng(g.auc)
        rows.append({"model": m, "auc_mean": a, "auc_p2.5": alo, "auc_p97.5": ahi,
                     "pr_auc_mean": g.pr_auc.mean(), "brier_mean": g.brier.mean(),
                     "pr_auc_baseline": prev, "brier_null": prev*(1-prev)})
        print(f"  {m:22s} AUC {a:.3f} [{alo:.3f}–{ahi:.3f}] · PR-AUC {g.pr_auc.mean():.3f}"
              f" · Brier {g.brier.mean():.3f}")
    pd.DataFrame(rows).to_csv(E / "07_cv_metrics.csv", index=False)

    print("\n=== A. ΔAUC (반복별 분포 + corrected t-test, 폴드 단위) ===")
    drows = []
    wide_rep = per_rep.pivot(index="repeat", columns="model", values="auc")
    wide_fold = folds.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    n_tr, n_te = folds["n_train"].mean(), folds["n_test"].mean()
    for a, b, desc in comps:
        dr = (wide_rep[b] - wide_rep[a]).to_numpy()
        pv = corrected_ttest((wide_fold[b] - wide_fold[a]).to_numpy(), n_tr, n_te)
        mu, lo, hi = _rng(dr)
        drows.append({"model_a": a, "model_b": b, "desc": desc, "delta_mean": mu,
                      "delta_p2.5": lo, "delta_p97.5": hi, "frac_repeats_positive": (dr > 0).mean(),
                      "p_corrected_t": pv})
        print(f"  {desc:32s} ΔAUC {mu:+.3f} [{lo:+.3f}, {hi:+.3f}] · 양수 반복 {(dr > 0).mean():.0%}"
              f" · p={pv:.4f} {_star(pv)}")
    if drows:
        pd.DataFrame(drows).to_csv(E / "07_cv_delta.csv", index=False)

    # ── B. 홀드아웃 ───────────────────────────────────────────────
    hp_path = E / "06_holdout_predictions.csv"
    hold = pd.read_csv(hp_path, dtype={cfg.COL_PID: str}) if hp_path.exists() else None
    hmodels = [m for m in models if hold is not None and f"prob__{m}" in hold.columns]
    if hold is not None and hold["pCR"].nunique() == 2 and hmodels:
        yh = hold["pCR"].to_numpy().astype(int); hprev = yh.mean()
        print(f"\n=== B. 홀드아웃 (n={len(yh)}, pCR {int(yh.sum())}명 {hprev:.1%}) — 1회 평가 ===")
        hrows = []
        for m in hmodels:
            p = hold[f"prob__{m}"].to_numpy()
            a, lo, hi = auc_ci(yh, p)
            hrows.append({"model": m, "auc": a, "ci_lo": lo, "ci_hi": hi,
                          "pr_auc": average_precision_score(yh, p), "pr_auc_baseline": hprev,
                          "brier": brier_score_loss(yh, p), "brier_null": hprev*(1-hprev)})
            print(f"  {m:22s} AUC {a:.3f} [{lo:.3f}–{hi:.3f}] · PR-AUC {hrows[-1]['pr_auc']:.3f}"
                  f" · Brier {hrows[-1]['brier']:.3f}")
        pd.DataFrame(hrows).to_csv(E / "07_holdout_metrics.csv", index=False)

        hd = []
        for a, b, desc in [c for c in comps if c[0] in hmodels and c[1] in hmodels]:
            p1, p2 = hold[f"prob__{a}"].to_numpy(), hold[f"prob__{b}"].to_numpy()
            auc_a, auc_b, pv = delong(yh, p1, p2)
            lo, hi = delta_auc_ci(yh, p1, p2)
            hd.append({"model_a": a, "model_b": b, "desc": desc, "auc_a": auc_a, "auc_b": auc_b,
                       "delta": auc_b-auc_a, "delta_ci_lo": lo, "delta_ci_hi": hi, "p_delong": pv})
            print(f"  {desc:32s} ΔAUC {auc_b-auc_a:+.3f} [{lo:+.3f}, {hi:+.3f}] · DeLong p={pv:.4f} {_star(pv)}")
        if hd:
            pd.DataFrame(hd).to_csv(E / "07_holdout_delta.csv", index=False)
        print("  ※ 중첩 모델(기준 ⊂ 비교) DeLong은 보수적일 수 있음 → ΔAUC CI와 함께 해석")
    else:
        hold = None
        print("\n[i] 홀드아웃 예측 없음 → B 생략")

    # ── C. 보조: 아형별 AUC (반복별) ─────────────────────────────
    print("\n=== C. 아형별 AUC (개발셋 반복별 평균 [2.5–97.5%]) — 아형 내부 0.5가 '아형만' 기준선 ===")
    sub = []
    for (m, st), g in long.groupby(["model", "subtype"]):
        if g.pid.nunique() < 15 or g.pCR.nunique() < 2:
            continue
        aucs = [roc_auc_score(d.pCR, d.prob) for _, d in g.groupby("repeat") if d.pCR.nunique() == 2]
        mu, lo, hi = _rng(aucs)
        sub.append({"model": m, "subtype": cfg.SUBTYPE_LABEL.get(st, st), "n": g.pid.nunique(),
                    "n_pcr": int(g.drop_duplicates("pid").pCR.sum()),
                    "auc_mean": mu, "auc_p2.5": lo, "auc_p97.5": hi})
    if sub:
        sdf = pd.DataFrame(sub)
        for m in models:
            txt = " · ".join(f"{r.subtype} {r.auc_mean:.3f}[{r.auc_p2_5:.2f}–{r.auc_p97_5:.2f}](n={r.n})"
                             for r in sdf[sdf.model == m].rename(
                                 columns={"auc_p2.5": "auc_p2_5", "auc_p97.5": "auc_p97_5"}).itertuples())
            print(f"  {m:22s} {txt}")
        sdf.to_csv(E / "07_subtype_auc.csv", index=False)

    _plots(oof, hold, models)
    print(f"\n[✓] 모든 지표 → {E}")
    return 0


def _plots(oof, hold, models):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            pass
        from sklearn.calibration import calibration_curve
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:  # noqa: BLE001
        print(f"[i] 그림 생략: {e}"); return

    # ROC: 홀드아웃이 있으면 홀드아웃, 없으면 개발셋 반복평균 OOF
    src, tag = (hold, "홀드아웃") if hold is not None else (oof, "개발 CV 평균 OOF")
    y = src["pCR"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in [m for m in models if f"prob__{m}" in src.columns]:
        p = src[f"prob__{m}"].to_numpy(); fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{m} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("1 - 특이도"); ax.set_ylabel("민감도"); ax.set_title(f"ROC ({tag}, n={len(y)})")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
    fig.savefig(cfg.EVAL_DIR / "07_roc.png", dpi=150)

    # Calibration: 표본이 큰 개발셋 OOF 사용
    y = oof["pCR"].to_numpy().astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in models:
        p = oof[f"prob__{m}"].to_numpy()
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=m, ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_xlabel("예측 확률"); ax.set_ylabel("실제 비율"); ax.set_title("Calibration (개발 CV 평균 OOF)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(cfg.EVAL_DIR / "07_calibration.png", dpi=150)
    print("[✓] 07_roc.png, 07_calibration.png")


if __name__ == "__main__":
    sys.exit(main())
