"""
06_train_cv.py ─ 누수 없는 nested 교차검증 + 홀드아웃 (임상 / 크기 / 영상 / 결합)
==========================================================================
2단계 설계의 핵심 학습 단계. 모든 모델을 **동일한 CV 분할**에서 비교한다.
  · clinical          : 임상 baseline (HR·HER2·나이·인종·폐경)
  · clinical_vol      : 임상 + 종양 부피(log)        ← 크기 효과 통제 baseline
  · imaging           : radiomics 만
  · combined          : 임상 + radiomics
  · combined_vol      : 임상 + 부피 + radiomics      ← 헤드라인(clinical_vol 대비 증분)
  · kinetic           : 동역학(03b) 만
  · combined_vol_kin  : 임상 + 부피 + 동역학
  · combined_vol_all  : 임상 + 부피 + radiomics + 동역학

누수 방지(문서 §8):
  · 홀드아웃(cfg.HOLDOUT_SPLITS)은 개발 CV·튜닝에서 완전히 제외, 마지막에 1회만 예측.
  · 결측대치·스케일·특징선택(L1)·분류기를 Pipeline으로 묶어 **폴드 안에서만** fit.
  · 분류기는 모든 모델 공통(cfg.CLASSIFIER, 기본 L2 로지스틱) — ΔAUC가 특징 차이만 반영.
  · L1 강도 C는 바깥 폴드의 학습분 안에서 안쪽 CV로 선택 (nested CV).
  · 임상·부피 변수는 L1 선택 대상에서 제외(항상 유지) — 영상 특징만 선택.
  · 환자(pid) 단위 그룹 분할 + 개발/홀드아웃 pid 교집합 검사.

입력:  Data/Features/analysis_table.csv (05)
출력:  Evaluate/06_oof_long.csv          (반복·폴드·모델별 OOF 확률 — 07의 반복별 ΔAUC용)
       Evaluate/06_fold_auc.csv          (폴드별 AUC — 07의 corrected t-test용)
       Evaluate/06_oof_predictions.csv   (반복 평균 OOF 확률, wide — 참고·보정곡선·아형 분해)
       Evaluate/06_cv_auc.csv            (모델별 반복 CV AUC 요약)
       Evaluate/06_holdout_predictions.csv, 06_holdout_params.csv

실행:  python 06_train_cv.py                 # 최종: 개발 CV + 홀드아웃 1회
       python 06_train_cv.py --no-holdout    # 개선안 탐색: 개발 CV만 (홀드아웃 안 봄)
"""
from __future__ import annotations
import argparse
import sys
import warnings
from importlib import import_module
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectorMixin, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
# sklearn≥1.8: penalty 인자 폐지 예고 경고. penalty="l1"은 여전히 적용됨(비영 계수로 확인) → 소음만 제거
warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1")
cfg = import_module("00_config")

RAD, KIN, KINX = f"{cfg.IMG_PREFIX}_", "kin_", "kinx_"
RAD_E, RAD_S, CLINX = "dceE_", "dceS_", "clinx__"   # 초기 PE 지도 질감, SER 지도 질감, 임상 파생
SPLINE_COLS = ["clin__age", "vol__log_tum_vol"]      # spline 옵션 시 비선형(3차 스플라인)으로
# 모델명 → (항상 유지할 변수 접두어들, L1 선택 대상 영상 접두어들[, 옵션])
#   옵션: {"clf": 분류기 이름(기본 cfg.CLASSIFIER), "spline": 나이·부피 스플라인}
MODELS = {
    "clinical":         (["clin__"], []),
    "clinical_vol":     (["clin__", "vol__"], []),
    "imaging":          ([], [RAD]),
    "combined":         (["clin__"], [RAD]),
    "combined_vol":     (["clin__", "vol__"], [RAD]),            # 헤드라인 (radiomics)
    "kinetic":          ([], [KIN]),
    "combined_vol_kin": (["clin__", "vol__"], [KIN]),            # 임상+부피+동역학
    "combined_vol_all": (["clin__", "vol__"], [RAD, KIN]),       # 임상+부피+radiomics+동역학
    "combined_vol_kinx":     (["clin__", "vol__"], [KINX]),      # 임상+부피+아형별 동역학
    "combined_vol_kin_kinx": (["clin__", "vol__"], [KIN, KINX]), # 임상+부피+동역학(공통+아형별)
    # ── 개선 라운드 2 (개발 CV로만 비교) ──
    # A. 임상 다듬기: HR×HER2 + 나이·부피 스플라인
    "clinical_vol_fe":      (["clin__", CLINX, "vol__"], [], {"spline": True}),
    "combined_vol_kin_fe":  (["clin__", CLINX, "vol__"], [KIN], {"spline": True}),
    # B. 분류기 (특징은 combined_vol_kin과 동일)
    "combined_vol_kin_gbm": (["clin__", "vol__"], [KIN], {"clf": "gbm_shallow"}),
    "combined_vol_kin_rf":  (["clin__", "vol__"], [KIN], {"clf": "rf"}),
    "combined_vol_kin_ens": (["clin__", "vol__"], [KIN], {"clf": "ens"}),
    # C. 동역학 지도 질감 (초기 PE 지도 / SER 지도)
    "combined_vol_kin_E":   (["clin__", "vol__"], [KIN, RAD_E]),
    "combined_vol_kin_S":   (["clin__", "vol__"], [KIN, RAD_S]),
    "combined_vol_kin_ES":  (["clin__", "vol__"], [KIN, RAD_E, RAD_S]),
    # ── 개선 라운드 3: 사전학습 CNN 특징 (04b) — 수천 차원이라 L1 대신 폴드 내 PCA ──
    "deepR":                ([], [], {"pca": ["deepR_"]}),
    "combined_vol_deepR":   (["clin__", "vol__"], [], {"pca": ["deepR_"]}),
    "combined_vol_kin_deepR": (["clin__", "vol__"], [KIN], {"pca": ["deepR_"]}),
    "deepD":                ([], [], {"pca": ["deepD_"]}),
    "combined_vol_deepD":   (["clin__", "vol__"], [], {"pca": ["deepD_"]}),
    "combined_vol_kin_deepD": (["clin__", "vol__"], [KIN], {"pca": ["deepD_"]}),
}
PCA_GRID = [5, 10, 20, 40]

# ── 개선 라운드 4: TCIA 임상(치료 전) — MammaPrint, 배정 약제 ──
MODELS.update({
    "clinical_vol_mp":      (["clin__", "vol__", "mp__"], []),
    "clinical_vol_arm":     (["clin__", "vol__", "arm__"], []),
    "clinical_vol_tcia":    (["clin__", "vol__", "mp__", "arm__"], []),
    "combined_vol_kin_tcia": (["clin__", "vol__", "mp__", "arm__"], [KIN]),
    # ── 라운드 5: MP·약제 활용 다듬기 (기준 = clinical_vol_mp) ──
    "clin_vol_mp_x":        (["clin__", "vol__", "mp__", "mpx__"], []),
    "clin_vol_mp_drug":     (["clin__", "vol__", "mp__", "drug__"], []),
    "clin_vol_mp_x_drug":   (["clin__", "vol__", "mp__", "mpx__", "drug__"], []),
    "clin_vol_mp_tuned":    (["clin__", "vol__", "mp__"], [], {"tune_C": True}),
    "clin_vol_mp_x_drug_tuned": (["clin__", "vol__", "mp__", "mpx__", "drug__"], [], {"tune_C": True}),
    "clin_vol_mp_x_drug_kin":   (["clin__", "vol__", "mp__", "mpx__", "drug__"], [KIN]),
})
LOGREG_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# ── 영상 라운드 (초기 계획 복귀): 임상 = HR·HER2·나이·인종·폐경, 영상만으로 끌어올리기 ──
PERI, BPE = "peri_", "peri_r20_40"
SHT = ["pT1_", "pT2_", "pT3_"]          # 고리 0–5 / 5–10 / 10–20mm 질감
MODELS.update({
    # G1 종양 주변 (1차 통계 → +질감)
    "cvk_peri":       (["clin__", "vol__"], [KIN, PERI]),
    "cvk_peri_tex":   (["clin__", "vol__"], [KIN, PERI] + SHT),
    # G2 배경 실질 증강(BPE 근사, 20–40mm 고리)만 추가
    "cvk_bpe":        (["clin__", "vol__"], [KIN, BPE]),
    # G3 하모나이제이션 (같은 특징, 촬영 조건 보정)
    "cvk_cb":         (["clin__", "vol__"], [KIN], {"combat": True}),
    "cvk_peri_cb":    (["clin__", "vol__"], [KIN, PERI], {"combat": True}),
    "cvk_rad_cb":     (["clin__", "vol__"], [KIN, RAD], {"combat": True}),
    # 실험 1용: 영상만 (부피 + 동역학 + 주변)
    "img_vol_kin":      (["vol__"], [KIN]),
    "img_vol_kin_peri": (["vol__"], [KIN, PERI]),
})

# ── 라운드 H: 임상+영상 끌어올리기 ──
MORPH, SHAPE = "morph_", f"{RAD}original_shape_"
MODELS.update({
    # H1 형태·다발성 (L1 대상)
    "cvk_morph":      (["clin__", "vol__"], [KIN, MORPH, SHAPE]),
    # H2 선택 없이 L2 축소 (동역학 전부 base로, 규제 강도 안쪽 CV)
    "cvk_l2":         (["clin__", "vol__", KIN], [], {"tune_C": True}),
    "cvk_morph_l2":   (["clin__", "vol__", KIN, MORPH, SHAPE], [], {"tune_C": True}),
    "img_vol_kin_morph": (["vol__"], [KIN, MORPH, SHAPE]),
})
# H3 관점 앙상블: 같은 반복·폴드에서 구성 모델 예측 확률의 단순 평균 (추가 학습 없음 → 누수 없음)
ENSEMBLES = {
    "ens_views":   ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES"],
    "ens_views_m": ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
                    "cvk_morph", "cvk_l2"],
    "ens_kin_l1l2": ["combined_vol_kin", "cvk_l2", "cvk_morph"],
}

# ── 라운드 J1: 전체 시간-증강 곡선 (03e) ──
TC, TCX = "tc_", ["tcW_", "tcO_", "tcA_", "tcF_"]       # 곡선 통계 / 유입·유출·면적·초기비율 지도 질감
MODELS.update({
    "cv_tc":          (["clin__", "vol__"], [TC]),               # 3시점 동역학 대신 전체 곡선
    "cvk_tc":         (["clin__", "vol__"], [KIN, TC]),          # 3시점 + 전체 곡선
    "cvk_tc_tex":     (["clin__", "vol__"], [KIN, TC] + TCX),    # + 곡선 지도 질감
    "cv_tctex":       (["clin__", "vol__"], TCX),                # 곡선 지도 질감만
    "img_vol_kin_tc": (["vol__"], [KIN, TC]),                    # 실험 1: 영상만
})
ENSEMBLES.update({
    "ens_views_tc": ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
                     "cvk_tc", "cvk_tc_tex"],
})

# ── 라운드 J4: 곡선 기반 하위 영역(habitat, 03f) + 앙상블 재구성(3시점 동역학 → 전체 곡선) ──
HAB = "hab"
MODELS.update({
    "cv_hab":         (["clin__", "vol__"], [HAB]),
    "cv_tc_hab":      (["clin__", "vol__"], [TC, HAB]),
    "img_vol_tc_hab": (["vol__"], [TC, HAB]),
    # 기존 관점 모델의 동역학 부분을 전체 곡선(tc_)으로 바꾼 판
    "cv_tc_peri":     (["clin__", "vol__"], [TC, PERI]),
    "cv_tc_bpe":      (["clin__", "vol__"], [TC, BPE]),
    "cv_tc_rad":      (["clin__", "vol__"], [TC, RAD]),
    "cv_tc_ES":       (["clin__", "vol__"], [TC, RAD_E, RAD_S]),
})
ENSEMBLES.update({
    "ens_views_tc_hab": ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
                         "cvk_tc", "cvk_tc_tex", "cv_tc_hab"],
    "ens_tc":           ["cv_tc", "cv_tc_peri", "cv_tc_bpe", "cv_tc_rad", "cv_tc_ES", "cv_tctex", "cv_tc_hab"],
    "ens_tc_all":       ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
                         "cvk_tc", "cvk_tc_tex", "cv_tc", "cv_tc_peri", "cv_tc_bpe", "cv_tc_rad", "cv_tc_ES",
                         "cv_tctex", "cv_tc_hab"],
})

# ── 라운드 J5: habitat 확장 (K=5,6 · 유형 간 경계/섞임 = hbx, 유형별 질감 = hT1~3) ──
HBX, HTEX = "hbx", ["hT1_", "hT2_", "hT3_"]
MODELS.update({
    "cv_tc_hab_x":      (["clin__", "vol__"], [TC, HAB, HBX]),
    "cv_tc_hab_tex":    (["clin__", "vol__"], [TC, HAB] + HTEX),
    "cv_tc_hab_all":    (["clin__", "vol__"], [TC, HAB, HBX] + HTEX),
    "img_vol_tc_hab_x": (["vol__"], [TC, HAB, HBX]),
})
ENSEMBLES.update({
    "ens_j5": ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
               "cvk_tc", "cvk_tc_tex", "cv_tc", "cv_tc_peri", "cv_tc_bpe", "cv_tc_rad", "cv_tc_ES",
               "cv_tctex", "cv_tc_hab", "cv_tc_hab_x", "cv_tc_hab_tex", "cv_tc_hab_all"],
})

# ── 라운드 K: 곡선 단어주머니(K1) · 부피×곡선 상호작용(K2) ──
MODELS.update({
    "cv_tc_bag16":    (["clin__", "vol__"], [TC, "bag16_"]),
    "cv_tc_bag32":    (["clin__", "vol__"], [TC, "bag32_"]),
    "cv_tc_hab_bag":  (["clin__", "vol__"], [TC, HAB, "bag16_"]),
    "cv_tc_vtc":      (["clin__", "vol__"], [TC, "vtc_"]),
    "cv_tc_hab_vtc":  (["clin__", "vol__"], [TC, HAB, "vtc_"]),
    "img_vol_tc_bag": (["vol__"], [TC, "bag16_"]),
})
ENSEMBLES.update({
    "ens_k": ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
              "cvk_tc", "cvk_tc_tex", "cv_tc", "cv_tc_peri", "cv_tc_bpe", "cv_tc_rad", "cv_tc_ES",
              "cv_tctex", "cv_tc_hab", "cv_tc_bag16", "cv_tc_hab_bag", "cv_tc_hab_vtc"],
})

# ── 라운드 I: 앙상블 강화 ──
MODELS.update({
    # I1 배깅된 관점 모델 (부트스트랩 25)
    "bag_cvk":       (["clin__", "vol__"], [KIN], {"bag": 25}),
    "bag_cv_rad":    (["clin__", "vol__"], [RAD], {"bag": 25}),
    "bag_cvk_peri":  (["clin__", "vol__"], [KIN, PERI], {"bag": 25}),
    "bag_cvk_bpe":   (["clin__", "vol__"], [KIN, BPE], {"bag": 25}),
    "bag_cvk_ES":    (["clin__", "vol__"], [KIN, RAD_E, RAD_S], {"bag": 25}),
})
ENSEMBLES.update({
    "ens_bag":     ["bag_cvk", "bag_cv_rad", "bag_cvk_peri", "bag_cvk_bpe", "bag_cvk_ES"],
    # I2 다양성: 선형 5관점 + 비선형 + 딥러닝 + 보정 radiomics + 형태
    "ens_diverse": ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
                    "combined_vol_kin_gbm", "combined_vol_kin_rf", "combined_vol_kin_deepR",
                    "combined_vol_kin_deepD", "cvk_rad_cb", "cvk_morph"],
    "ens_all":     ["combined_vol_kin", "combined_vol", "cvk_peri", "cvk_bpe", "combined_vol_kin_ES",
                    "combined_vol_kin_gbm", "combined_vol_kin_rf", "combined_vol_kin_deepR",
                    "combined_vol_kin_deepD", "cvk_rad_cb", "cvk_morph",
                    "bag_cvk", "bag_cv_rad", "bag_cvk_peri", "bag_cvk_bpe", "bag_cvk_ES"],
})


class L1Select(SelectorMixin, BaseEstimator):
    """L1 로지스틱으로 영상 특징 선택. 0개 선택 시 전체 유지(학습 실패 방지) + 기록."""

    def __init__(self, C=0.1):
        self.C = C

    def fit(self, X, y):
        with warnings.catch_warnings():   # 병렬 작업자 프로세스에서도 penalty 폐지 경고 억제
            warnings.filterwarnings("ignore", message="Inconsistent values: penalty=l1")
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=self.C,
                                    class_weight="balanced", random_state=cfg.RANDOM_STATE).fit(X, y)
        sel = np.abs(lr.coef_.ravel()) > 1e-8
        self.n_features_in_ = X.shape[1]
        self.n_selected_ = int(sel.sum())
        self.support_ = sel if sel.any() else np.ones(X.shape[1], dtype=bool)
        return self

    def _get_support_mask(self):
        return self.support_


class BatchStandardize(BaseEstimator):
    """G3 하모나이제이션: 마지막 열 = 촬영 조건 그룹. 그룹별 평균·표준편차를 학습분에서만 추정해
    각 그룹을 전체(pooled) 평균·표준편차로 맞춘 뒤 그룹 열은 버린다. (ComBat의 경험적 베이즈 없는 판)"""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float); b, F = X[:, -1], X[:, :-1]
        self.mu_, self.sd_ = F.mean(0), F.std(0) + 1e-8
        self.stats_ = {}
        for g in np.unique(b):
            s = F[b == g]
            if len(s) >= 10:   # 너무 작은 그룹은 보정하지 않음
                self.stats_[g] = (s.mean(0), s.std(0) + 1e-8)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float); b, F = X[:, -1], X[:, :-1].copy()
        for g, (m, s) in self.stats_.items():
            k = b == g
            F[k] = (F[k] - m) / s * self.sd_ + self.mu_
        return F


class BagPipe(ClassifierMixin, BaseEstimator):
    """I1 배깅: 같은 파이프라인을 부트스트랩 표본 n_bags개로 학습 → 확률 평균.
    (sklearn BaggingClassifier는 DataFrame 열 이름을 잃을 수 있어 ColumnTransformer와 충돌 → 직접 구현)"""

    def __init__(self, estimator=None, n_bags=25, random_state=0):
        self.estimator, self.n_bags, self.random_state = estimator, n_bags, random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state); y = np.asarray(y)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        self.models_ = []
        for _ in range(self.n_bags):   # 층화 부트스트랩 (pCR 비율 유지)
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            self.models_.append(clone(self.estimator).fit(X.iloc[idx], y[idx]))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


def cols_of(df, prefixes):
    return [c for c in df.columns if any(c.startswith(p) for p in prefixes)]


def make_clf(name=None):
    name = name or cfg.CLASSIFIER
    rs = cfg.RANDOM_STATE
    if name == "gbm":
        return GradientBoostingClassifier(random_state=rs)
    if name == "gbm_shallow":   # 과적합 억제: 얕은 트리·작은 학습률·subsample·큰 잎
        return GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=2,
                                          subsample=0.8, min_samples_leaf=20, random_state=rs)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt",
                                      n_jobs=1, random_state=rs)
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=5000, random_state=rs)
    if name == "ens":           # 선형 + 비선형 soft voting
        return VotingClassifier([("lr", make_clf("logreg")), ("gbm", make_clf("gbm_shallow"))],
                                voting="soft")
    raise ValueError(f"알 수 없는 분류기: {name}")


def build(base_cols, img_cols, inner_seed, clf=None, spline=False, pca_cols=None, tune_C=False,
          combat=False, bag=0):
    if bag:   # 배깅: 각 가방 안에서는 튜닝 없이 고정 C (반복 CV에서 가장 자주 선택된 0.1)
        inner_pipe = build(base_cols, img_cols, inner_seed, clf=clf, spline=spline, pca_cols=pca_cols,
                           combat=combat)
        if isinstance(inner_pipe, GridSearchCV):
            inner_pipe = inner_pipe.estimator.set_params(
                **({"prep__img__select__C": 0.1} if img_cols else {}))
        return BagPipe(inner_pipe, n_bags=bag, random_state=inner_seed)
    trs = []
    pca_cols = pca_cols or []
    spl = [c for c in SPLINE_COLS if c in base_cols] if spline else []
    lin = [c for c in base_cols if c not in spl]
    if lin:
        trs.append(("base", Pipeline([("impute", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())]), lin))
    if spl:   # 매듭은 학습분 분위수로 정해짐 → Pipeline 안(폴드 내)에서만 fit
        trs.append(("spline", Pipeline([("impute", SimpleImputer(strategy="median")),
                                        ("spl", SplineTransformer(n_knots=4, degree=3, extrapolation="linear")),
                                        ("scale", StandardScaler())]), spl))
    if img_cols:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if combat:   # 그룹 열(batch__id)을 마지막에 붙여 넘기고, 보정 후 제거
            steps.append(("combat", BatchStandardize()))
        steps += [("var", VarianceThreshold()), ("scale", StandardScaler()), ("select", L1Select())]
        trs.append(("img", Pipeline(steps), img_cols + (["batch__id"] if combat else [])))
    if pca_cols:   # 고차원 CNN 특징: 폴드 내 표준화 → PCA(성분 수는 안쪽 CV로 선택) → 재표준화
        trs.append(("pca", Pipeline([("impute", SimpleImputer(strategy="median")),
                                     ("var", VarianceThreshold()),
                                     ("scale", StandardScaler()),
                                     ("pca", PCA(n_components=10, random_state=cfg.RANDOM_STATE)),
                                     ("scale2", StandardScaler())]), pca_cols))
    pipe = Pipeline([("prep", ColumnTransformer(trs)),
                     ("clf", make_clf(clf))])
    grid = {}
    if img_cols:
        grid["prep__img__select__C"] = cfg.LASSO_C_GRID
    if pca_cols:
        grid["prep__pca__pca__n_components"] = PCA_GRID
    if tune_C:   # 최종 로지스틱의 L2 규제 강도도 안쪽 CV로 선택
        grid["clf__C"] = LOGREG_C_GRID
    if not grid:
        return pipe
    inner = StratifiedKFold(cfg.INNER_FOLDS, shuffle=True, random_state=inner_seed)
    return GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc", n_jobs=cfg.N_JOBS, refit=True)


def chosen_params(est):
    out = {"C": np.nan, "n_selected": np.nan, "n_pca": np.nan}
    if isinstance(est, GridSearchCV):
        prep = est.best_estimator_.named_steps["prep"].named_transformers_
        if "img" in prep:
            out.update(C=est.best_params_["prep__img__select__C"],
                       n_selected=prep["img"].named_steps["select"].n_selected_)
        if "pca" in prep:
            out["n_pca"] = est.best_params_["prep__pca__pca__n_components"]
        if "clf__C" in est.best_params_:
            out["clf_C"] = est.best_params_["clf__C"]
    return out


def main() -> int:
    cfg.ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-holdout", action="store_true", help="개선안 탐색: 개발 CV만, 홀드아웃 예측 안 함")
    ap.add_argument("--models", nargs="*", help="실행할 모델 이름만 (기본: 전부)")
    args = ap.parse_args()
    tp = cfg.DATA_FEATURES / "analysis_table.csv"
    if not tp.exists():
        print(f"[!] 분석 테이블 없음: {tp}. 먼저 05 실행."); return 1
    df = pd.read_csv(tp); df[cfg.COL_PID] = df[cfg.COL_PID].astype(str)

    # ── 개발 / 홀드아웃 분리 ────────────────────────────────────────
    is_hold = df["split"].isin(cfg.HOLDOUT_SPLITS).to_numpy()
    dev, hold = df[~is_hold].reset_index(drop=True), df[is_hold].reset_index(drop=True)
    assert not set(dev[cfg.COL_PID]) & set(hold[cfg.COL_PID]), "개발/홀드아웃 pid 중복"
    y, groups = dev["pCR"].to_numpy().astype(int), dev[cfg.COL_PID].to_numpy()

    sets = {}
    for name, spec in MODELS.items():
        base_pfx, img_pfx = spec[0], spec[1]
        opts = dict(spec[2]) if len(spec) > 2 else {}
        pca_pfx = opts.pop("pca", [])
        base, im = cols_of(dev, base_pfx), cols_of(dev, img_pfx)
        pc = cols_of(dev, pca_pfx)
        # 요청한 접두어 그룹이 하나라도 비어 있으면 그 모델은 건너뜀
        if any(not cols_of(dev, [p]) for p in base_pfx + img_pfx + pca_pfx) or not (base or im or pc):
            continue
        if args.models and name not in args.models:
            continue
        if pc:
            opts["pca_cols"] = pc
        sets[name] = (base, im, opts)
    if not sets:
        print("[!] 특징 없음."); return 2
    print(f"[i] 개발 {len(dev)}명 (pCR {y.mean():.1%}) · 홀드아웃 {len(hold)}명"
          f" · radiomics {len(cols_of(dev, [RAD]))} · 동역학 {len(cols_of(dev, [KIN]))}")
    print(f"[i] 모델 {list(sets)} · 분류기 {cfg.CLASSIFIER} · {cfg.CV_FOLDS}-fold × {cfg.CV_REPEATS}회"
          f" (안쪽 {cfg.INNER_FOLDS}-fold 튜닝)")

    # ── 반복 nested CV (모든 모델이 같은 분할 공유) ─────────────────
    long_rows, fold_rows = [], []
    for r in range(cfg.CV_REPEATS):
        cv = StratifiedGroupKFold(cfg.CV_FOLDS, shuffle=True, random_state=r)
        for f, (tr, te) in enumerate(cv.split(dev, y, groups)):
            for name, (base, im, opts) in sets.items():
                cols = base + im + opts.get("pca_cols", []) + (["batch__id"] if opts.get("combat") else [])
                est = build(base, im, inner_seed=1000 * r + f, **opts)
                est.fit(dev.iloc[tr][cols], y[tr])
                p = est.predict_proba(dev.iloc[te][cols])[:, 1]
                long_rows += [{"pid": pid, "repeat": r, "fold": f, "model": name, "prob": pp}
                              for pid, pp in zip(groups[te], p)]
                fold_rows.append({"repeat": r, "fold": f, "model": name,
                                  "n_train": len(tr), "n_test": len(te),
                                  "auc": roc_auc_score(y[te], p), **chosen_params(est)})
        print(f"  반복 {r+1}/{cfg.CV_REPEATS} 완료", flush=True)

    long = pd.DataFrame(long_rows)
    # 앙상블: 구성 모델이 모두 이번 실행에 있으면 같은 반복·폴드 확률을 평균
    for ens, members in ENSEMBLES.items():
        if not all(mm in sets for mm in members):
            continue
        e = (long[long.model.isin(members)].groupby(["pid", "repeat", "fold"], as_index=False)["prob"].mean())
        e["model"] = ens
        long = pd.concat([long, e[long.columns]], ignore_index=True)
        ymap_ = dict(zip(groups, y))
        for (r_, f_), g in e.groupby(["repeat", "fold"]):
            fold_rows.append({"repeat": r_, "fold": f_, "model": ens,
                              "n_train": len(dev) - len(g), "n_test": len(g),
                              "auc": roc_auc_score(g.pid.map(ymap_), g.prob)})
        sets[ens] = ([], [], {})
    long.to_csv(cfg.EVAL_DIR / "06_oof_long.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(cfg.EVAL_DIR / "06_fold_auc.csv", index=False)

    # 반복별 pooled AUC 요약 + 반복 평균 OOF(wide, 참고용)
    ymap = dict(zip(groups, y))
    oof = dev[[cfg.COL_PID, "pCR", "subtype", "split"]].copy()
    rows = []
    print("\n=== 반복 CV AUC (개발셋, 반복별 pooled OOF) ===")
    for name in sets:
        g = long[long.model == name]
        aucs = np.array([roc_auc_score(d["pid"].map(ymap), d["prob"]) for _, d in g.groupby("repeat")])
        oof[f"prob__{name}"] = oof[cfg.COL_PID].map(g.groupby("pid")["prob"].mean())
        rows.append({"model": name, "n_features": len(sets[name][0]) + len(sets[name][1])
                     + len(sets[name][2].get("pca_cols", [])),
                     "auc_mean": aucs.mean(), "auc_std": aucs.std(),
                     "auc_p2.5": np.percentile(aucs, 2.5), "auc_p97.5": np.percentile(aucs, 97.5)})
        print(f"  {name:22s} AUC {aucs.mean():.3f} ± {aucs.std():.3f}  (특징 {rows[-1]['n_features']})")
    oof.to_csv(cfg.EVAL_DIR / "06_oof_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(cfg.EVAL_DIR / "06_cv_auc.csv", index=False)

    # ── 홀드아웃: 개발셋 전체로 (안쪽 CV 튜닝 포함) 학습 → 1회 예측 ─────
    # 개선안 탐색 중에는 --no-holdout: 홀드아웃을 보고 방법을 고르면 그 자체가 누수
    if args.no_holdout:
        stale = cfg.EVAL_DIR / "06_holdout_predictions.csv"
        if stale.exists():   # 이전 실행의 홀드아웃 예측이 이번 모델과 섞여 평가되지 않도록 표시
            stale.rename(stale.with_name("06_holdout_predictions.prev.csv"))
        print("\n[i] --no-holdout: 홀드아웃 예측 생략 (개발 CV만)")
    elif len(hold):
        hp = hold[[cfg.COL_PID, "pCR", "subtype", "split"]].copy(); prm = []
        for name, (base, im, opts) in sets.items():
            if name in ENSEMBLES:      # 앙상블은 구성 모델 예측의 평균 (아래)
                continue
            cols = base + im + opts.get("pca_cols", []) + (["batch__id"] if opts.get("combat") else [])
            est = build(base, im, inner_seed=cfg.RANDOM_STATE, **opts)
            est.fit(dev[cols], y)
            hp[f"prob__{name}"] = est.predict_proba(hold[cols])[:, 1]
            prm.append({"model": name, **chosen_params(est)})
        for ens, members in ENSEMBLES.items():
            if ens in sets:
                hp[f"prob__{ens}"] = hp[[f"prob__{mm}" for mm in members]].mean(axis=1)
        hp.to_csv(cfg.EVAL_DIR / "06_holdout_predictions.csv", index=False)
        pd.DataFrame(prm).to_csv(cfg.EVAL_DIR / "06_holdout_params.csv", index=False)
        print(f"\n[i] 홀드아웃 {len(hold)}명 예측 저장 (pCR {hold['pCR'].mean():.1%})")

    print(f"\n[✓] Evaluate/06_oof_long.csv, 06_fold_auc.csv, 06_oof_predictions.csv, 06_cv_auc.csv"
          + (", 06_holdout_*.csv" if len(hold) and not args.no_holdout else ""))
    print("    → python 07_evaluate.py (반복별 ΔAUC·corrected t-test·홀드아웃 DeLong·아형층화)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
