"""
00_config.py  ─ 전역 설정 · 경로 · 컬럼 정의  (2단계 파이프라인)
================================================================
연구 설계: **2단계만** — "임상 모델 vs 임상+영상 모델"로 영상의 증분 가치 검증.
  · 임상 baseline = 데이터에 있는 임상 변수 (HR·HER2·나이·인종·폐경)
  · 영상           = 치료 전 DCE-MRI radiomics
  · 헤드라인 지표  = DeLong (임상 vs 임상+영상 AUC 차이의 유의성)
  · 보조 점검      = 아형별 성능 분해 (영상이 아형만 베낀 건 아닌지)

폴더 규칙 (C:\\BreastCancer)
  Data/Original/   원본 (tarball, mask, metadata.csv)
  Data/Interim/    전처리 중간본 (subtraction, 정렬 마스크)
  Data/Features/   radiomics·분석 테이블
  Code1/           코드 (00_, 01_ ...)
  Evaluate/        지표·그림·리포트
"""
from __future__ import annotations
import os
from pathlib import Path

# ── 루트 ──────────────────────────────────────────────────────────
_ENV = os.environ.get("BREASTCANCER_ROOT")
PROJECT_ROOT = Path(_ENV).expanduser().resolve() if _ENV else Path(__file__).resolve().parent.parent

# ── 폴더 ──────────────────────────────────────────────────────────
CODE_DIR      = PROJECT_ROOT / "Code1"
DATA_DIR      = PROJECT_ROOT / "Data"
DATA_ORIGINAL = DATA_DIR / "Original"
DATA_INTERIM  = DATA_DIR / "Interim"
DATA_FEATURES = DATA_DIR / "Features"
EVAL_DIR      = PROJECT_ROOT / "Evaluate"
_ENSURE = [DATA_INTERIM, DATA_FEATURES, EVAL_DIR]

# ── 원본 파일 규약 (BreastDCEDL_ISPY2) ────────────────────────────
METADATA_CSV   = DATA_ORIGINAL / "BreastDCEDL_ISPY2_metadata.csv"

# 마스크: tar 해제 시 BreastDCEDL_ISPY2_mask/<pid>_<pid>_spy2_vis1_mask.nii.gz (pid 2번)
MASK_SUBDIR    = "BreastDCEDL_ISPY2_mask"
MASK_PATTERN   = "{pid}_{pid}_spy2_vis1_mask.nii.gz"   # vis1 = 치료 전

# DCE 영상 tarball (Original 및 Original/pcr 아래)
IMAGE_TARBALLS = {
    "nopcr": [DATA_ORIGINAL / f"BreastDCEDL_ISPY2_nopcr_dce_part{i}.tar.gz" for i in range(1, 9)],
    "pcr":   [DATA_ORIGINAL / "pcr" / f"BreastDCEDL_ISPY2_pcr_dce_part{i}.tar.gz" for i in range(1, 7)],
}
# 영상 NIfTI = phase(시점)별 개별 파일 (tar 목록으로 확인):
#   dce_partN/{pid}/{pid}_spy2_vis1_dce_aqc_{timepoint}.nii.gz   (pid 1번)
#     timepoint 0 = 조영전(pre), 1~n = 조영후
#   metadata의 pre/post_early/post_late 가 바로 이 timepoint 번호.
# ★ 파일명을 vis1까지 정확히 고정 — 치료 중/후(vis2+) 영상이 섞이는 것을 원천 차단.
DCE_TP_PATTERN = "{pid}_spy2_vis1_dce_aqc_{t}.nii.gz"

# ★ NIfTI 헤더에 물리 정보가 없음(affine=단위행렬, pixdim=1,1,1; 헤더로 확인).
#   SimpleITK 축 순서 = (slice, row, col) → spacing = (slice_thick, xy_spacing, xy_spacing)
#   을 metadata에서 복원해야 shape·질감 특징이 mm 단위가 된다. (tum_vol 과 일치 확인)
COL_SLICE_THICK = "slice_thick"
COL_XY_SPACING  = "xy_spacing"
COL_TUM_VOL     = "tum_vol"            # 마스크 부피(mm³) — 종양크기 baseline 용

# ── metadata 컬럼 (실제 헤더) ─────────────────────────────────────
COL_PID        = "pid"
COL_PCR        = "pCR"
COL_SUBTYPE    = "HR_HER2_STATUS"      # HER2pos / TripleNeg / HRposHER2neg
COL_SPLIT      = "test"                # 0=train(784) 1=val(99) 2=test(99)

# 4D DCE 시점 인덱스 (환자별, metadata에서 읽음)
COL_T_PRE      = "pre"                 # 조영 전 (대개 0)
COL_T_EARLY    = "post_early"          # 초기
COL_T_LATE     = "post_late"           # 후기 (subtraction에 사용)

# TCIA I-SPY2 Imaging Cohort 1 임상 파일 (CC BY 4.0, 985명) — BreastDCEDL에 없는 치료 전 정보:
#   Arm(배정 약제, 13종), MP(MammaPrint 0/1). pid 끝 숫자 = Patient_ID 로 982/982 매칭, pCR·HR·HER2 100% 일치 확인.
#   https://www.cancerimagingarchive.net/collection/ispy2/
TCIA_CLINICAL_XLSX = DATA_ORIGINAL / "tcia_clinical" / "ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx"

# ★ 임상 baseline 변수 (BPR 명시). 결측은 파이프라인에서 중앙값 대치.
CLINICAL_COLS  = ["HR", "HER2", "age", "race_white", "race_black", "menopause"]

# ── 모델링 설정 ───────────────────────────────────────────────────
RANDOM_STATE = 0
CV_FOLDS     = 5         # 바깥 CV (성능 추정)
CV_REPEATS   = 20
INNER_FOLDS  = 3         # 안쪽 CV (LASSO C 튜닝) — nested CV
N_BOOTSTRAP  = 2000
LASSO_C_GRID = [0.03, 0.1, 0.3, 1.0]   # 영상 특징 L1 선택 강도 후보 (안쪽 CV에서 선택)
N_JOBS       = -1        # 안쪽 GridSearch 병렬. 병렬 오류 시 1로

# 분류기 — 모든 모델이 같은 분류기를 써야 ΔAUC가 "특징 차이"만 반영한다.
#   "logreg": L2 로지스틱 (기본). 임상 6변수에 GBM은 과적합(아형 내부 AUC<0.5)이 확인됨.
#   "gbm"   : GradientBoosting (비선형 비교용 민감도 분석)
CLASSIFIER   = "logreg"

# ★ 홀드아웃: 내장 split 중 이 값들은 개발(CV·튜닝)에 절대 쓰지 않고 최종 1회 평가만.
#   튜닝은 nested CV가 맡으므로 val(1)도 홀드아웃에 합쳐 검정력 확보 (198명).
HOLDOUT_SPLITS = [1, 2]

# radiomics 추출 접두어
IMG_PREFIX   = "dce"

SUBTYPE_LABEL = {"HER2pos": "HER2+", "TripleNeg": "TNBC", "HRposHER2neg": "HR+HER2-"}

# ── 전처리 스위치 ─────────────────────────────────────────────────
# 영상 = voxel별 percent enhancement  PE = 100·(late−pre)/pre
#   · 비율이라 곱셈형 bias field가 상쇄 → N4 불필요 (pre/late 따로 N4 후 빼기는 왜곡 유발)
#   · 종양 ROI z-score는 조영 강도 자체를 지우므로 폐기. PE는 스캐너 이득과 무관한 % 단위.
DO_N4          = False
PE_FLOOR_FRAC  = 0.1      # 분모 하한 = 종양 내 pre 중앙값 × 이 값 (0 나눗셈 방지)
RESAMPLE_MM    = (1.0, 1.0, 1.0)   # (slice, row, col) 공통 격자
CROP_MARGIN_MM = 10.0     # 마스크 bounding box 주변 여유 (계산량 절감)

# radiomics 이산화: PE(%) 단위 고정 binWidth. 04가 환자별 bin 수를 QC로 남김 (권장 16~128).
BIN_WIDTH      = 10.0

def ensure_dirs() -> None:
    for d in _ENSURE:
        d.mkdir(parents=True, exist_ok=True)

def summary() -> str:
    return "\n".join([
        "=== BreastCancer (2단계 파이프라인) ===",
        f"ROOT          : {PROJECT_ROOT}",
        f"METADATA_CSV  : {METADATA_CSV}  (존재={METADATA_CSV.exists()})",
        f"임상 변수     : {CLINICAL_COLS}",
        f"CV            : {CV_FOLDS}-fold × {CV_REPEATS}회 (안쪽 {INNER_FOLDS}-fold 튜닝)",
        f"홀드아웃 split: {HOLDOUT_SPLITS}",
    ])

if __name__ == "__main__":
    ensure_dirs()
    print(summary())
