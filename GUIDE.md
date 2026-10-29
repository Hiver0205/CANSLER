# BPR (Breast pCR Prediction) — 코드·결과 안내

> 팀 CANSLER · 치료 전 유방 DCE-MRI로 신보조 항암치료의 병리학적 완전관해(pCR)를 예측하는 연구 코드
> 데이터: I-SPY2 — BreastDCEDL_ISPY2(치료 전 MRI, 982명) + TCIA I-SPY2(치료 3주 차 MRI, 942명)

---

## 1. 큰 흐름

```
데이터 → 전처리 → 특징 추출 → 모델(+CNN) → 치료 3주 차 MRI 추가 분석 → 검증
```

| 단계 | 폴더 | 하는 일 |
|---|---|---|
| 설정 | `00) Settings` | 경로·임상 변수 등 공통 설정 |
| 데이터 | `01) Data` | 데이터 통계 확인, 영상 압축 해제 |
| 전처리 | `02) Preprocessing` | 화소 크기 복원, 종양 크롭, 조영 증강률 변환, 1 mm 리샘플링 |
| 특징 추출 | `03) Feature Extraction` | 동역학·종양 주변·시간 곡선·habitat·질감 특징, CNN 입력 |
| 모델 | `04) Analysis Table` → `05) Model Training` → `06) CNN Training` → `07) Evaluation` | 분석 테이블 → 로지스틱 회귀 14개 → ResNet-18 → 교차검증 평가 |
| 치료 3주 차 MRI 추가 분석 | `08) Week-3 MRI Analysis` | 3주 차 MRI 다운로드 → 특징 → 모델 → CNN 결합 |
| 검증 | `09) Final Validation` | 내부 검증(198명) 1회 평가 |

**파일 이름 규칙:** `실행 순서) 단계_핵심 내용.py` (예: `00) Settings_Path, Clinical Variables.py`)
**검증 방식:** 982명을 학습 784명 / 내부 검증 198명으로 나눔. 학습 환자로만 5-fold × 20회 교차검증으로 모델을 정하고, 내부 검증 환자는 마지막에 한 번만 평가.

---

## 2. 폴더별 코드 설명 (`Code/`)

### 00) Settings — 설정

**00) Settings_Path, Clinical Variables.py**
- 모든 코드가 공통으로 불러오는 설정 파일.
- 데이터·중간 결과·특징·평가 폴더 경로와 metadata 컬럼 이름 정의.
- 임상 기준 변수 6개 정의: HR, HER2, 나이, 인종(백인·흑인), 폐경.

### 01) Data — 데이터

**01) Data_Metadata Summary, Subtype pCR Rate.py**
- metadata로 pCR 분포, 아형별 pCR률, 임상 변수 결측, 내부 검증 분할 확인.
- 아형별 pCR률(HR+/HER2− 16.8%, HER2+ 45.9%, TNBC 39.3%) 산출.
- 모델 결과에는 영향 없음(데이터 통계용).

**01) Data_Image Extraction, NIfTI Unpack.py**
- BreastDCEDL_ISPY2 DCE 영상 압축 파일(pCR·non-pCR)과 종양 마스크 압축 해제(약 54 GB).
- `--proto`로 일부만 먼저 풀어 파일 이름 규칙 확인 가능.
- 압축 해제 결과를 로그로 기록(QC).

### 02) Preprocessing — 전처리

**02) Preprocessing_Spacing Restore, Crop, PE-SER, Resampling.py**
- NIfTI 헤더에 화소 크기 정보가 없어 metadata의 슬라이스 두께·평면 해상도로 voxel spacing 복원.
- 종양 마스크 주변 +10 mm 크롭 → 조영 증강률(PE, %)·SER 변환 → 1 mm 등방 리샘플링.
- N4 보정·z-score는 사용하지 않음(밝기 편차가 증강률 비율에서 상쇄됨).
- 출력: 환자별 `pe`, `pe_early`, `ser`, `mask` 영상.

### 03) Feature Extraction — 특징 추출

**04) Feature_Kinetic, FTV, SER.py**
- 조영제가 들어오고 빠지는 패턴을 요약한 동역학 특징.
- 시점별 증강률, SER 유형(washout·plateau·persistent) 비율, 최고 증강·도달 시점.
- 종양 바깥 고리(ring) 영역 증강도 함께 계산.

**05) Feature_Peritumoral, BPE.py**
- 종양 경계로부터 거리별 고리(0–5·5–10·10–20·20–40 mm) 영역의 조영 특징.
- 가장 바깥 고리는 배경 실질 증강(BPE) 근사값으로 사용.

**06) Feature_Time Curve Maps, Wash-in, Wash-out.py**
- 모든 조영 후 시점(4–11개)을 사용해 voxel별 시간-증강 곡선 생성.
- 유입 속도(wash-in), 유출 속도(wash-out), 곡선 아래 면적, 초기 비율 지도 저장.
- 이 지도들은 Radiomics 질감(09)과 habitat(07)의 입력. 성능 향상의 핵심 특징.

**07) Feature_Habitat, K-means.py**
- 종양 voxel을 곡선 모양 4특징으로 K-means 군집(K=3·4) → 종양 내부를 조영 패턴별 영역으로 분할.
- 영역별 비율·부피·위치(가장자리 vs 중심)·조각 수를 환자 특징으로 사용.
- 군집 기준은 학습 환자 voxel로만 학습(내부 검증 환자는 배정만).

**08) Feature_Time Stack, CNN Input 6ch.py**
- 환자마다 다른 조영 시점 수를 상대 시간 6지점으로 보간한 6채널 증강 영상 생성.
- CNN(12)의 입력.

**09) Feature_Radiomics Texture, PyRadiomics.py**
- PyRadiomics(IBSI)로 형태·1차 통계·질감 특징 추출.
- 옵션으로 입력 지도를 바꿔 여러 번 실행: 후기 PE, 초기 PE, SER, 시간 곡선 지도, habitat 영역, 종양 주변 고리.
- `--root`로 3주 차 MRI 지도에도 같은 방식으로 적용.

### 04) Analysis Table — 분석 테이블

**10) Table_Clinical, Imaging Features, pCR Label.py**
- metadata의 임상 변수·pCR 라벨과 모든 영상 특징을 환자 ID로 병합.
- 학습/내부 검증 분할 표시를 포함한 단일 분석 테이블(`analysis_table.csv`) 생성.
- 모델 학습(11)·CNN(12)·추가 분석(17)이 모두 이 테이블을 사용.

### 05) Model Training — 모델 학습

**11) Model_Nested CV, Logistic Regression, 14 Views.py**
- 학습 환자 784명으로 5-fold × 20회 반복 교차검증(모든 모델이 같은 분할 사용).
- 모델: L1으로 특징 선택 → L2 로지스틱 회귀. 전처리·선택은 폴드 안에서만 학습(데이터 누수 방지).
- 특징 묶음을 달리한 관점별 모델 14개 학습 → 확률 평균으로 결합.
- 임상 기준 모델과 같은 분할에서 비교 가능한 예측값 저장.

### 06) CNN Training — CNN 학습

**12) CNN_ResNet-18 Fine-tuning, 2.5D, 3D.py**
- 사전학습 ResNet-18을 pCR 라벨로 미세조정: 2.5D(ImageNet, 종양 슬라이스) + 3D(r3d_18, Kinetics).
- 입력은 6채널 시간 증강 영상(08). 증강(뒤집기·90° 회전·밝기·크롭), 예측 시 8방향 TTA, 3개 시드 평균.
- 학습 환자의 CNN 점수는 안쪽 3-fold 예측으로 만들어 누수 방지.
- `--final`: 학습 환자 전체로 학습해 내부 검증 예측 저장. `--root T1`: 3주 차 MRI용.

### 07) Evaluation — 결과 평가

**13) Evaluation_CV AUC, Delta AUC, Subtype AUC.py**
- 교차검증 결과로 모델별 AUC·PR-AUC·Brier와 반복 범위 계산.
- 임상 기준 대비 ΔAUC와 보정된 t-검정 p값(반복 CV의 과신 방지).
- 아형별 AUC, 보정 곡선(calibration) 그림.
- 결과: 임상 0.656 → 임상+영상 0.715.

### 08) Week-3 MRI Analysis — 치료 3주 차 MRI 추가 분석

**14) Week3_TCIA T1 DICOM Download.py**
- TCIA 공개 API로 같은 I-SPY2 환자의 치료 3주 차(T1) DCE 원본과 분석 마스크(DICOM) 다운로드.
- 중단되어도 이어받기 가능. 다운로드 로그·목록 저장.

**15) Week3_T1 Features, FTV Change.py**
- T1 DICOM을 읽어 치료 전과 같은 정의로 특징 계산(시간 곡선, habitat 비율 등).
- 치료 전 대비 종양 부피(FTV) 변화율, 종양 소실 여부, 특징 변화량 추가.

**16) Week3_T1 Maps, Radiomics-CNN Input.py**
- T1 영상을 치료 전과 같은 형식(크롭, 1 mm, NIfTI)으로 저장.
- 이후 Radiomics(09)와 CNN(12)을 `--root T1`으로 그대로 재사용.

**17) Week3_Early Response Model, CV.py**
- 모델 학습(11)과 같은 방식으로 3주 차 특징을 더한 모델 교차검증(T1이 있는 학습 환자 745명).
- 결과: 임상 0.653 → +FTV 변화 0.742 → +T1 전체 0.764.
- `--final`: 학습 환자 전체로 학습해 내부 검증 예측 저장.

**18) Week3_CNN Ensemble.py**
- 3주 차 모델들의 확률 평균에 T1 CNN 확률을 더해 비교(추가 학습 없음).
- 결과: T1 앙상블 0.768 → +CNN 0.770.

### 09) Final Validation — 최종 검증

**19) Validation_Internal Validation, Final Model.py**
- 미리 고정한 모델만 학습 환자 전체로 학습해 내부 검증 198명에서 1회 평가.
- AUC와 부트스트랩 95% CI, 임상 대비 ΔAUC(DeLong), PR-AUC, Brier, 아형별 AUC.
- 결과: 치료 전 0.641 → **0.757** (ΔAUC +0.116, p=0.0001) · 3주 차 포함 0.636 → **0.816** (197명).

---

## 3. 실행 순서

| 순서 | 실행 |
|---|---|
| 1 | 00 → 01 → 02 → 03 |
| 2 | 04 → 05 → 06 → 07 → 08 → 09 (옵션별 여러 번) |
| 3 | 10 → 11 → 12 → 13 |
| 4 | 14 → 15 → 16 → 09·12 (`--root T1`) → 17 → 18 |
| 5 | 12 `--final` · 17 `--source full --final` → 19 |

> **주의:** 이 폴더의 코드는 보기 쉽도록 이름을 바꾼 사본입니다. 코드 안에서는 원래 파일 이름(`00_config`, `03_preprocess`, `06_train_cv`, `07_evaluate` 등)으로 서로를 불러오므로, 그대로 실행하려면 원래 이름으로 한 폴더에 두어야 합니다. 경로는 `00) Settings`에서 설정합니다.

| 새 이름 번호 | 원래 파일 |
|---|---|
| 00 | `00_config.py` |
| 01 / 02 | `01_explore_metadata.py` / `02_extract_images.py` |
| 03 | `03_preprocess.py` |
| 04 / 05 / 06 / 07 / 08 | `03b_kinetic_features.py` / `03c_peritumoral.py` / `03e_timecurve.py` / `03f_habitat.py` / `03g_timestack.py` |
| 09 | `04_extract_radiomics.py` |
| 10 | `05_build_features.py` |
| 11 | `06_train_cv.py` |
| 12 | `04c_cnn_finetune.py` |
| 13 | `07_evaluate.py` |
| 14 / 15 / 16 | `09_t1_download.py` / `10_t1_features.py` / `10b_t1_maps.py` |
| 17 / 18 | `08_exp3_early_response.py` / `08b_exp3_cnn_ensemble.py` |
| 19 | `11_final_holdout.py` |

환경: Python 3.11, Windows 11, NVIDIA CUDA — 패키지는 `Code/requirements.txt`.

---

## 4. 결과 파일 설명 (`Results/`)

파일 이름 앞 번호 = 그 결과를 만든 코드 번호. CSV는 Excel에서 한글이 깨지지 않도록 UTF-8(BOM)으로 저장.

**모델별 결과 파일의 정렬 규칙** — `구분` 열로 세 단으로 나누고, 각 단 안에서는 성능(AUC 또는 ΔAUC) 높은 순.

| 구분 | 뜻 |
|---|---|
| ① 주요 결과 모델 | 포스터·최종 보고에 나온 모델 (임상, 임상 + 부피, + 동역학, 14관점 앙상블, 최종 모델 등) |
| ② 최종 앙상블 구성 모델 | 포스터에 직접 나오지 않지만 최종 모델의 평균에 들어간 로지스틱 회귀 모델 |
| ③ 미채택·참고 모델 | 시도했지만 성능 향상이 없어 최종 모델에서 뺀 모델, 또는 비교용 참고 모델 |

| 파일 | 내용 |
|---|---|
| `01) Data_Subtype pCR Rate.csv` | 아형별 환자 수·pCR 수·pCR률 |
| `07) Feature_Habitat Cluster Centers (K3, K4).csv` | 최종 모델에 쓴 habitat 유형(K=3·4)별 곡선 특징 중심값(유형 해석용) |
| `13) Evaluation_CV Performance by Model.csv` | 교차검증(784명, 5-fold × 20회) 모델별 특징 수·AUC(평균·범위)·PR-AUC·Brier |
| `13) Evaluation_CV Delta AUC.csv` | 교차검증 모델 비교: ΔAUC, 향상된 반복 비율, 보정 t-검정 p값 |
| `13) Evaluation_CV Subtype AUC.csv` | 교차검증 아형별 AUC (모델 순서는 위 성능 파일과 같음) |
| `17) Week3_CV Performance by Model.csv` | 치료 3주 차 MRI 추가 분석 교차검증(745명) 모델별 AUC |
| `17) Week3_CV Delta AUC.csv` | 3주 차 모델 비교: ΔAUC, 향상된 반복 비율, p값 |
| `18) Week3_CV CNN Ensemble AUC by Repeat.csv` | 반복별 3주 차 관점 앙상블 + CNN 결합 AUC와 평균 |
| `19) Validation_Internal Validation Performance.csv` | **최종 결과** — 내부 검증(198명) AUC·95% CI·임상 대비 ΔAUC·DeLong p |
| `19) Validation_Internal Validation Subtype AUC.csv` | 내부 검증 아형별 AUC·95% CI |

참고
- `임상 + 부피`(0.689)는 원래 결과 파일에 없어, 같은 분할로 다시 계산해 추가함(임상 0.656·+동역학 0.699도 함께 재계산해 원본과 같은지 확인).
- 3주 차 관점 앙상블(0.768)과 + CNN(0.770)은 CNN을 5회만 반복 학습해 **반복 5회 평균**이고, 범위는 최소~최대. 나머지 교차검증은 20회 반복, 범위는 2.5~97.5%.
- ③의 곡선 유형 비율(`bag`) 모델은 코드(`03h`)를 이 저장소에서 뺐지만, 시도 기록으로 결과는 남겨 둠.

**Figures/** (그림 1~6은 포스터 그림, 흰 배경으로 변환)

| 파일 | 내용 |
|---|---|
| `Fig1) Data_Subtype pCR Rate.png` | 아형별 pCR률 (HER2+ 46%, TNBC 39%, HR+HER2− 17%) |
| `Fig2) Model Architecture.png` | 모델 구조 (로지스틱 회귀 14개 + ResNet-18 2개 확률 결합) |
| `Fig3) CV AUC Step-up.png` | 교차검증 AUC 단계별 향상 (임상 0.656 → 3주 차 0.770) |
| `Fig4) Internal Validation AUC.png` | 내부 검증 AUC (치료 전 0.757, 3주 차 0.816) |
| `Fig5) Internal Validation Subtype AUC.png` | 내부 검증 아형별 AUC |
| `Fig6) Internal Validation ROC Curve.png` | 내부 검증 ROC 곡선 (임상 vs 최종 모델) |
| `Fig7) Internal Validation Calibration.png` | 내부 검증 보정 곡선 (임상 vs 최종 모델, Brier 0.201 → 0.183 → 0.163) |

---

## 5. 포함하지 않은 것

- **영상 데이터:** 용량(약 180 GB)과 데이터 이용 약관 때문에 제외. BreastDCEDL·TCIA에서 직접 받아야 합니다.
- **환자별 예측값 파일:** 요약 결과만 포함.
- **시도했지만 최종 모델에 쓰지 않은 코드:** 종양 형태 특징(`03d`), 곡선 유형 비율(`03h`), 고정 CNN 특징(`04b`), 외부 데이터 CNN 학습(`04d`). 원본은 `Code1/`에 있습니다.
