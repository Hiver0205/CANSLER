# BPR (Breast pCR Response) — 코드·결과 안내

> 팀 CANSLER · 치료 전 유방 DCE-MRI로 신보조 항암치료의 병리학적 완전관해(pCR)를 예측하는 연구 코드
> 데이터: I-SPY2 — BreastDCEDL_ISPY2(치료 전 MRI, 982명) + TCIA I-SPY2(치료 3주 차 MRI, 942명)

---

## 1. 큰 흐름

```
데이터 → 전처리 → 특징 추출 → 모델(+CNN) → 치료 3주 차 MRI 추가 분석 → 검증
```

| 단⁠계 | 폴⁠더 | 하⁠는 일 |
|---|---|---|
| 설⁠정 | `00) Settings` | 경⁠로⁠·⁠임⁠상 변⁠수 등 공⁠통 설⁠정 |
| 데⁠이⁠터 | `01) Data` | 데⁠이⁠터 통⁠계 확⁠인⁠, 영⁠상 압⁠축 해⁠제 |
| 전⁠처⁠리 | `02) Preprocessing` | 화⁠소 크⁠기 복⁠원⁠, 종⁠양 크⁠롭⁠, 조⁠영 증⁠강⁠률 변⁠환⁠, 1 mm 리⁠샘⁠플⁠링 |
| 특⁠징 추⁠출 | `03) Feature Extraction` | 동⁠역⁠학⁠·⁠종⁠양 주⁠변⁠·⁠시⁠간 곡⁠선⁠·habitat·⁠질⁠감 특⁠징⁠, CNN 입⁠력 |
| 모⁠델 | `04) Analysis Table` → `05) Model Training` → `06) CNN Training` → `07) Evaluation` | 분⁠석 테⁠이⁠블 → 로⁠지⁠스⁠틱 회⁠귀 14⁠개 → ResNet-18 → 교⁠차⁠검⁠증 평⁠가 |
| 치⁠료 3⁠주 차 MRI 추⁠가 분⁠석 | `08) Week-3 MRI Analysis` | 3⁠주 차 MRI 다⁠운⁠로⁠드 → 특⁠징 → 모⁠델 → CNN 결⁠합 |
| 검⁠증 | `09) Final Validation` | 내⁠부 검⁠증⁠(198⁠명⁠) 1⁠회 평⁠가 |
| 웹⁠앱 | `10) Web` | 의⁠료⁠진⁠용 pCR 예⁠측 웹⁠앱 (⁠영⁠상 업⁠로⁠드 → 분⁠석 → 결⁠과⁠·⁠근⁠거 표⁠시⁠) |

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


### 10) Web — 의료진용 예측 웹앱

최종 모델(치료 전 · 3주 차)을 의료진이 화면에서 쓸 수 있게 만든 Streamlit 웹앱.

**20) Web_App, Screen Layout.py**
- 화면 전체: 의료진 로그인 · 환자 목록(검색·아형 필터·환자 등록) · 진료 대기자(최대 5명, 끌어서 순서 변경, 상태 표시).
- 가운데: 치료 전 / 3주 차 MRI 선택, 2D 단면(조영 시점 막대) · 3D 재구성, 정량 수치(부피·최대 직경·시간-신호 곡선·종양 분석 결과).
- 오른쪽: pCR 도달 확률(원형 게이지, 같은 아형 평균 눈금), 판단 근거, 의사 소견, 결과 저장(CSV), Print(간편 비밀번호 확인).

**21) Web_Inference, Pre and Week-3.py**
- 업로드한 DICOM에서 특징 추출 → 최종 모델(로지스틱 회귀 앙상블 + CNN) 확률 계산.
- 치료 전 결과를 저장해 두고 3주 차 분석에서 다시 씀(속도 개선). 단계별 진행률 표시.

**22) Web_Core, Patients, Queue, Image Stats.py**
- 환자 목록·등록, 분석 실행·결과 저장, 진료 대기자 목록, 의사 소견 저장.
- 영상 수치: 종양 부피, 최대 직경(볼록 껍질), 시간-신호 곡선과 곡선 유형, SER·유출 영역 비율.

**23) Web_Accounts, Login, PIN.py**
- 의료진 계정: 비밀번호 PBKDF2-SHA256 해시, 6자리 간편 비밀번호, 30분 미사용 시 자동 로그아웃, 5회 실패 시 5분 잠금.
- 계정 만들기는 기관 등록 코드 필요 (`.streamlit/secrets.toml` 의 `[auth] signup_code`, 저장소에 올리지 않음).

**24) Web_Batch Precompute.py**
- 내부 검증 환자의 예측을 미리 계산해 두는 일괄 실행.

**기타:** `run_web.bat`(실행), `assets/`(로고), `.streamlit/config.toml`(화면 색·테마).

> **올리지 않은 파일:** 환자 단위 자료(`Web/cache/`, 업로드 영상), 계정·세션 정보, `secrets.toml`, 환자 이름 파일. 실행하려면 원래 이름(`app.py`, `21_infer.py`, `bpr_core.py`, `auth.py`, `precompute.py`)으로 되돌려 `Web/`(21_infer.py는 `Code1/`)에 두어야 합니다.

---

## 3. 실행 순서

| 순⁠서 | 실⁠행 |
|---|---|
| 1 | 00 → 01 → 02 → 03 |
| 2 | 04 → 05 → 06 → 07 → 08 → 09 (⁠옵⁠션⁠별 여⁠러 번⁠) |
| 3 | 10 → 11 → 12 → 13 |
| 4 | 14 → 15 → 16 → 09·12 (`--root T1`) → 17 → 18 |
| 5 | 12 `--final` · 17 `--source full --final` → 19 |

> **주의:** 이 폴더의 코드는 보기 쉽도록 이름을 바꾼 사본입니다. 코드 안에서는 원래 파일 이름(`00_config`, `03_preprocess`, `06_train_cv`, `07_evaluate` 등)으로 서로를 불러오므로, 그대로 실행하려면 원래 이름으로 한 폴더에 두어야 합니다. 경로는 `00) Settings`에서 설정합니다.

| 새 이⁠름 번⁠호 | 원⁠래 파⁠일 |
|---|---|
| 00 | `00_config.py` |
| 01 / 02 | `01_explore_metadata.py` / `02_extract_images.py` |
| 03 | `03_preprocess.py` |
| 04 / 05 / 06 / 07 / 08 | `03b_kinetic_features.py` / `03c_peritumoral.py` / `03e_timecurve.py` / `03f_habitat.py` / `03g_timestack.py` |
| 09 | `04_extract_radiomics.py` |
| 10 | `05_build_features.py` |
| 11 | `06_train_cv.py` |
| 12 | `04c_cnn_finetune.py` |
| 13 | `07_evaluate.py` |
| 14 / 15 / 16 | `09_t1_download.py` / `10_t1_features.py` / `10b_t1_maps.py` |
| 17 / 18 | `08_exp3_early_response.py` / `08b_exp3_cnn_ensemble.py` |
| 19 | `11_final_holdout.py` |
| 20 / 21 / 22 / 23 / 24 | `Web/app.py` / `Code1/21_infer.py` / `Web/bpr_core.py` / `Web/auth.py` / `Web/precompute.py` |

환경: Python 3.11, Windows 11, NVIDIA CUDA — 패키지는 `Code/requirements.txt`.

---

## 4. 결과 파일 설명 (`Results/`)

파일 이름 앞 번호 = 그 결과를 만든 코드 번호. CSV는 Excel에서 한글이 깨지지 않도록 UTF-8(BOM)으로 저장.

**모델별 결과 파일의 정렬 규칙** — `구분` 열로 세 단으로 나누고, 각 단 안에서는 성능(AUC 또는 ΔAUC) 높은 순.

| 구⁠분 | 뜻 |
|---|---|
| ① 주⁠요 결⁠과 모⁠델 | 포⁠스⁠터⁠·⁠최⁠종 보⁠고⁠에 나⁠온 모⁠델 (⁠임⁠상⁠, 임⁠상 + 부⁠피⁠, + 동⁠역⁠학⁠, 14⁠관⁠점 앙⁠상⁠블⁠, 최⁠종 모⁠델 등⁠) |
| ② 최⁠종 앙⁠상⁠블 구⁠성 모⁠델 | 포⁠스⁠터⁠에 직⁠접 나⁠오⁠지 않⁠지⁠만 최⁠종 모⁠델⁠의 평⁠균⁠에 들⁠어⁠간 로⁠지⁠스⁠틱 회⁠귀 모⁠델 |
| ③ 미⁠채⁠택⁠·⁠참⁠고 모⁠델 | 시⁠도⁠했⁠지⁠만 성⁠능 향⁠상⁠이 없⁠어 최⁠종 모⁠델⁠에⁠서 뺀 모⁠델⁠, 또⁠는 비⁠교⁠용 참⁠고 모⁠델 |

| 파⁠일 | 내⁠용 |
|---|---|
| `01) Data_Subtype pCR Rate.csv` | 아⁠형⁠별 환⁠자 수⁠·pCR 수⁠·pCR⁠률 |
| `07) Feature_Habitat Cluster Centers (K3, K4).csv` | 최⁠종 모⁠델⁠에 쓴 habitat 유⁠형⁠(K=3·4)⁠별 곡⁠선 특⁠징 중⁠심⁠값⁠(⁠유⁠형 해⁠석⁠용⁠) |
| `13) Evaluation_CV Performance by Model.csv` | 교⁠차⁠검⁠증⁠(784⁠명⁠, 5-fold × 20⁠회⁠) 모⁠델⁠별 특⁠징 수⁠·AUC(⁠평⁠균⁠·⁠범⁠위⁠)·PR-AUC·Brier |
| `13) Evaluation_CV Delta AUC.csv` | 교⁠차⁠검⁠증 모⁠델 비⁠교⁠: ΔAUC, 향⁠상⁠된 반⁠복 비⁠율⁠, 보⁠정 t-⁠검⁠정 p⁠값 |
| `13) Evaluation_CV Subtype AUC.csv` | 교⁠차⁠검⁠증 아⁠형⁠별 AUC (⁠모⁠델 순⁠서⁠는 위 성⁠능 파⁠일⁠과 같⁠음⁠) |
| `17) Week3_CV Performance by Model.csv` | 치⁠료 3⁠주 차 MRI 추⁠가 분⁠석 교⁠차⁠검⁠증⁠(745⁠명⁠) 모⁠델⁠별 AUC |
| `17) Week3_CV Delta AUC.csv` | 3⁠주 차 모⁠델 비⁠교⁠: ΔAUC, 향⁠상⁠된 반⁠복 비⁠율⁠, p⁠값 |
| `18) Week3_CV CNN Ensemble AUC by Repeat.csv` | 반⁠복⁠별 3⁠주 차 관⁠점 앙⁠상⁠블 + CNN 결⁠합 AUC⁠와 평⁠균 |
| `19) Validation_Internal Validation Performance.csv` | **최⁠종 결⁠과** — 내⁠부 검⁠증⁠(198⁠명⁠) AUC·95% CI·⁠임⁠상 대⁠비 ΔAUC·DeLong p |
| `19) Validation_Internal Validation Subtype AUC.csv` | 내⁠부 검⁠증 아⁠형⁠별 AUC·95% CI |

참고
- `임상 + 부피`(0.689)는 원래 결과 파일에 없어, 같은 분할로 다시 계산해 추가함(임상 0.656·+동역학 0.699도 함께 재계산해 원본과 같은지 확인).
- 3주 차 관점 앙상블(0.768)과 + CNN(0.770)은 CNN을 5회만 반복 학습해 **반복 5회 평균**이고, 범위는 최소~최대. 나머지 교차검증은 20회 반복, 범위는 2.5~97.5%.
- ③의 곡선 유형 비율(`bag`) 모델은 코드(`03h`)를 이 저장소에서 뺐지만, 시도 기록으로 결과는 남겨 둠.

**Figures/** (그림 1~6은 포스터 그림, 흰 배경으로 변환)

| 파⁠일 | 내⁠용 |
|---|---|
| `Fig1) Data_Subtype pCR Rate.png` | 아⁠형⁠별 pCR⁠률 (HER2+ 46%, TNBC 39%, HR+HER2− 17%) |
| `Fig2) Model Architecture.png` | 모⁠델 구⁠조 (⁠로⁠지⁠스⁠틱 회⁠귀 14⁠개 + ResNet-18 2⁠개 확⁠률 결⁠합⁠) |
| `Fig3) CV AUC Step-up.png` | 교⁠차⁠검⁠증 AUC 단⁠계⁠별 향⁠상 (⁠임⁠상 0.656 → 3⁠주 차 0.770) |
| `Fig4) Internal Validation AUC.png` | 내⁠부 검⁠증 AUC (⁠치⁠료 전 0.757, 3⁠주 차 0.816) |
| `Fig5) Internal Validation Subtype AUC.png` | 내⁠부 검⁠증 아⁠형⁠별 AUC |
| `Fig6) Internal Validation ROC Curve.png` | 내⁠부 검⁠증 ROC 곡⁠선 (⁠임⁠상 vs 최⁠종 모⁠델⁠) |
| `Fig7) Internal Validation Calibration.png` | 내⁠부 검⁠증 보⁠정 곡⁠선 (⁠임⁠상 vs 최⁠종 모⁠델⁠, Brier 0.201 → 0.183 → 0.163) |

---

## 5. 포함하지 않은 것

- **영상 데이터:** 용량(약 180 GB)과 데이터 이용 약관 때문에 제외. BreastDCEDL·TCIA에서 직접 받아야 합니다.
- **환자별 예측값 파일:** 요약 결과만 포함.
- **시도했지만 최종 모델에 쓰지 않은 코드:** 종양 형태 특징(`03d`), 곡선 유형 비율(`03h`), 고정 CNN 특징(`04b`), 외부 데이터 CNN 학습(`04d`). 원본은 `Code1/`에 있습니다.
