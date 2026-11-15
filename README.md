<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/bpr-logo-response-dark.png" />
  <img src="assets/bpr-logo-response-light.png" width="240" alt="BPR · Breast pCR Response" />
</picture>

<h2>BPR — Breast pCR Response</h2>

<p>
<b>치료 전 유방 MRI로 신보조 항암치료의 완전관해(pCR)를 예측하고,<br/>
영상이 기존 임상 정보를 넘어 예측에 기여하는지 검증합니다.</b>
</p>

<p>
<sub><i>Predicting pathologic complete response (pCR) to neoadjuvant chemotherapy in breast cancer
from pre-treatment DCE-MRI, and testing whether imaging adds value beyond clinical variables.</i></sub>
</p>

<p>
<img src="https://img.shields.io/badge/Python-3.11-b40e26?style=flat-square&labelColor=1f1e22" />
<img src="https://img.shields.io/badge/PyTorch-ResNet--18-b40e26?style=flat-square&labelColor=1f1e22" />
<img src="https://img.shields.io/badge/scikit--learn-Logistic_Regression-b40e26?style=flat-square&labelColor=1f1e22" />
<img src="https://img.shields.io/badge/PyRadiomics-IBSI-b40e26?style=flat-square&labelColor=1f1e22" />
<img src="https://img.shields.io/badge/Streamlit-Web_App-b40e26?style=flat-square&labelColor=1f1e22" />
<img src="https://img.shields.io/badge/I--SPY2-982_patients-b40e26?style=flat-square&labelColor=1f1e22" />
</p>

<p>
<img src="https://img.shields.io/badge/%EB%82%B4%EB%B6%80%20%EA%B2%80%EC%A6%9D%20AUC-%EC%B9%98%EB%A3%8C%20%EC%A0%84%200.757%20%C2%B7%203%EC%A3%BC%20%EC%B0%A8%200.816-b40e26?style=for-the-badge&labelColor=1f1e22" alt="내부 검증 AUC: 치료 전 0.757 · 3주 차 0.816" />
</p>

<p>
<a href="#problem">문제 정의</a> ·
<a href="#approach">접근 방법</a> ·
<a href="#data">데이터셋</a> ·
<a href="#results">결과</a> ·
<a href="#code">코드 구성</a> ·
<a href="#web">웹앱 · 결과 읽는 법</a> ·
<a href="#method">방법론 · 일정</a> ·
<a href="#team">팀 소개</a>
</p>

</div>

<br/>

<h2 id="problem">01. 문제 정의</h2>

**pCR(병리학적 완전관해)** 은 신보조 항암(수술 전 항암) 후 수술로 떼어 낸 조직에서 침습 암세포가 완전히 사라진 상태입니다.
pCR에 도달한 환자는 그렇지 않은 환자보다 5년 생존·무병생존이 뚜렷이 높습니다(예: 5년 OS 88% vs 58%).
그러나 pCR 여부는 **수술 후 병리 검사로만** 확인할 수 있습니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/problem-stats-dark.png" />
  <img src="assets/problem-stats-light.png" width="860" alt="pCR 도달 22%, 생검-수술조직 아형 불일치 16.4%, HR+/HER2- 아형 pCR률 16.8%" />
</picture>

| # | 문제 | 핵심 근거 |
|:---:|:---|:---|
| **①** | **치료 반응의 불확실성** — pCR 도달은 소수(약 22%)에 그치고, 수술 후 병리로만 확정되어 치료 전에는 알 수 없음 | Antonini et al. (2023); Fang et al. (2024) |
| **②** | **침습적 검사 의존** — 정밀 예측 지표(Ki-67·유전자 등)는 생검 기반인데, 생검은 종양 일부만 떼어 전체를 대표하지 못함 (생검–수술조직 아형 불일치 **16.4%**, 저Ki-67 종양 **32.0%**) | Xu et al. (2025) |
| **③** | **기존 예측 도구의 한계** — 상용 도구(Ataraxis = 병리영상, MammaPrint = 유전자)는 모두 조직 기반·침습이며 비용·접근성 제약, **비침습 MRI 기반 도구는 부재** | Clinical Lab Products (2025); Agendia (2024) |

<br/>

<h2 id="approach">02. 접근 방법</h2>

### 핵심 연구 질문

> **"비침습 MRI 영상이, 아형 등 기존 임상 정보에 *더해* pCR 예측에 추가로 기여하는가?"**

- **비교** — 임상 정보(HR·HER2·나이·인종·폐경) 모델 **vs** 임상 정보 + 영상 모델
- **아형 내 검증** — 같은 아형 안에서도 영상이 pCR을 구별하는지 아형별 성능을 따로 평가
- **추가 분석** — 치료 3주 차 MRI를 더하면 예측이 얼마나 더 좋아지는가

### 분석 흐름

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/analysis-pipeline-dark.png" />
  <img src="assets/analysis-pipeline-light.png" width="860" alt="분석 흐름: 데이터 → 전처리 → 특징 추출 → 예측 모델 → 3주 차 MRI → 내부 검증" />
</picture>

### 영상 특징 (총 801개)

| 그룹 | 개수 | 내용 |
|:---|:---:|:---|
| 종양 부피 | 1 | 조영 증강 기준 기능적 종양 부피(FTV) |
| 시간 곡선 | 56 | 조영제가 들어오고 빠지는 속도, 곡선 아래 면적 등 (3시점 동역학 + 전체 시점 곡선) |
| Habitat | 37 | 종양 내부를 조영 패턴별 영역으로 나눈 비율·위치 (K-means) |
| 질감 | 665 | PyRadiomics 질감 — 조영 증강률·SER·시간 곡선 지도 |
| 종양 주변 | 42 | 종양 경계 바깥 0–40 mm 고리 영역과 배경 실질 증강 |

### 예측 모델

- **로지스틱 회귀 14개** — 특징 묶음(관점)을 달리해 학습. L1으로 특징을 고른 뒤 L2 로지스틱 회귀로 예측
- **ResNet-18 2개** — 사전학습 CNN을 6채널 시간 증강 영상으로 미세조정 (2.5D·3D)
- **결합** — 16개 모델의 pCR 확률을 **평균**해 최종 예측
- **치료 3주 차 모델** — 3주 차 특징(종양 부피 변화 등)을 더한 로지스틱 회귀 11개 + CNN 1개

<br/>

<h2 id="data">03. 데이터셋</h2>

| 데이터 | 내용 | 규모 | 용도 |
|:---|:---|:---:|:---|
| **BreastDCEDL_ISPY2** | 치료 전 DCE-MRI (NIfTI), 종양 ROI 마스크, 임상 정보·pCR 라벨 | 982명 · 약 54 GB | 주 분석 |
| **TCIA I-SPY2** | 같은 환자의 치료 3주 차 DCE-MRI와 분석 마스크 (DICOM) | 942명 · 약 128 GB | 3주 차 추가 분석 |

- **환자 정보** — pCR, HR, HER2, 나이, 인종, 폐경
- **분할** — 학습 784명 / 내부 검증 198명. 학습 환자로만 5-fold × 20회 교차검증으로 모델을 정하고, 내부 검증 환자는 **마지막에 한 번만** 평가

> 영상 데이터는 용량과 이용 약관 때문에 저장소에 포함하지 않습니다. 각 출처에서 직접 받아야 합니다.

<br/>

<h2 id="results">04. 결과</h2>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/internal-auc-dark.png" />
  <img src="assets/internal-auc-light.png" width="860" alt="내부 검증 AUC: 임상 0.641, 치료 전 MRI 0.757, 치료 3주 차 MRI 0.816" />
</picture>

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/subtype-auc-dark.png" />
  <img src="assets/subtype-auc-light.png" width="860" alt="아형별 내부 검증 AUC: HR+/HER2- 0.48→0.87→0.88, TNBC 0.45→0.70→0.77, HER2+ 0.56→0.60→0.71" />
</picture>

- 치료 전 MRI만 더해도 AUC가 **0.641 → 0.757** (ΔAUC +0.116, DeLong p < 0.001)
- 치료 3주 차 MRI까지 더하면 **0.816** (95% CI 0.753–0.872)
- 영상의 효과는 **같은 아형 안에서도** 나타남 → 영상이 아형으로 설명되지 않는 개인차를 구별

<details>
<summary>교차검증 단계별 결과 (학습 784명, 5-fold × 20회)</summary>

<br/>

| 단계 | AUC | 임상 대비 ΔAUC |
|:---|:---:|:---:|
| 임상 정보 | 0.656 | — |
| + 종양 부피 | 0.689 | +0.033 |
| + 동역학 | 0.699 | +0.044 |
| + 영상 14관점 결합 | 0.715 | +0.060 |
| + 치료 3주 차 MRI (745명) | 0.770 | +0.117 (같은 745명의 임상 0.653 대비) |

- 내부 검증 결과가 교차검증보다 높게 나왔습니다(0.757 vs 0.715). 표본 구성의 차이일 수 있어 **두 값의 범위로 해석**합니다.
- 보정 곡선(예측 확률과 실제 pCR 비율의 일치)과 모델별 전체 결과는 [`Results/`](Results/)에 있습니다.

</details>

<br/>

<h2 id="code">05. 코드 구성</h2>

```
Code/
├── 00) Settings              경로 · 임상 변수
├── 01) Data                  데이터 통계 · 영상 압축 해제
├── 02) Preprocessing         spacing 복원 · 크롭 · 조영 증강률 · 리샘플링
├── 03) Feature Extraction    동역학 · 종양 주변 · 시간 곡선 · habitat · 질감 · CNN 입력
├── 04) Analysis Table        임상 + 영상 특징 + pCR 라벨
├── 05) Model Training        교차검증 · 로지스틱 회귀 14개
├── 06) CNN Training          ResNet-18 미세조정 (2.5D · 3D)
├── 07) Evaluation            교차검증 AUC · ΔAUC · 아형별 AUC
├── 08) Week-3 MRI Analysis   치료 3주 차 MRI 다운로드 · 특징 · 모델
├── 09) Final Validation      내부 검증 1회 평가
└── 10) Web                   의료진용 예측 웹앱 (Streamlit)
Results/                      모델별 결과 CSV · 그림
```

폴더·파일별 설명, 실행 순서, 결과 파일 안내는 **[GUIDE.md](GUIDE.md)** 에 정리했습니다.

<details>
<summary>기술 스택</summary>

<br/>

| 구분 | 사용 |
|:---|:---|
| 언어 | Python 3.11 |
| 영상 처리 | SimpleITK, pydicom, NumPy, SciPy |
| 특징 추출 | PyRadiomics |
| 모델링 | scikit-learn (로지스틱 회귀), PyTorch · torchvision (ResNet-18, 3D ResNet-18) |
| 데이터 · 시각화 | pandas, Matplotlib, Plotly |
| 웹앱 | Streamlit, scikit-image |
| 환경 | Windows 11, Conda, NVIDIA CUDA (GPU) |

</details>

<br/>

<h2 id="web">06. 의료진용 웹앱</h2>

최종 모델을 의료진이 바로 쓸 수 있도록 **Streamlit 웹앱**으로 구현했습니다. 코드는 [`Code/10) Web`](Code/10%29%20Web/)에 있습니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/web-main-dark.png" />
  <img src="assets/web-main-light.png" width="860" alt="BPR 웹앱 분석 화면: 왼쪽 환자 정보·진료 대기자·환자 목록, 가운데 종양 3D 재구성, 오른쪽 pCR 도달 확률과 판단 근거" />
</picture>

<table>
<tr>
<td align="center"><b>분석 결과</b><br/><sub>pCR 도달 확률 · 같은 아형 평균 비교</sub><br/><br/><picture><source media="(prefers-color-scheme: dark)" srcset="assets/web-result-dark.png" /><img src="assets/web-result-light.png" width="300" alt="" /></picture></td>
<td align="center"><b>정량 수치</b><br/><sub>종양 크기 · 시간-신호 곡선 · 종양 분석 결과</sub><br/><br/><picture><source media="(prefers-color-scheme: dark)" srcset="assets/web-numbers-dark.png" /><img src="assets/web-numbers-light.png" width="380" alt="" /></picture></td>
</tr>
</table>

<sub>※ 공개용 캡처: 환자 이름·등록번호는 가상으로 바꿨고, MRI 원본 영상 대신 종양 3D 재구성 화면을 실었습니다.</sub>

<details>
<summary>로그인 화면</summary>

<br/>

<img src="assets/web-login.png" width="860" alt="BPR 웹앱 의료진 로그인 화면" />

</details>

| 영역 | 기능 |
|:---|:---|
| **환자 관리** | 환자 목록(등록번호·이름 검색, 아형 필터), 새 환자 등록(영상 업로드), 진료 대기자 5명(끌어서 순서 변경, 진료 중·수술 중·대기 중 표시) |
| **영상 보기** | 치료 전 / 3주 차 MRI 선택, 2D 단면(조영 시점 막대로 시점별 영상) · 3D 종양 재구성 |
| **정량 수치** | 종양 부피 · 최대 직경 · 시간-신호 곡선과 곡선 유형 · 유출 영역 비율 · 평균 SER · 3주 차 부피 변화 |
| **예측 결과** | pCR 도달 확률 원형 게이지(같은 아형 평균과 비교), 판단 근거, 의사 소견 기록, 결과 저장(CSV) · Print |
| **보안** | 의료진 계정 로그인(기관 등록 코드로 가입), 6자리 간편 비밀번호, 30분 미사용 시 자동 로그아웃, 의료진별 환자 정보 분리 |

- 분석 속도: 치료 전 약 3초, 3주 차 약 2–5초 (분석 결과를 저장해 다시 씀)
- 화면 모드: 기본 · 다크 모드
- 환자 단위 자료(분석 결과, 업로드 영상)와 계정 정보는 저장소에 포함하지 않습니다.

### 결과 화면 읽는 법

> 화면은 **① 분석 결과 → ② 판단 근거 → ③ 정량 수치** 순서로 보면 됩니다.
> ①에서 *얼마나 가능성이 있는지*, ②에서 *왜 그렇게 나왔는지*, ③에서 *영상에서 실제로 무엇이 보이는지*를 확인합니다.

#### ① 분석 결과 — pCR 도달 확률

| 화면 표시 | 의미 | 읽는 법 |
|:---|:---|:---|
| **원형 게이지 숫자** | 항암치료 후 종양이 완전히 사라질(pCR) 확률. 로지스틱 회귀 앙상블과 CNN의 확률 평균 | 빨간 부분이 길수록 pCR 가능성이 높음 |
| **하얀 눈금** | 같은 아형 환자 982명의 실제 pCR 비율 (HER2+ 45.9% · TNBC 39.3% · HR+/HER2− 16.8%) | 빨간 부분이 눈금을 넘으면 같은 아형 평균보다 높음 |
| **평균보다 높음 · 낮음 · 비슷함** | 예측 확률과 같은 아형 평균의 차이 | 차이가 5%p 넘으면 높음/낮음, 5%p 이내면 비슷함 |
| **치료 전 예측 → 3주 차 반영** | 3주 차 MRI를 더했을 때 확률이 어떻게 바뀌었는지 | 올라가면 치료 초기 반응이 좋다는 신호 |
| **실제 결과** | I-SPY2 환자의 수술 후 병리 결과 (검증용) | 새로 등록한 환자는 수술 전이므로 표시되지 않음 |

**확률은 이렇게 받아들이면 됩니다** — 내부 검증 환자에서 예측 확률 구간별로 실제 pCR에 도달한 비율입니다.

| 예측 확률 | 치료 전 MRI (198명) | 치료 전 + 3주 차 MRI (197명) |
|:---:|:---:|:---:|
| 20% 미만 | **2%** (52명 중 1명) | **7%** (68명 중 5명) |
| 20 – 40% | 39% (85명 중 33명) | 27% (55명 중 15명) |
| 40 – 60% | 44% (46명 중 20명) | 57% (51명 중 29명) |
| 60% 이상 | **67%** (15명 중 10명) | **65%** (23명 중 15명) |
| 전체 | 32% | 32% |

- 확률이 높을수록 실제 pCR 비율도 높아집니다. 특히 **20% 미만이면 pCR이 매우 드뭅니다.**
- 3주 차 MRI를 더하면 40% 이상 구간의 실제 pCR 비율이 올라가 구분이 더 뚜렷해집니다 (AUC 0.757 → 0.816).

#### ② 판단 근거 — 왜 이런 확률이 나왔나

| 화면 표시 | 의미 |
|:---|:---|
| **항목 이름** | 예측에 쓰인 특징을 묶은 그룹 (호르몬 수용체, HER2, 나이, 종양 부피 변화, 종양 내부 구성, 조영 변화 등) |
| **높임 · 낮춤** | 평균적인 환자와 비교해 이 환자의 값이 pCR 확률을 올렸는지 내렸는지 |
| **막대 길이** | 영향의 크기. 이 환자에서 가장 큰 요인을 기준(100%)으로 한 상대 크기 |

- 로지스틱 회귀 모델의 *계수 × 표준화한 값*을 그룹별로 더한 값이라, 모델이 실제로 계산한 기여도를 그대로 보여 줍니다 (CNN 부분은 제외).
- 예측이 어떤 근거로 나왔는지 보여 주는 것이지, 원인과 결과를 뜻하지는 않습니다.

#### ③ 정량 수치 — 영상에서 계산한 값

| 수치 | 뜻 | 읽는 법 |
|:---|:---|:---|
| **종양 부피 · 최대 직경** | 종양 영역의 부피(cm³)와 가장 긴 길이(cm) | 치료 전후 크기 비교의 기준 |
| **부피 변화 (치료 전 대비)** | 3주 차 조영 증강 종양 부피(FTV)가 치료 전보다 줄어든 비율 | **크게 줄수록 pCR 가능성이 높은 경향** |
| **초기 증강률** | 조영 초기 영상에서 종양 신호가 조영 전보다 늘어난 비율 | 높을수록 혈류가 많은 종양 |
| **곡선 유형 · 시간-신호 곡선** | 조영제 주입 후 시점별 종양 평균 신호 변화 | **유출형**(올랐다 내려감, 악성에 흔함) · **평탄형** · **지속 상승형**(계속 오름) |
| **최고 증강률 (시점)** | 시간-신호 곡선에서 가장 높은 값과 그 시점 | 빨리 정점에 이를수록 조영제가 빨리 들어오는 종양 |
| **유출 영역 비율** | 종양 안에서 SER > 1.1인 화소 비율 | 조영제가 빨리 빠지는 부분이 얼마나 넓은지 |
| **평균 SER** | (초기 − 조영 전) ÷ (후기 − 조영 전)의 평균 | 1보다 크면 빨리 들어왔다 빠짐, 작으면 계속 차오름 |
| **강하게 조영되는 종양 비율** | 초기 신호가 70% 넘게 늘어난 화소 비율 (FTV 기준) | 높을수록 종양 전체가 고르게 강하게 조영됨 |
| **치료 전 종양 내부 구성** | 조영제가 들어오고 빠지는 속도로 나눈 3가지 영역(habitat)의 비율 | 같은 종양 안의 이질성 — 비율이 한쪽으로 치우쳤는지 확인 |

#### 예시 — 위 화면의 환자

| 단계 | 화면에서 본 것 | 해석 |
|:---:|:---|:---|
| ① | TNBC, 치료 전 **50% → 3주 차 76%**, 같은 아형 평균 39.3%보다 높음 | 치료를 시작한 뒤 pCR 가능성이 크게 올라감 |
| ② | 호르몬 수용체 음성 · 치료 전 대비 조영 변화 · 나이가 **높임** | 아형과 3주 차 영상 변화가 확률을 끌어올림 |
| ③ | 조영 증강 부피 **21.8 → 10.8 cm³ (−50%)** | 3주 만에 종양 부피가 절반으로 줄어 초기 반응이 좋음 |
| 실제 | 수술 결과 **pCR** | 예측과 일치 |

> **참고** — 이 확률은 I-SPY2 내부 검증(198명)으로 성능을 확인한 연구용 예측입니다. 진단을 대신하지 않으며, 다른 기관 데이터에서의 검증은 아직 진행 전입니다.

<br/>

<h2 id="method">07. 개발 방법론 · 일정</h2>

### 방법론 — 탐색적 프로토타이핑 (PoC 선행) + CRISP-DM

치료 전 MRI 특징으로 pCR 예측이 가능한지는 직접 만들어 보기 전에는 알 수 없었습니다.
그래서 **문제 정의 후 예측 가능성을 먼저 검증(PoC)** 하고, 그 결과로 확인된 유효 특징·모델 구조·실제 발생 위험을 근거로 분석·설계 문서를 작성했습니다.
데이터 분석은 **CRISP-DM** 절차(데이터 준비 → 모델링 → 평가)를 반복하며 진행했습니다.

### 일정

| 기간 | 내용 |
|:---|:---|
| 2026.08.27 ~ 2026.12.11 | 전체 프로젝트 (설계 및 프로젝트 기본) |
| 2026.10.02 ~ 2026.10.08 | 구현 — 전처리 → 특징 추출 → 모델 학습 → 3주 차 추가 분석 → 내부 검증 |
| 2026.10 ~ | 분석·설계 문서 작성, 포스터·발표 (KY-FESTA) |
| 2026.11 | 의료진용 웹앱 구현 — 영상 보기 · 예측 결과 · 로그인 · 진료 대기자 |

<br/>

<h2 id="scope">08. 구현 범위 · 향후 계획</h2>

### 구현 범위와 보완점

- 본 프로젝트는 비침습 MRI 영상이 기존 임상 정보에 더해 pCR 예측에 기여하는 정도를 **정량적으로 확인하는 탐색적 연구**입니다.
- 기준 모델은 데이터에서 확보할 수 있는 임상 정보(HR·HER2·나이·인종·폐경)로 구성했으며, 종양 크기는 MRI에서 계산한 종양 부피로 반영했습니다. 종양 등급·Ki-67 등 임상 정보를 더하면 비교 기준을 한층 보완할 수 있습니다.
- 성능은 **I-SPY2 내부 검증(198명)** 으로 확인했으며, 다른 기관 데이터를 통한 외부 검증으로 범용성을 넓혀 갈 예정입니다.
- 의료진용 웹앱(영상 업로드·2D/3D 보기·확률과 근거 표시)을 구현했으며, 실제 진료 환경에서의 사용성 평가는 앞으로 진행할 과제입니다.

### 향후 계획

| | 계획 | 내용 |
|:---:|:---|:---|
| 1 | **외부 데이터셋 검증** | 다른 기관 데이터로 재현성을 확인해 범용 예측 모델 구축 |
| 2 | **치료 중 MRI 추적** | 미사용 치료 중기(T2)·수술 전(T3) MRI까지 활용해 조기 반응 예측 고도화 |
| 3 | **웹앱 임상 적용** | 의료진 사용성 평가, 병원 영상 시스템(PACS)과 연동해 영상 업로드 과정 단순화 |

<br/>

<h2 id="team">Team CANSLER</h2>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/team-cards-dark.png" />
  <img src="assets/team-cards-light.png" width="860" alt="정가영 GaYeong Jeong, PM · PL · CM / 김예슬 YeSeul Kim, ENG · QA" />
</picture>

<p align="center"><b>정가영</b> GaYeong Jeong &nbsp;·&nbsp; <b>김예슬</b> YeSeul Kim &nbsp;|&nbsp; 지도교수 <b>한주혁</b></p>

<br/>

## 참고 문헌

- Antonini, M., et al. (2023). Real-world evidence of neoadjuvant chemotherapy for breast cancer treatment in a Brazilian multicenter cohort. *The Breast, 72*, 103577. https://doi.org/10.1016/j.breast.2023.103577
- Fang, S., et al. (2024). A real-world clinicopathological model for predicting pathological complete response to neoadjuvant chemotherapy in breast cancer. *Frontiers in Oncology, 14*, 1323226. https://doi.org/10.3389/fonc.2024.1323226
- Xu, M., et al. (2025). Expression and subtype discordance between core needle biopsy and surgical specimen in breast cancer. *Journal of Surgical Research, 307*, 42–52. https://doi.org/10.1016/j.jss.2025.01.010
- Shi, Z., et al. (2023). MRI-based quantification of intratumoral heterogeneity for predicting treatment response to neoadjuvant chemotherapy in breast cancer. *Radiology, 308*(1), e222830. https://doi.org/10.1148/radiol.222830
- Clinical Lab Products. (2025). *AI test predicts neoadjuvant response in early breast cancer.* https://clpmag.com/disease-states/cancer/breast/ai-test-predicts-neoadjuvant-response-early-breast-cancer/
- Agendia. (2024). *New neoadjuvant trial (MINT) confirms the predictive utility of MammaPrint & BluePrint.* https://agendia.com/neoadjuvant-mint-trial-confirms-predictive-utility-of-mammaprint-blueprint/

## 데이터 출처

- **BreastDCEDL** — I-SPY2 치료 전 DCE-MRI · 종양 마스크 · 임상 정보 (Zenodo 배포, CC BY-NC 4.0)
- **The Cancer Imaging Archive (TCIA), I-SPY2 / ACRIN 6698** — 치료 3주 차 DCE-MRI · 분석 마스크 (CC BY 4.0). https://www.cancerimagingarchive.net/collection/ispy2/

<br/>

<div align="center">
<sub>2026 KY-FESTA · 건양대학교 의료IT공학과 · Team CANSLER</sub>
</div>
