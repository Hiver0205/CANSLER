# 🩺 BPR — Breast pCR Response

> **치료 전 유방 MRI의 radiomics로 신보조 항암 완전관해(pCR)를 비침습적으로 예측하고,
> 영상이 기존 임상 지표를 넘어 기여하는지 검증하는 탐색적 프로젝트**
>
> *A non-invasive, MRI-radiomics approach to predict pathologic complete response (pCR)
> to neoadjuvant chemotherapy in breast cancer, and to test whether imaging adds value
> beyond clinical variables.*

---

## 📌 프로젝트 개요

"BPR(Breast pCR Response)"은 이미 촬영된 **치료 전 유방 MRI**만으로, 추가 침습 검사 없이
신보조 항암(수술 전 항암)의 "완전관해(pCR)" 여부를 예측하는 개념 검증(proof-of-concept) 프로젝트입니다.

- **pCR이란?** 신보조 항암 후 수술로 떼낸 조직에서 침습 암세포가 완전히 사라진 상태.
  pCR에 도달한 환자는 그렇지 않은 환자보다 5년 생존·무병생존이 뚜렷이 높습니다(예: 5년 OS 88% vs 58%).

---

## 🎯 문제 배경

| # | 문제 | 핵심 근거 |
|---|------|-----------|
| **①** | **치료 반응의 불확실성** — pCR 도달은 소수(약 22%)에 그치고, pCR은 수술 후 병리로만 확정되어 치료 전엔 알 수 없음 | Antonini et al. (2023); Fang et al. (2024) |
| **②** | **침습적 검사 의존** — 정밀 예측 지표(Ki-67·유전자 등)는 생검 기반인데, 생검은 종양 일부만 떼어 전체를 대표 못 함 (생검–수술조직 아형 불일치 **16.4%**, 저Ki-67 종양 **32.0%**) | Xu et al. (2025) |
| **③** | **기존 예측 도구의 한계** — 상용 도구(Ataraxis=병리영상, MammaPrint=유전자)는 모두 조직 기반·침습이며 비용·접근성 제약, **비침습 MRI 기반 도구는 부재** | Clinical Lab Products (2025); Agendia (2024) |

---

## 💡 접근 방법

치료 전 MRI에서 **radiomics 특징**(신호 강도·모양·질감·조영 증강 패턴)을 종양 전 영역에서 추출하여
개별 환자의 **pCR 확률**을 산출합니다.

```
MRI (치료 전 DCE) + 종양 ROI  →  radiomics 특징 추출  →  pCR 예측 모델  →  환자별 pCR 확률
```

### 🔬 핵심 연구 질문

> **“비침습 MRI 영상이, 아형 등 기존 임상 지표에 *더해* pCR 예측에 추가로 기여하는가?”**

- **비교:** 임상 변수(HR·HER2·나이 등) 모델  **vs**  임상 변수 + 영상 모델
- **평가:** 전체 성능 + **아형별(TNBC·HER2+·Luminal) 성능** 을 나누어 확인
- 영상을 더해 예측이 개선되면 → 영상이 기존 지표로는 담을 수 없는 **개인별 정보**를 제공한다는 근거

> 📍 본 프로젝트의 **구현 범위는 "임상 모델 vs 임상+영상" 비교**이며,
> 아형 단독 효과는 별도 모델 대신 **아형별 성능 평가**로 확인합니다.

---

## 🗂 데이터셋

**BreastDCEDL_ISPY2**
- 치료 전 **DCE-MRI** 영상 (NIfTI)
- 종양 **ROI 마스크**
- 환자 라벨: **pCR / HR / HER2 / 나이** 등

> ⚠️ 데이터·모델 파일은 용량과 라이선스 문제로 저장소에 포함하지 않습니다(`.gitignore` 처리).

---

## 🛠 기술 스택

| 구분 | 사용 |
|------|------|
| 언어 | Python 3.12 |
| 영상 처리 | SimpleITK, nibabel |
| 특징 추출 | PyRadiomics |
| 모델링 | scikit-learn, LightGBM |
| 환경 | Windows 11 |

---

## ⚠️ 구현 범위 및 목표

- 본 프로젝트는 **탐색적 연구**입니다.
- 목표는 "비침습 영상이 기존 지표에 기여하는지의 정량화"입니다.
- **분자·유전자 아형 분석은 수행하지 않습니다** — 영상이라는 별도 경로를 사용합니다.
- 데이터에 종양 크기·등급·Ki-67 등이 없어, baseline은 "데이터에 있는 임상 변수 기준"임을 명시합니다.

---

## 📅 일정

**2026.08.27. ~ 2026.12.11. (총 84일)** — 폭포수(Waterfall) 방법론
- 중간고사 이전: 제안 · 분석 · 설계 일부
- 중간고사 이후: 남은 설계 · 구현 · 시험 · 완료

---

## 📚 참고 문헌

- Antonini, M., et al. (2023). Real-world evidence of neoadjuvant chemotherapy for breast cancer treatment in a Brazilian multicenter cohort. *The Breast, 72*, 103577. https://doi.org/10.1016/j.breast.2023.103577
- Fang, S., et al. (2024). A real-world clinicopathological model for predicting pathological complete response to neoadjuvant chemotherapy in breast cancer. *Frontiers in Oncology, 14*, 1323226. https://doi.org/10.3389/fonc.2024.1323226
- Xu, M., et al. (2025). Expression and subtype discordance between core needle biopsy and surgical specimen in breast cancer. *Journal of Surgical Research, 307*, 42–52. https://doi.org/10.1016/j.jss.2025.01.010
- Shi, Z., et al. (2023). MRI-based quantification of intratumoral heterogeneity for predicting treatment response to neoadjuvant chemotherapy in breast cancer. *Radiology, 308*(1), e222830. https://doi.org/10.1148/radiol.222830
- Clinical Lab Products. (2025). *AI test predicts neoadjuvant response in early breast cancer.* https://clpmag.com/disease-states/cancer/breast/ai-test-predicts-neoadjuvant-response-early-breast-cancer/
- Agendia. (2024). *New neoadjuvant trial (MINT) confirms the predictive utility of MammaPrint & BluePrint.* https://agendia.com/neoadjuvant-mint-trial-confirms-predictive-utility-of-mammaprint-blueprint/
