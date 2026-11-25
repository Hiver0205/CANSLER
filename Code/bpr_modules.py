"""
bpr_modules.py ─ 코드 파일 이름 연결
==================================================================
저장소의 코드는 읽기 쉽게 "03) Preprocessing_Spacing Restore, ....py"처럼 단계 이름을 붙여 두었지만,
코드 안에서는 원래 모듈 이름(import_module("03_preprocess") 등)으로 서로 불러온다.
이 파일을 먼저 import 하면 원래 이름 → 저장소 파일을 찾아 주므로 파일 이름을 바꾸지 않고 그대로 실행된다.

  import sys; sys.path.insert(0, "Code"); import bpr_modules   # 이후 import_module("21_infer") 등 그대로 동작
"""
from __future__ import annotations
import importlib.abc
import importlib.util
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent

# 원래 모듈 이름 → 저장소 안 파일 (Code/ 기준)
MODULES = {
    "00_config": "00) Settings/00) Settings_Path, Clinical Variables.py",
    "01_explore_metadata": "01) Data/01) Data_Metadata Summary, Subtype pCR Rate.py",
    "02_extract_images": "01) Data/02) Data_Image Extraction, NIfTI Unpack.py",
    "03_preprocess": "02) Preprocessing/03) Preprocessing_Spacing Restore, Crop, PE-SER, Resampling.py",
    "03b_kinetic_features": "03) Feature Extraction/04) Feature_Kinetic, FTV, SER.py",
    "03c_peritumoral": "03) Feature Extraction/05) Feature_Peritumoral, BPE.py",
    "03e_timecurve": "03) Feature Extraction/06) Feature_Time Curve Maps, Wash-in, Wash-out.py",
    "03f_habitat": "03) Feature Extraction/07) Feature_Habitat, K-means.py",
    "03g_timestack": "03) Feature Extraction/08) Feature_Time Stack, CNN Input 6ch.py",
    "04_extract_radiomics": "03) Feature Extraction/09) Feature_Radiomics Texture, PyRadiomics.py",
    "05_build_features": "04) Analysis Table/10) Table_Clinical, Imaging Features, pCR Label.py",
    "06_train_cv": "05) Model Training/11) Model_Nested CV, Logistic Regression, 14 Views.py",
    "04c_cnn_finetune": "06) CNN Training/12) CNN_ResNet-18 Fine-tuning, 2.5D, 3D.py",
    "07_evaluate": "07) Evaluation/13) Evaluation_CV AUC, Delta AUC, Subtype AUC.py",
    "09_t1_download": "08) Week-3 MRI Analysis/14) Week3_TCIA T1 DICOM Download.py",
    "10_t1_features": "08) Week-3 MRI Analysis/15) Week3_T1 Features, FTV Change.py",
    "10b_t1_maps": "08) Week-3 MRI Analysis/16) Week3_T1 Maps, Radiomics-CNN Input.py",
    "08_exp3_early_response": "08) Week-3 MRI Analysis/17) Week3_Early Response Model, CV.py",
    "08b_exp3_cnn_ensemble": "08) Week-3 MRI Analysis/18) Week3_CNN Ensemble.py",
    "11_final_holdout": "09) Final Validation/19) Validation_Internal Validation, Final Model.py",
    "20_export_models": "09) Final Validation/20) Validation_Export Final Models.py",
    "21_infer": "10) Web/21) Web_Inference, Pre and Week-3.py",
    "bpr_core": "10) Web/22) Web_Core, Patients, Queue, Image Stats.py",
    "auth": "10) Web/23) Web_Accounts, Login, PIN.py",
    "23_dicom_io": "10) Web/25) Web_DICOM Upload, I-SPY2 Format Check.py",
}
WEB_APP = CODE / "10) Web" / "20) Web_App, Screen Layout.py"
PRECOMPUTE = CODE / "10) Web" / "24) Web_Batch Precompute.py"


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        rel = MODULES.get(name)
        if rel is None:
            return None
        return importlib.util.spec_from_file_location(name, CODE / rel)


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
