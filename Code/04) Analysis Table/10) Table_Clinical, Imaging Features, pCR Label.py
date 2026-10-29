"""
05_build_features.py ─ 분석 테이블 구성 (임상 + 영상 + 정답)
=========================================================
metadata의 임상 baseline(HR·HER2·나이·인종·폐경)과 radiomics 특징을 pid로 병합,
06/07이 읽을 단일 테이블을 만든다.

입력:  Data/Features/radiomics_features.csv (04), metadata.csv
출력:  Data/Features/analysis_table.csv

컬럼 규약:
  pid, pCR, subtype, split, clin__<임상>, vol__log_tum_vol, dce_<radiomics...>
QC:    Evaluate/05_cohort_flow.csv  (환자별 영상 유무 — 라벨 연관 이탈 점검)

실행:  python 05_build_features.py
       python 05_build_features.py --no-image   # 영상 전이라 임상만으로 구조 점검
"""
from __future__ import annotations
import argparse
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="radiomics_features.csv")
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--out", default="analysis_table.csv")
    args = ap.parse_args()
    cfg.ensure_dirs()

    if not cfg.METADATA_CSV.exists():
        print(f"[!] metadata 없음: {cfg.METADATA_CSV}"); return 1
    meta = pd.read_csv(cfg.METADATA_CSV)
    meta[cfg.COL_PID] = meta[cfg.COL_PID].astype(str)

    base = pd.DataFrame({
        cfg.COL_PID: meta[cfg.COL_PID],
        "pCR": pd.to_numeric(meta[cfg.COL_PCR], errors="coerce"),
        "subtype": meta[cfg.COL_SUBTYPE].astype(str),
        "split": meta[cfg.COL_SPLIT] if cfg.COL_SPLIT in meta.columns else 0,
    })
    for c in cfg.CLINICAL_COLS:
        if c in meta.columns:
            base[f"clin__{c}"] = pd.to_numeric(meta[c], errors="coerce")
        else:
            print(f"  [!] 임상변수 없음(건너뜀): {c}")
    # TCIA 임상(치료 전 정보): MammaPrint(mp__) + 배정 약제 원핫(arm__). 행 단위 → 누수 없음.
    if cfg.TCIA_CLINICAL_XLSX.exists():
        tc = pd.read_excel(cfg.TCIA_CLINICAL_XLSX)
        tc["num"] = tc["Patient_ID"].astype(int)
        num = base[cfg.COL_PID].str.split("-").str[-1].astype(int)
        tc = tc.set_index("num")
        chk = tc.reindex(num.values)
        ok = (chk["pCR"].to_numpy() == base["pCR"].to_numpy())
        print(f"[i] TCIA 임상 매칭 {int(chk['Arm'].notna().sum())}/{len(base)} · pCR 일치 {ok.mean():.1%}")
        base["mp__MP"] = chk["MP"].to_numpy().astype(float)
        arm = chk["Arm"].fillna("unknown").to_numpy()
        for a in sorted(pd.unique(arm)):
            slug = "".join(ch if ch.isalnum() else "_" for ch in a).strip("_")
            base[f"arm__{slug}"] = (arm == a).astype(float)
        # MP × 아형 (MP 효과가 아형마다 다름)
        base["mpx__MPxHR"] = base["mp__MP"] * base["clin__HR"]
        base["mpx__MPxHER2"] = base["mp__MP"] * base["clin__HER2"]
        # 약제 계열 (13종 원핫은 희소 → 작용기전별로 묶음; 한 환자가 여러 계열에 속할 수 있음)
        arm_s = pd.Series(arm)
        has = lambda s: arm_s.str.contains(s, regex=False).to_numpy().astype(float)  # noqa: E731
        base["drug__antiHER2"]   = np.clip(has("Trastuzumab") + has("Pertuzumab") + has("T-DM1")
                                           + has("Neratinib"), 0, 1)
        base["drug__dualHER2"]   = has("Pertuzumab")
        base["drug__pembro"]     = has("Pembrolizumab")
        base["drug__parp_plat"]  = has("Carboplatin")
        base["drug__akt"]        = has("MK-2206")
        base["drug__other_new"]  = np.clip(has("AMG 386") + has("Ganitumab") + has("Ganetespib"), 0, 1)
        base["drug__control"]    = (arm_s == "Paclitaxel").to_numpy().astype(float)
    else:
        print(f"[i] TCIA 임상 파일 없음 → MP·Arm 생략 ({cfg.TCIA_CLINICAL_XLSX.name})")

    # 촬영 조건 그룹 (G3 하모나이제이션용, 예측 변수 아님): 기관(ACRIN/ISPY2) × slice 두께(≤1.5 / >1.5mm)
    site = base[cfg.COL_PID].str.startswith("ACRIN").astype(int)
    thick = (pd.to_numeric(meta[cfg.COL_SLICE_THICK], errors="coerce") > 1.5).astype(int).to_numpy()
    base["batch__id"] = (site * 2 + thick).astype(float)

    # 임상 파생: HR×HER2 (HER2+ 안에서 HR−/HR+ 구분 — HR−HER2+의 pCR률이 특히 높음). 행 단위 → 누수 없음.
    if {"clin__HR", "clin__HER2"} <= set(base.columns):
        base["clinx__HRxHER2"] = base["clin__HR"] * base["clin__HER2"]
    # 종양크기 baseline: 치료 전 종양 부피(cm³, log). 마스크에서 나온 값이지만 임상에서도
    # T병기로 이미 아는 정보 → "영상 증분"이 단순 크기 효과인지 가리기 위한 별도 그룹(vol__).
    if cfg.COL_TUM_VOL in meta.columns:
        base["vol__log_tum_vol"] = np.log1p(pd.to_numeric(meta[cfg.COL_TUM_VOL], errors="coerce") / 1000.0)

    # 영상 병합
    fpath = cfg.DATA_FEATURES / args.features
    if args.no_image or not fpath.exists():
        if not args.no_image:
            print(f"[i] 영상 특징 없음({fpath.name}) → 임상만 테이블 (구조 점검용)")
        table = base.copy()
    else:
        feats = pd.read_csv(fpath)
        feats[cfg.COL_PID] = feats[cfg.COL_PID].astype(str)
        print(f"[i] 영상 특징 {feats.shape[0]}명 × {feats.shape[1]-1}")
        table = pd.merge(base, feats, on=cfg.COL_PID, how="inner")

        # QC: 영상 없는 환자 = 코호트 이탈. tarball이 pcr/nopcr로 나뉘어 있어 이탈이
        # 라벨과 연관될 수 있음 → 이탈군 vs 잔류군 pCR률을 비교·기록.
        flow = base[[cfg.COL_PID, "pCR", "subtype", "split"]].copy()
        flow["has_image"] = flow[cfg.COL_PID].isin(set(feats[cfg.COL_PID]))
        flow.to_csv(cfg.EVAL_DIR / "05_cohort_flow.csv", index=False)
        drop = flow[~flow.has_image]
        if len(drop):
            print(f"  [!] 영상 없음 → 제외 {len(drop)}명 · 제외군 pCR {drop['pCR'].mean():.1%}"
                  f" vs 잔류군 {flow[flow.has_image]['pCR'].mean():.1%}  (Evaluate/05_cohort_flow.csv)")
            print("      split별 제외: " + drop.groupby("split").size().to_string().replace("\n", ", "))

    # 동역학 특징(03b) — 있으면 left join (일부 실패 시 결측 → 06의 폴드 내 중앙값 대치)
    kpath = cfg.DATA_FEATURES / "kinetic_features.csv"
    if not args.no_image and kpath.exists():
        kin = pd.read_csv(kpath); kin[cfg.COL_PID] = kin[cfg.COL_PID].astype(str)
        table = pd.merge(table, kin, on=cfg.COL_PID, how="left")
        n_miss = int(table[[c for c in kin.columns if c != cfg.COL_PID]].isna().all(axis=1).sum())
        print(f"[i] 동역학 특징 {kin.shape[1]-1}개 병합 (전부 결측 {n_miss}명)")

        # 아형 × 동역학 상호작용: 아형별 기울기를 따로 학습할 수 있게 아형 지시변수를 곱한 복사본.
        #   행 단위 결정론적 변환(학습 없음) → 누수 없음. 결측은 그대로 두어 06에서 폴드 내 대치.
        hr, her2 = table["clin__HR"], table["clin__HER2"]
        ind = {"HER2pos": her2 == 1, "TN": (hr == 0) & (her2 == 0), "HRpos": (hr == 1) & (her2 == 0)}
        kcols = [c for c in kin.columns if c != cfg.COL_PID]
        inter = {f"kinx_{st}__{c[4:]}": table[c].where(msk, 0.0)
                 for st, msk in ind.items() for c in kcols}
        table = pd.concat([table, pd.DataFrame(inter, index=table.index)], axis=1)
        print(f"[i] 아형×동역학 상호작용 {len(inter)}개 생성")

    # 사전학습 CNN 특징(04b) — 백본별 접두어로 구분해 left join (qc_* 열은 제외: 크기 대리변수)
    for bb, pfx in (("resnet50", "deepR_"), ("dinov2_vits14", "deepD_")):
        dp = cfg.DATA_FEATURES / f"deep_features_{bb}.csv"
        if not args.no_image and dp.exists():
            dfp = pd.read_csv(dp); dfp[cfg.COL_PID] = dfp[cfg.COL_PID].astype(str)
            keep = [c for c in dfp.columns if c.startswith("deep_")]
            dfp = dfp[[cfg.COL_PID] + keep].rename(columns={c: pfx + c[5:] for c in keep})
            table = pd.merge(table, dfp, on=cfg.COL_PID, how="left")
            print(f"[i] CNN 특징 {bb} {len(keep)}개 병합 ({dfp.shape[0]}명) → {pfx}*")

    # 종양 주변·배경 실질(03c, peri_*) + 고리별 질감(04 --mask shells, pT1_~pT3_) — 있으면 left join
    for fname in ("peri_features.csv", "radiomics_shell1.csv", "radiomics_shell2.csv", "radiomics_shell3.csv",
                  "morph_features.csv", "timecurve_features.csv", "radiomics_washin.csv",
                  "radiomics_washout.csv", "radiomics_auc.csv", "radiomics_early_frac.csv",
                  "habitat_features.csv", "radiomics_hab1.csv", "radiomics_hab2.csv", "radiomics_hab3.csv",
                  "curvebag_features.csv"):
        xp = cfg.DATA_FEATURES / fname
        if not args.no_image and xp.exists():
            xf = pd.read_csv(xp); xf[cfg.COL_PID] = xf[cfg.COL_PID].astype(str)
            table = pd.merge(table, xf, on=cfg.COL_PID, how="left")
            print(f"[i] {fname} {xf.shape[1]-1}개 병합 ({xf.shape[0]}명)")

    # 동역학 지도 질감(04 --image pe_early / ser) — 있으면 left join
    for fname in ("radiomics_pe_early.csv", "radiomics_ser.csv"):
        xp = cfg.DATA_FEATURES / fname
        if not args.no_image and xp.exists():
            xf = pd.read_csv(xp); xf[cfg.COL_PID] = xf[cfg.COL_PID].astype(str)
            table = pd.merge(table, xf, on=cfg.COL_PID, how="left")
            print(f"[i] {fname} {xf.shape[1]-1}개 병합 ({xf.shape[0]}명)")

    # K2: 부피 × 핵심 곡선 특징 상호작용 (행 단위 곱 → 누수 없음). "큰 종양 + 빠른 유출" 같은 조합 효과.
    if "vol__log_tum_vol" in table:
        for c in ("tc_washin_mean", "tc_washout_mean", "tc_auc_mean", "tc_early_frac_mean",
                  "tc_frac_washout", "tc_frac_persistent"):
            if c in table:
                table[f"vtc_{c[3:]}"] = table["vol__log_tum_vol"] * table[c]

    if table[cfg.COL_PID].duplicated().any():
        print("[!] pid 중복 행 존재 — 환자 단위 분할이 깨짐. 중단."); return 3

    before = len(table)
    table = table.dropna(subset=["pCR"]); table["pCR"] = table["pCR"].astype(int)
    out = cfg.DATA_FEATURES / args.out
    table.to_csv(out, index=False)

    n_clin = sum(c.startswith("clin__") for c in table.columns)
    n_vol  = sum(c.startswith("vol__") for c in table.columns)
    n_img  = sum(c.startswith(f"{cfg.IMG_PREFIX}_") for c in table.columns)
    n_kin  = sum(c.startswith("kin_") for c in table.columns)
    print(f"\n[✓] {out}")
    print(f"    {len(table)}명 (pCR결측 제거 {before-len(table)}) · 임상 {n_clin} · 크기 {n_vol} · radiomics {n_img}"
          f" · 동역학 {n_kin} · pCR {table['pCR'].mean():.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
