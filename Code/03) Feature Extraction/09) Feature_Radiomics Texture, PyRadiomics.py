"""
04_extract_radiomics.py ─ PyRadiomics 특징 추출 (IBSI)
=====================================================
전처리된 percent enhancement(PE) 영상 + ROI 마스크에서 형태·1차통계·질감 특징을 추출한다.

입력 (03): Data/Interim/<pid>/pe.nii.gz, mask.nii.gz
출력:      Data/Features/radiomics_features.csv  (환자 × 특징)
QC:        Evaluate/04_radiomics_qc.csv  (환자별 성공/실패, ROI 범위, 실제 bin 수)

실행:  python 04_extract_radiomics.py --limit 5 / (전체)
       # 동역학 지도 질감 (03이 저장한 추가 지도, shape는 pe와 중복이라 제외)
       python 04_extract_radiomics.py --image pe_early --prefix dceE --no-shape --out radiomics_pe_early.csv
       python 04_extract_radiomics.py --image ser --prefix dceS --binwidth 0.05 --no-shape --out radiomics_ser.csv

재현성(문서 §14): binWidth·리샘플 고정, normalize=False (PE가 이미 % 단위로 보정됨).
  binWidth는 PE(%) 단위 cfg.BIN_WIDTH. 구버전의 binWidth=25 + z-score 조합은 bin이 1~2개뿐이라
  질감 특징이 상수화됐음 → QC의 n_bins 분포로 반드시 확인 (권장 16~128).
params.yaml 이 Code1 에 있으면 그것을 우선 사용.
"""
from __future__ import annotations
import argparse
import logging
import sys
from importlib import import_module
import pandas as pd

cfg = import_module("00_config")

DEFAULT_SETTINGS = {
    "binWidth": cfg.BIN_WIDTH,
    "resampledPixelSpacing": None,   # 03에서 이미 등방 리샘플
    "normalize": False,               # PE(%) 자체가 정규화된 단위
    "label": 1,
}
ENABLED = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]


def _extractor(bin_width=None, no_shape=False):
    try:
        from radiomics import featureextractor  # type: ignore
    except ImportError:
        print("[!] pyradiomics 필요: pip install pyradiomics (또는 conda-forge)"); raise
    yaml = cfg.CODE_DIR / "params.yaml"
    if yaml.exists() and bin_width is None and not no_shape:
        ext = featureextractor.RadiomicsFeatureExtractor(str(yaml))
    else:
        settings = dict(DEFAULT_SETTINGS)
        if bin_width is not None:
            settings["binWidth"] = bin_width
        ext = featureextractor.RadiomicsFeatureExtractor(**settings)
        ext.disableAllFeatures()
        for c in ENABLED:
            if not (no_shape and c == "shape"):   # 추가 지도는 같은 마스크 → shape 중복
                ext.enableFeatureClassByName(c)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return ext


def _roi_range(img_path, mask_path, label=1):
    """ROI 내 PE 최소·최대 → 실제 bin 수 추정 (binWidth 적정성 QC)."""
    import SimpleITK as sitk  # noqa: N813  (pyradiomics 의존성)
    f = sitk.LabelStatisticsImageFilter()
    f.Execute(sitk.ReadImage(str(img_path)), sitk.ReadImage(str(mask_path), sitk.sitkUInt8))
    return f.GetMinimum(label), f.GetMaximum(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="radiomics_features.csv")
    ap.add_argument("--image", default="pe", help="Data/Interim/<pid>/<image>.nii.gz (pe / pe_early / ser)")
    ap.add_argument("--prefix", default=cfg.IMG_PREFIX, help="특징 컬럼 접두어 (예: dceE, dceS)")
    ap.add_argument("--binwidth", type=float, default=None, help="기본 cfg.BIN_WIDTH (PE % 단위)")
    ap.add_argument("--no-shape", action="store_true", help="shape 제외 (같은 마스크의 추가 지도용)")
    ap.add_argument("--mask", default="mask", help="Data/Interim/<pid>/<mask>.nii.gz (예: shells)")
    ap.add_argument("--label", type=int, default=1, help="마스크 안의 라벨 번호 (shells: 1–4)")
    ap.add_argument("--root", default=None, help="환자 폴더 상위 경로 (기본 Data/Interim, T1은 Data/Interim/T1)")
    args = ap.parse_args()
    cfg.ensure_dirs()
    img_name, mask_name = f"{args.image}.nii.gz", f"{args.mask}.nii.gz"
    root = cfg.DATA_INTERIM / args.root if args.root else cfg.DATA_INTERIM

    pts = [d.name for d in sorted(root.iterdir())
           if (d / img_name).exists() and (d / mask_name).exists()] \
          if root.exists() else []
    if not pts:
        print(f"[!] 전처리 결과 없음: {cfg.DATA_INTERIM}/*/{img_name}. 먼저 03 실행."); return 1
    if args.limit:
        pts = pts[: args.limit]

    ext = _extractor(args.binwidth, args.no_shape)
    ext.settings["label"] = args.label
    bw = ext.settings.get("binWidth", cfg.BIN_WIDTH)
    print(f"[i] 특징 추출 {len(pts)}명 · 지도 {args.image} · 접두어 {args.prefix}_ · binWidth {bw}")
    recs, qc = [], []
    for i, pid in enumerate(pts, 1):
        d = root / pid
        q = {"pid": pid}
        try:
            lo, hi = _roi_range(d / img_name, d / mask_name, args.label)
            q.update(roi_min=lo, roi_max=hi, n_bins=int((hi - lo) // bw) + 1)
            res = ext.execute(str(d / img_name), str(d / mask_name), label=args.label)
            row = {cfg.COL_PID: pid}
            for k, v in res.items():
                if not k.startswith("diagnostics_"):
                    row[f"{args.prefix}_{k}"] = float(v)
            recs.append(row); q["status"] = "OK"
            if i % 25 == 0 or i <= 5:
                print(f"  [{i}/{len(pts)}] {pid} ✓ ({len(row)-1}특징 · bin {q['n_bins']})")
        except Exception as e:  # noqa: BLE001
            q["status"] = f"실패:{e}"
            print(f"  [{i}/{len(pts)}] {pid} 실패: {e}")
        qc.append(q)

    qdf = pd.DataFrame(qc)
    qname = ("04_radiomics_qc.csv" if args.image == "pe" and args.mask == "mask"
             else f"04_radiomics_qc_{args.prefix}.csv")
    qdf.to_csv(cfg.EVAL_DIR / qname, index=False)
    if "n_bins" in qdf:
        nb = qdf["n_bins"].dropna()
        print(f"\n[QC] bin 수 중앙값 {nb.median():.0f} (5~95% {nb.quantile(.05):.0f}~{nb.quantile(.95):.0f})"
              f" · 16 미만 {int((nb < 16).sum())}명 · 128 초과 {int((nb > 128).sum())}명")
    if not recs:
        print("[!] 추출 0건"); return 2
    df = pd.DataFrame(recs)
    out = cfg.DATA_FEATURES / args.out
    df.to_csv(out, index=False)
    print(f"\n[✓] {out}  ({df.shape[0]}명 × {df.shape[1]-1}특징) · QC: Evaluate/{qname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
