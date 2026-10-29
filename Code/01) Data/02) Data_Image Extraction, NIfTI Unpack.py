"""
02_extract_images.py ─ DCE 영상 tarball 압축 해제 + 구조 확인
===========================================================
pcr(6개) + nopcr(8개) tarball과 mask tarball을 Data/Interim/extracted 로 푼다.
전체는 ~54GB라 시간이 걸리므로, 먼저 --proto 로 소량만 풀어 명명 규칙을 확인하는 걸 권장.

실행:
    python 02_extract_images.py --proto          # pcr_part1 + nopcr_part1 + mask 만
    python 02_extract_images.py                  # 전체
    python 02_extract_images.py --mask-only      # mask만

해제 후 실제 NIfTI 파일명과 vis1 아닌 파일 수를 출력하고,
tarball별 해제 결과를 Evaluate/02_extract_log.csv 에 남긴다 (QC).
"""
from __future__ import annotations
import argparse
import sys
import tarfile
from importlib import import_module
import pandas as pd

cfg = import_module("00_config")

EXTRACT_DIR = cfg.DATA_INTERIM / "extracted"
MASK_TAR    = cfg.DATA_ORIGINAL / "BreastDCEDL_ISPY2_mask.tar.gz"


LOG = []   # QC: tarball별 해제 결과 → Evaluate/02_extract_log.csv


def extract(tar_path, dest, group=""):
    if not tar_path.exists():
        print(f"  [skip] 없음: {tar_path.name}")
        LOG.append({"tarball": tar_path.name, "group": group, "status": "없음", "entries": 0})
        return 0
    print(f"  풀는 중: {tar_path.name} ({tar_path.stat().st_size/1e9:.1f} GB) ...", flush=True)
    n = 0
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            for m in tf:
                tf.extract(m, dest)   # nosec - 신뢰된 데이터셋
                n += 1
        status = "OK"
    except Exception as e:  # noqa: BLE001  (손상·중단 시에도 로그는 남김)
        status = f"오류:{e}"
        print(f"    [!] {status}")
    print(f"    → {n} 엔트리")
    LOG.append({"tarball": tar_path.name, "group": group, "status": status, "entries": n})
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", action="store_true", help="각 그룹 part1만")
    ap.add_argument("--mask-only", action="store_true")
    args = ap.parse_args()

    cfg.ensure_dirs()
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    # 마스크
    extract(MASK_TAR, EXTRACT_DIR, "mask")
    if not args.mask_only:
        # 영상
        for group, tars in cfg.IMAGE_TARBALLS.items():
            use = tars[:1] if args.proto else tars
            print(f"[{group}] {len(use)} tarball")
            for t in use:
                extract(t, EXTRACT_DIR, group)

    pd.DataFrame(LOG).to_csv(cfg.EVAL_DIR / "02_extract_log.csv", index=False)
    _report(proto=args.proto)
    return 0


def _report(proto=False):
    print(f"\n[해제 위치] {EXTRACT_DIR}")
    niis = list(EXTRACT_DIR.rglob("*.nii.gz"))
    masks = [p for p in niis if "mask" in p.name.lower()]
    imgs  = [p for p in niis if "mask" not in p.name.lower()]
    non_vis1 = [p for p in imgs if "_vis1_" not in p.name]
    print(f"[요약] NIfTI 총 {len(niis)}  (mask {len(masks)} · image {len(imgs)} · vis1 아닌 image {len(non_vis1)})")
    for p in non_vis1[:5]:
        print("   [vis1 아님 → 03에서 무시]", p.relative_to(EXTRACT_DIR))
    print("[예시 image 파일명]")
    for p in imgs[:5]:
        print("   ", p.relative_to(EXTRACT_DIR))
    print("[✓] Evaluate/02_extract_log.csv")
    if proto:
        print("\n⚠ --proto: tarball이 pcr/nopcr로 나뉘어 있어 일부만 풀면 코호트의 pCR 비율이 왜곡됨."
              " 모델링(06)은 전체 해제 후에만 실행.")


if __name__ == "__main__":
    sys.exit(main())
