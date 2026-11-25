"""
make_synthetic.py ─ 가상 유방 DCE-MRI 생성 (실제 환자 자료 없음)
==================================================================
I-SPY2 공개 자료(BreastDCEDL_ISPY2)와 같은 파일 형식·이름 규칙으로 가상 환자 몇 명을 만든다.
  - 조영 시점별 NIfTI 8개 (t=0 조영 전, 2 = 초기, 6 = 후기), uint16, 배열 (256, 256, 80)
  - 종양 마스크 NIfTI (조영 초기 증강률 70% 이상 = I-SPY2 FTV와 같은 기준)
  - 메타데이터 CSV (시점 번호·화소 크기·임상 정보) — 데모용 가상 값
영상은 프로그램으로 만든 모형이다: 지방·유선 조직·흉벽·심장 + 환자마다 다른 조영 곡선을 가진 종양
(빨리 들어왔다 빠짐 / 정체 / 계속 상승, 괴사 중심). 예측 숫자에는 의미가 없고, 파이프라인이 끝까지 도는지 보이는 용도.

  python demo/make_synthetic.py --root demo_root --n 4
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SHAPE = (256, 256, 80)            # (좌우, 앞뒤, 단면) — 원본과 같은 배열 모양
XY, THICK = 0.7, 2.0              # 화소 크기 mm
N_TIMES, T_EARLY, T_LATE = 8, 2, 6

# 가상 환자: (아형, 조영 양상, 종양 반지름 mm, 괴사, 나이, 폐경, HR, HER2)
PROFILES = [
    ("HER2pos", "washout", 16, False, 47, 0, 0, 1),
    ("TripleNeg", "washout", 19, True, 39, 0, 0, 0),
    ("HRposHER2neg", "persistent", 11, False, 58, 1, 1, 0),
    ("HRposHER2neg", "plateau", 14, False, 52, 1, 1, 0),
    ("HER2pos", "plateau", 13, False, 44, 0, 1, 1),
    ("TripleNeg", "washout", 12, False, 61, 1, 0, 0),
]


def curve(kind, t):
    """조영 전 대비 상대 증강 (0 = 조영 전). t = 0..7 시점."""
    if t == 0:
        return 0.0
    rise = 1 - np.exp(-1.3 * t)
    if kind == "washout":
        return 2.0 * rise * np.exp(-0.06 * max(t - T_EARLY, 0))
    if kind == "plateau":
        return 1.6 * rise
    return 1.1 * rise + 0.12 * t                       # persistent: 계속 상승


def smooth_noise(rng, shape, sigma):
    n = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return n / (n.std() + 1e-6)


def make_patient(rng, prof):
    sub, kind, r_mm, necro, *_ = prof
    x, y, z = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in SHAPE), indexing="ij")
    X, Y, Z = (x - 128) * XY, y * XY, (z - 40) * THICK     # mm, Y = 흉벽(0) → 유두 방향
    # 몸: 흉벽(앞뒤 0~35 mm) + 유방(반타원체)
    chest = Y < 35
    breast = ((X / 85) ** 2 + ((Y - 35) / 120) ** 2 + (Z / 70) ** 2 < 1) & (Y >= 35)
    body = chest | breast
    heart = chest & (((X + 10) / 55) ** 2 + ((Y + 5) / 30) ** 2 + (Z / 45) ** 2 < 1)
    muscle = (Y >= 22) & (Y < 35) & (np.abs(X) < 80)
    fgt = breast & (smooth_noise(rng, SHAPE, (6, 6, 2)) > 0.35) & (Y < 120)   # 유선 조직 (가지 모양)
    # 종양: 울퉁불퉁한 타원체
    c = np.array([rng.uniform(-35, 35), rng.uniform(65, 95), rng.uniform(-20, 20)])
    d = np.sqrt(((X - c[0]) / 1.0) ** 2 + ((Y - c[1]) / 0.85) ** 2 + ((Z - c[2]) / 1.1) ** 2)
    rr = r_mm * (1 + 0.22 * smooth_noise(rng, SHAPE, (5, 5, 2)))
    tumor = (d < rr) & breast
    het = 0.75 + 0.25 * smooth_noise(rng, SHAPE, (2, 2, 1))                   # 종양 내부 불균일
    if necro:
        core = d < 0.45 * r_mm
        het = np.where(core, 0.08, het)
    base = np.zeros(SHAPE, np.float32)
    base[breast] = 120; base[fgt] = 260; base[muscle] = 210; base[heart] = 240
    base[chest & ~heart & ~muscle] = 150; base[tumor] = 230
    frames = []
    for t in range(N_TIMES):
        e = np.zeros(SHAPE, np.float32)
        e[fgt] = 0.04 * t                                                      # 배경 실질 조영 (천천히)
        e[muscle] = 0.25 * (1 - np.exp(-0.8 * t))
        e[heart] = 2.5 * curve("washout", t)
        e[tumor] = (curve(kind, t) * het)[tumor]
        img = base * (1 + e)
        img = ndi.gaussian_filter(img, (0.7, 0.7, 0.4))
        img += rng.normal(0, 6, SHAPE).astype(np.float32) * body + np.abs(rng.normal(0, 3, SHAPE)) * ~body
        frames.append(np.clip(img, 0, 4000).astype(np.uint16))
    pe = (frames[T_EARLY].astype(np.float32) - frames[0]) / np.maximum(frames[0].astype(np.float32), 1)
    mask = tumor & (pe > 0.7)                                                  # FTV 기준 (I-SPY2와 같음)
    lab, n = ndi.label(mask)
    if n:                                                                      # 잡음 조각(20화소 미만) 제거
        sz = np.bincount(lab.ravel()); sz[0] = 0
        mask = (sz >= 20)[lab]
    return frames, mask.astype(np.uint8)


def main():
    import SimpleITK as sitk  # noqa: N813
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="데모 데이터 루트 (BREASTCANCER_ROOT로 씀)")
    ap.add_argument("--n", type=int, default=4, help=f"가상 환자 수 (최대 {len(PROFILES)})")
    a = ap.parse_args()
    root = Path(a.root)
    ext = root / "Data" / "Interim" / "extracted"
    rows = []
    for i, prof in enumerate(PROFILES[:a.n], 1):
        pid = f"DEMO-{i:04d}"
        rng = np.random.default_rng(1000 + i)
        frames, mask = make_patient(rng, prof)
        d = ext / "BreastDCEDL_ISPY2_demo_dce" / pid; d.mkdir(parents=True, exist_ok=True)
        for t, f in enumerate(frames):
            sitk.WriteImage(sitk.GetImageFromArray(f), str(d / f"{pid}_spy2_vis1_dce_aqc_{t}.nii.gz"))
        md = ext / "BreastDCEDL_ISPY2_mask"; md.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(mask), str(md / f"{pid}_{pid}_spy2_vis1_mask.nii.gz"))
        nz = np.nonzero(mask)
        sub, kind, r_mm, necro, age, meno, hr, her2 = prof
        rows.append({"pid": pid, "pCR": 0.0, "n_xy": 256.0, "n_z": 80.0, "n_times": float(N_TIMES), "pre": 0.0,
                     "post_early": float(T_EARLY), "post_late": float(T_LATE), "slice_thick": THICK, "xy_spacing": XY,
                     "mask_start": float(nz[2].min()), "mask_end": float(nz[2].max()), "sraw": float(nz[1].min()),
                     "eraw": float(nz[1].max()), "scol": float(nz[0].min()), "ecol": float(nz[0].max()),
                     "tum_vol": float(mask.sum() * XY * XY * THICK), "age": float(age), "menopause": float(meno),
                     "race_white": 1.0, "race_black": 0.0, "HR": float(hr), "HER2": float(her2), "HR_HER2_STATUS": sub,
                     "TripleNeg": float(sub == "TripleNeg"), "HER2pos": float(sub == "HER2pos"),
                     "HRposHER2neg": float(sub == "HRposHER2neg"), "dataset": "demo", "test": 1.0})
        print(f"[✓] {pid}: {sub}, {kind}{' · 괴사' if necro else ''}, 종양 {mask.sum() * XY * XY * THICK / 1000:.1f} cm³",
              flush=True)
    meta = root / "Data" / "Original"; meta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(meta / "BreastDCEDL_ISPY2_metadata.csv", index=False)
    print(f"[✓] 메타데이터: {meta / 'BreastDCEDL_ISPY2_metadata.csv'}")


if __name__ == "__main__":
    main()
