"""
03g_timestack.py ─ J2: 시간축 CNN 입력 (고정 6시점 증강 곡선)
=============================================================
환자마다 조영 후 시점 수(3~10)가 달라, voxel별 PE 곡선을 상대 시간 [0, 1] 위 고정 6점으로 선형 보간한다.
  채널 c (c = 0..5) = PE( 상대시간 c/5 ),  상대시간 = (조영후 시점 순번 − 1) / (조영후 시점 수 − 1)
03의 pe.nii.gz와 같은 격자로 저장 → 같은 mask.nii.gz로 슬라이스 선택 가능. 환자 내부 계산(라벨 미사용).

출력 (Data/Interim/<pid>/): pe_t0.nii.gz … pe_t5.nii.gz
실행:  python 03g_timestack.py [--limit N]
"""
from __future__ import annotations
import argparse
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
pp = import_module("03_preprocess")
N_CH = 6


def stack_one(pid, r, spacing, idx, sitk):
    t_pre, n_t = int(r[cfg.COL_T_PRE]), int(r["n_times"])
    mask_f, err = pp._lookup(idx, cfg.MASK_PATTERN.format(pid=pid))
    ref_pe = cfg.DATA_INTERIM / pid / "pe.nii.gz"
    if err or not ref_pe.exists():
        return "입력없음"
    files = {t: f for t in range(n_t) if (f := pp._lookup(idx, cfg.DCE_TP_PATTERN.format(pid=pid, t=t))[0])}
    post = sorted(t for t in files if t > t_pre)
    if t_pre not in files or len(post) < 2:
        return f"시점부족({len(post)})"
    mask = sitk.Cast(sitk.ReadImage(str(mask_f), sitk.sitkUInt8) > 0, sitk.sitkUInt8)
    mask.SetSpacing(spacing); mask.SetOrigin((0.0, 0.0, 0.0)); mask.SetDirection(pp.IDENTITY)
    box = pp._crop_box(mask, spacing, sitk)
    crop = lambda im: sitk.RegionOfInterest(im, box[1], box[0])  # noqa: E731
    S0i = crop(pp._read(files[t_pre], spacing, sitk, sitk.sitkFloat32))
    a0 = sitk.GetArrayFromImage(S0i); m = sitk.GetArrayFromImage(crop(mask)).astype(bool)
    ref = float(np.median(a0[m]))
    if not ref > 0:
        return "pre신호이상"
    den = np.maximum(a0, cfg.PE_FLOOR_FRAC * ref)
    PE = np.stack([100.0 * (sitk.GetArrayFromImage(crop(pp._read(files[t], spacing, sitk, sitk.sitkFloat32))) - a0)
                   / den for t in post])                                         # (T, …)
    src = np.linspace(0.0, 1.0, PE.shape[0]); dst = np.linspace(0.0, 1.0, N_CH)
    ref_img = sitk.ReadImage(str(ref_pe))
    for c, x in enumerate(dst):                                                  # 선형 보간
        j = min(np.searchsorted(src, x, side="right") - 1, len(src) - 2)
        w = (x - src[j]) / (src[j + 1] - src[j])
        ch = ((1 - w) * PE[j] + w * PE[j + 1]).astype(np.float32)
        im = sitk.GetImageFromArray(ch); im.CopyInformation(S0i)
        sitk.WriteImage(sitk.Resample(im, ref_img, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32),
                        str(cfg.DATA_INTERIM / pid / f"pe_t{c}.nii.gz"))
    return "OK"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sitk = pp._sitk()
    meta = pd.read_csv(cfg.METADATA_CSV)
    idx = pp._index(pp.EXTRACT_DIR)
    st = []
    for i, r in enumerate(meta.to_dict("records")[: args.limit or None], 1):
        pid = str(r[cfg.COL_PID])
        try:
            sp = (float(r[cfg.COL_SLICE_THICK]), float(r[cfg.COL_XY_SPACING]), float(r[cfg.COL_XY_SPACING]))
            s = stack_one(pid, r, sp, idx, sitk)
        except Exception as e:  # noqa: BLE001
            s = f"오류:{e}"
        st.append(s)
        if s != "OK" or i % 200 == 0:
            print(f"  [{i}] {pid} {s}", flush=True)
    print(f"[✓] 시간 스택 OK {st.count('OK')}/{len(st)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
