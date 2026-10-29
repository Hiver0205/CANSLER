"""
10b_t1_maps.py ─ 실험 3: T1 지도를 T0와 같은 형식의 NIfTI로 저장 (CNN·radiomics용)
====================================================================================
10_t1_features의 DICOM 읽기를 재사용해, T0(03·03g)와 같은 규약으로 저장한다.
  · 종양(FTV) bounding box + 10mm crop → 1mm 등방 리샘플
  · 축 규약: numpy 마지막 축 = 축상 슬라이스 (T0 NIfTI와 동일 → 04·04c를 그대로 사용)
  · T1에서 종양이 사라진 환자(FTV < 20 voxel): TCIA 마스크의 수동 VOI(비트 32 미설정 영역)를 마스크로 대체
출력 (Data/Interim/T1/<pid>/): pe_early, pe(후기), ser, pe_t0..5, mask  (.nii.gz)
실행:  python 10b_t1_maps.py [--limit N] [--workers 6]
"""
from __future__ import annotations
import argparse
import glob
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
t1 = import_module("10_t1_features")
OUT = cfg.DATA_INTERIM / "T1"
N_CH, MARGIN_MM = 6, 10.0


def read_mask_full(d, zs, nrm, shape, dz):
    """FTV(값 0)와 수동 VOI(비트 32 미설정) 둘 다 반환."""
    import pydicom
    s = pydicom.dcmread(glob.glob(str(d / "*.dcm"))[0]); px = s.pixel_array
    px = px[None] if px.ndim == 2 else px
    fpos = np.array([float(np.dot(np.array(f.PlanePositionSequence[0].ImagePositionPatient, float), nrm))
                     for f in s.PerFrameFunctionalGroupsSequence])
    ftv = np.zeros(shape, bool); voi = np.zeros(shape, bool)
    for fi, p in enumerate(fpos):
        k = int(np.argmin(np.abs(zs - p)))
        if abs(zs[k] - p) <= 0.5 * dz + 1e-3 and px[fi].shape == shape[1:]:
            ftv[k] = px[fi] == 0; voi[k] = (px[fi].astype(np.int64) & 32) == 0
    return ftv, voi


def one(pid, t0row):
    import SimpleITK as sitk  # noqa: N813
    try:
        arr, zs, sp, nrm = t1.read_dce(t1.T1DIR / pid / "dce")              # (T, Z, Y, X), sp=(z, y, x)
        ftv, voi = read_mask_full(t1.T1DIR / pid / "mask", zs, nrm, arr.shape[1:], sp[0])
        m = ftv if ftv.sum() >= t1.MIN_VOX else voi
        if m.sum() < t1.MIN_VOX:
            return pid, "마스크없음"
        T = arr.shape[0]; n0 = max(int(t0row["n_times"]), 2)
        e_i = int(np.clip(round(t0row["post_early"] * (T - 1) / (n0 - 1)), 1, T - 1))
        l_i = int(np.clip(round(t0row["post_late"] * (T - 1) / (n0 - 1)), e_i, T - 1))
        # crop: 마스크 bbox + 10mm
        nz = np.nonzero(m); sl = []
        for ax, s_mm in zip(range(3), sp):
            mg = int(np.ceil(MARGIN_MM / s_mm))
            sl.append(slice(max(0, nz[ax].min() - mg), min(m.shape[ax], nz[ax].max() + 1 + mg)))
        a = arr[:, sl[0], sl[1], sl[2]]; mc = m[sl[0], sl[1], sl[2]]
        S0 = a[0]; ref = float(np.median(S0[mc])) if mc.any() else float(np.median(S0))
        den = np.maximum(S0, cfg.PE_FLOOR_FRAC * max(ref, 1e-6))
        PE = 100.0 * (a[1:] - S0) / den                                       # (Tpost, z, y, x)
        maps = {"pe_early": PE[e_i - 1], "pe": PE[l_i - 1]}
        d_l = a[l_i] - S0; ok = mc & (d_l > 0)
        ser = np.zeros_like(S0); ser[ok] = np.clip((a[e_i] - S0)[ok] / d_l[ok], 0.0, 3.0); maps["ser"] = ser
        src = np.linspace(0, 1, PE.shape[0]) if PE.shape[0] > 1 else np.array([0.0, 1.0])
        P = PE if PE.shape[0] > 1 else np.concatenate([PE, PE])
        for c, x in enumerate(np.linspace(0, 1, N_CH)):
            j = min(np.searchsorted(src, x, side="right") - 1, len(src) - 2); w = (x - src[j]) / (src[j + 1] - src[j])
            maps[f"pe_t{c}"] = (1 - w) * P[j] + w * P[j + 1]
        maps["mask"] = mc.astype(np.uint8)
        out = OUT / pid; out.mkdir(parents=True, exist_ok=True)
        ref_img = None
        for k, v in maps.items():
            im = sitk.GetImageFromArray(np.transpose(v, (2, 1, 0)).astype(np.uint8 if k == "mask" else np.float32))
            im.SetSpacing((sp[0], sp[1], sp[2]))                              # sitk 인덱스 순서 = (slice, y, x)
            if ref_img is None:
                nsz = [max(1, int(round(s * o / n))) for s, o, n in zip(im.GetSize(), im.GetSpacing(), cfg.RESAMPLE_MM)]
                ref_img = sitk.Resample(im, nsz, sitk.Transform(), sitk.sitkLinear, im.GetOrigin(),
                                        cfg.RESAMPLE_MM, im.GetDirection(), 0.0, sitk.sitkFloat32)
            interp = sitk.sitkNearestNeighbor if k == "mask" else sitk.sitkLinear
            pix = sitk.sitkUInt8 if k == "mask" else sitk.sitkFloat32
            sitk.WriteImage(sitk.Resample(im, ref_img, sitk.Transform(), interp, 0.0, pix), str(out / f"{k}.nii.gz"))
        return pid, "OK" if ftv.sum() >= t1.MIN_VOX else "OK(VOI대체)"
    except Exception as e:  # noqa: BLE001
        return pid, f"오류:{e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_csv(cfg.METADATA_CSV); meta[cfg.COL_PID] = meta[cfg.COL_PID].astype(str); meta = meta.set_index(cfg.COL_PID)
    pts = sorted(p.name for p in t1.T1DIR.iterdir() if (p / "dce" / ".done").exists() and (p / "mask" / ".done").exists()
                 and p.name in meta.index)[: args.limit or None]
    st = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(one, p, meta.loc[p].to_dict()) for p in pts]
        for i, f in enumerate(as_completed(futs), 1):
            pid, s = f.result(); st.append(s)
            if i % 100 == 0 or not s.startswith("OK"):
                print(f"  [{i}/{len(pts)}] {pid} {s}", flush=True)
    print(pd.Series(st).str.split(":").str[0].value_counts().to_string())
    print(f"[✓] T1 지도 → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
