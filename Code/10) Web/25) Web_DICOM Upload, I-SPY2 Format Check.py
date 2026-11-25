"""
23_dicom_io.py ─ 병원 DICOM 입력 → 추론 입력 (웹 도구용)
==================================================================
  · DCE 시리즈 읽기: 10_t1_features.read_dce와 같은 규칙 (TemporalPositionIdentifier, 없으면 위치별 InstanceNumber)
  · 시점 선택: 촬영 시각이 시점마다 다르면 조영 후 ≈150초(초기)·≈450초(후기)에 가장 가까운 시점,
               시각이 없거나 모두 같으면 I-SPY2 메타데이터 규칙 (초기 = 조영 후 2번째, 후기 = T≤6이면 마지막, 아니면 끝에서 두 번째)
  · 종양 영역: (1) DICOM SEG 파일 — I-SPY "Analysis Mask"는 역부호(0 = 종양)라 10.read_mask, 그 외는 값>0
               (2) 반자동 — 의료진이 지정한 범위(상자) ∩ 초기 증강률 ≥ 70% (I-SPY FTV 정의, 04d의 Duke 처리와 같음)
  · 입력 검사: 시점 수, 크기 일치, 화소 크기, 종양 크기 — 문제가 있으면 무엇이 부족한지 문장으로 반환
  · 저장: 시점별 NIfTI + 마스크 (축 규약은 10b와 같음: numpy 마지막 축 = 슬라이스) → 21_infer.predict_pre 입력
"""
from __future__ import annotations
import glob
from datetime import datetime
from importlib import import_module
from pathlib import Path
import numpy as np

cfg = import_module("00_config")
t1m = import_module("10_t1_features")
EARLY_S, LATE_S, FTV_PE, MIN_VOX = 150.0, 450.0, 70.0, 10


def _secs(h):
    for tag in ("TriggerTime",):                      # ms, 조영 주입 기준
        v = getattr(h, tag, None)
        if v not in (None, ""):
            return float(v) / 1000.0
    for tag in ("AcquisitionTime", "ContentTime"):
        v = str(getattr(h, tag, "") or "")
        if len(v) >= 6:
            t = datetime.strptime(v.split(".")[0][:6], "%H%M%S")
            return t.hour * 3600 + t.minute * 60 + t.second + (float("0." + v.split(".")[1]) if "." in v else 0.0)
    return None


def find_series(root):
    """폴더(압축 해제본) 안 DICOM을 시리즈별로 → 파일 수가 가장 많은 MR 시리즈 폴더 대신 파일 목록 반환."""
    import pydicom
    groups = {}
    for f in glob.glob(str(Path(root) / "**" / "*"), recursive=True):
        p = Path(f)
        if not p.is_file():
            continue
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:  # noqa: BLE001  — DICOM 아닌 파일은 건너뜀
            continue
        groups.setdefault((str(getattr(h, "Modality", "")), str(getattr(h, "SeriesInstanceUID", ""))), []).append(p)
    return groups


def read_dce_files(files):
    """10.read_dce와 같은 정렬·시점 분리 + 시점별 촬영 시각(초). 반환 arr (T,Z,Y,X), zs, sp (z,y,x), nrm, times, flip."""
    import pydicom
    hs = [pydicom.dcmread(str(f)) for f in files]
    nrm = t1m._normal(hs[0])
    pos = np.array([float(np.dot(np.array(h.ImagePositionPatient, float), nrm)) for h in hs])
    if all(hasattr(h, "TemporalPositionIdentifier") for h in hs):
        tp = np.array([int(h.TemporalPositionIdentifier) for h in hs])
    else:
        inst = np.array([int(h.InstanceNumber) for h in hs]); tp = np.zeros(len(hs), int)
        for p in np.unique(np.round(pos, 2)):
            k = np.where(np.round(pos, 2) == p)[0]; tp[k[np.argsort(inst[k])]] = np.arange(len(k))
    vols, zs, times = [], None, []
    for t in np.unique(tp):
        k = np.where(tp == t)[0]; k = k[np.argsort(pos[k])]
        vols.append(np.stack([hs[i].pixel_array.astype(np.float32) * float(getattr(hs[i], "RescaleSlope", 1))
                              + float(getattr(hs[i], "RescaleIntercept", 0)) for i in k]))
        s = [_secs(hs[i]) for i in k]
        times.append(min(x for x in s if x is not None) if any(x is not None for x in s) else None)
        if zs is None:
            zs = pos[k]
    n = min(v.shape[0] for v in vols)
    arr = np.stack([v[:n] for v in vols]); zs = zs[:n]
    dz = float(np.median(np.diff(zs))) if n > 1 else float(hs[0].SliceThickness)
    ps = [float(x) for x in hs[0].PixelSpacing]
    iop = np.array(hs[0].ImageOrientationPatient, float)
    flip = bool(iop[4] < 0)          # 행이 내려갈수록 앞쪽(LPS y 감소) → 화면에서 위아래 뒤집어 앞쪽을 위로
    return arr, zs, (abs(dz), ps[0], ps[1]), nrm, times, flip


def choose_timepoints(times, T):
    """(초기, 후기, 설명). 시점 0 = 조영 전."""
    if times and all(t is not None for t in times) and len(set(np.round(times, 1))) == T:
        rel = np.array(times) - times[0]
        e = int(np.argmin(np.abs(rel[1:] - EARLY_S))) + 1
        l_ = int(np.argmin(np.abs(rel[1:] - LATE_S))) + 1
        l_ = max(l_, e)
        return e, l_, f"촬영 시각 기준 (초기 {rel[e]:.0f}초, 후기 {rel[l_]:.0f}초)"
    e = min(2, T - 1)
    l_ = max(e, T - 1 if T <= 6 else T - 2)
    return e, l_, "촬영 시각 정보 없음 → I-SPY2 시점 규칙"


def read_seg(files, zs, nrm, shape, dz):
    """DICOM SEG → 종양 영역 (Z,Y,X). I-SPY Analysis Mask는 역부호."""
    import pydicom
    s = pydicom.dcmread(str(files[0]))
    if "Analysis Mask" in str(getattr(s, "SeriesDescription", "")):
        m, matched, nf = t1m.read_mask(Path(files[0]).parent, zs, nrm, shape, dz)
        return m
    px = s.pixel_array; px = px[None] if px.ndim == 2 else px
    m = np.zeros(shape, bool)
    for fi, f in enumerate(s.PerFrameFunctionalGroupsSequence):
        p = float(np.dot(np.array(f.PlanePositionSequence[0].ImagePositionPatient, float), nrm))
        k = int(np.argmin(np.abs(zs - p)))
        if abs(zs[k] - p) <= 0.5 * dz + 1e-3 and px[fi].shape == shape[1:]:
            m[k] |= px[fi] > 0
    return m


def pe_early(arr, early, region):
    """초기 증강률(%) — 03·04d와 같은 정의 (분모 하한 = 영역 내 조영 전 중앙값 × 0.1)."""
    pre = arr[0]; ref = float(np.median(pre[region])) if region.any() else float(np.median(pre))
    return 100.0 * (arr[early] - pre) / np.maximum(pre, cfg.PE_FLOOR_FRAC * max(ref, 1e-6))


def box_mask(arr, early, box):
    """box = ((z0,z1),(y0,y1),(x0,x1)) 포함 범위 → 상자 ∩ PE_early ≥ 70% 중 가장 큰 연결 영역. 너무 작으면 상자 전체.
    I-SPY 분석 마스크와의 Dice 중앙값 ≈ 0.59 (T1 DICOM 12명, 종양 bbox를 상자로 줬을 때) — 근사 영역이다.
    연결 영역 조건은 상자 안의 혈관·배경 실질 증강을 떼어 냄 (조건 없을 때 중앙값 0.36)."""
    from scipy import ndimage
    b = np.zeros(arr.shape[1:], bool)
    (z0, z1), (y0, y1), (x0, x1) = box
    b[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = True
    m = b & (pe_early(arr, early, b) >= FTV_PE)
    lab, n = ndimage.label(m)
    if n:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return (m if m.sum() >= MIN_VOX else b), int(m.sum())


def validate(arr, sp, mask, early, late):
    """치명적 문제 목록(분석 불가)과 주의 목록."""
    err, warn = [], []
    T = arr.shape[0]
    if T < 3:
        err.append(f"조영 시점이 {T}개입니다. 조영 전 1개 + 조영 후 2개 이상이 필요합니다.")
    if not 0 < early <= late < T:
        err.append("조영 초기·후기 시점을 정할 수 없습니다.")
    if not all(0.2 <= s <= 10 for s in sp):
        err.append(f"화소 크기가 비정상입니다 ({sp[0]:.2f}×{sp[1]:.2f}×{sp[2]:.2f} mm).")
    if mask is None or mask.sum() < MIN_VOX:
        err.append("종양 영역이 비어 있거나 너무 작습니다.")
    if not 5 <= T <= 9:
        warn.append(f"조영 시점 {T}개: 학습 데이터(I-SPY2, 대부분 6~8개)와 달라 시간 곡선 특징이 덜 정확할 수 있습니다.")
    if sp[0] > 3:
        warn.append(f"슬라이스 두께 {sp[0]:.1f} mm: 학습 데이터보다 두꺼워 질감 특징이 덜 정확할 수 있습니다.")
    return err, warn


def write_inputs(arr, mask, out):
    """시점별 NIfTI + 마스크 저장 (10b와 같은 축 규약). 반환 dce {t: path}, mask path."""
    import SimpleITK as sitk  # noqa: N813
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    dce = {}
    for t in range(arr.shape[0]):
        dce[t] = out / f"dce_t{t}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(np.transpose(arr[t], (2, 1, 0)).astype(np.float32)), str(dce[t]))
    mp = out / "mask.nii.gz"
    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask, (2, 1, 0)).astype(np.uint8)), str(mp))
    return dce, mp
