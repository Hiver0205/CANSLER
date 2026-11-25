"""
bpr_core.py ─ 웹 화면(app.py)과 일괄 분석(precompute.py)이 함께 쓰는 공통 로직 (Streamlit 의존 없음)
==================================================================
- 환자 목록(I-SPY2 메타데이터), 모델 입력 변환, I-SPY2 자료에서 영상 경로 찾기
- 분석 실행 + 결과 저장(Web/cache/<모드>/<등록번호>/) — 한 번 분석한 환자는 다시 열 때 바로 보임
- 화면 표시용 배열(축상·시상·관상·MIP)과 3D 표면(marching cubes)
"""
from __future__ import annotations
import json
import os
import pickle
import sys
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Code1"))
import torch  # noqa: F401,E402  — Windows DLL 충돌 방지: SimpleITK·pyradiomics보다 먼저
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

cfg = import_module("00_config")
pp = import_module("03_preprocess")
CACHE = WEB / "cache"          # 환자별 분석 결과 (환자 단위 자료 — GitHub에 올리지 않음)
NAMES_CSV = WEB / "patient_names.csv"   # (선택) 등록번호→환자명. I-SPY2 공개 자료는 이름이 없어 "익명 번호"로 표시
NOTES = CACHE / "notes"        # 의사 소견 (환자별 json)
REG = CACHE / "registered"     # 웹에서 새로 등록한 환자 (정보 patients.json + 영상 <등록번호>/, T1/<등록번호>/)
DEMO = os.environ.get("BPR_DEMO") == "1"   # demo/run_demo.py: 가상 MRI 데이터로 실행 (실제 환자 자료 없음)
SUBTYPES = ["HER2+", "HR+HER2−", "TNBC"]
SUBTYPE_RATE = {"HER2+": 45.9, "TNBC": 39.3, "HR+HER2−": 16.8}   # I-SPY2 982명 실제 pCR 비율 (Results 01)


def subtype_of(hr, her2):
    if her2 == "양성":
        return "HER2+"
    if hr == "양성":
        return "HR+HER2−"
    return "TNBC" if hr == "음성" and her2 == "음성" else "판정 불가"


def to_clinical(p):
    """환자 정보 → 모델 입력 (05와 같은 0/1 규칙, 모르면 None)."""
    yn = {"양성": 1, "음성": 0}
    return {"HR": yn.get(p["HR"]), "HER2": yn.get(p["HER2"]),
            "age": float(p["나이"]) if str(p["나이"]).strip() not in ("", "nan", "None") else None,
            "race_white": 1 if p["인종"] == "백인" else 0,  # 아시아인·기타 = 둘 다 0 (학습 때 '기타'와 같음) "race_black": 1 if p["인종"] == "흑인" else 0,
            "menopause": {"폐경 후": 1, "폐경 전": 0}.get(p["폐경"])}


def patients():
    """I-SPY2 메타데이터 → 환자 목록. 구분 = 학습(개발셋 784명) / 검증(학습에 안 쓴 198명)."""
    m = pd.read_csv(cfg.METADATA_CSV)
    m = m[m["pCR"].notna()].copy()
    yn = lambda v, one, zero: "" if pd.isna(v) else (one if int(v) == 1 else zero)  # noqa: E731
    t1 = cfg.DATA_INTERIM / "T1_dicom"
    df = pd.DataFrame({
        "등록번호": m[cfg.COL_PID].astype(str),
        "나이": m["age"].map(lambda v: "" if pd.isna(v) else str(int(v))),
        "HR": m["HR"].map(lambda v: yn(v, "양성", "음성")),
        "HER2": m["HER2"].map(lambda v: yn(v, "양성", "음성")),
        "폐경": m["menopause"].map(lambda v: yn(v, "폐경 후", "폐경 전")),
        "인종": np.where(m["race_white"] == 1, "백인", np.where(m["race_black"] == 1, "흑인", "기타")),
        "구분": np.where(m[cfg.COL_SPLIT].isin(cfg.HOLDOUT_SPLITS), "검증", "학습"),
        "실제": m["pCR"].astype(int).map({1: "pCR", 0: "잔존 종양"}),
        "부피": (m["tum_vol"] / 1000).round(1),             # 치료 전 종양 부피 cm³ (I-SPY2 기록)
    }).reset_index(drop=True)
    df["아형"] = [subtype_of(h, e) for h, e in zip(df["HR"], df["HER2"])]
    df["3주차"] = [(t1 / p / "dce" / ".done").exists() for p in df["등록번호"]]
    names = {}
    if NAMES_CSV.exists():
        n = pd.read_csv(NAMES_CSV, dtype=str)
        names = dict(zip(n["등록번호"].str.strip(), n["환자명"].str.strip()))
    df["환자명"] = [names.get(p, "익명 " + p.split("-")[-1]) for p in df["등록번호"]]
    if DEMO:                                   # 가상 데이터 데모: 실제 수술 결과가 없는 가상 환자
        df["환자명"] = ["가상 환자 " + p.split("-")[-1] for p in df["등록번호"]]
        df["실제"] = ""
    df["기타"], df["담당"] = "", ""          # 담당 = 새로 등록한 의료진 아이디 (I-SPY2 공개 환자는 공용)
    reg = registered()
    if reg:
        R = pd.DataFrame([{k: r.get(k, "") for k in ("등록번호", "환자명", "나이", "HR", "HER2", "폐경", "인종", "기타", "담당")}
                          for r in reg])
        R["구분"], R["실제"] = "등록", ""
        R["부피"] = [r.get("부피", float("nan")) for r in reg]
        R["아형"] = [subtype_of(h, e) for h, e in zip(R["HR"], R["HER2"])]
        R["3주차"] = [bool(r.get("inputs") and r["inputs"].get("w3")) for r in reg]
        df = pd.concat([R, df[~df["등록번호"].isin(R["등록번호"])]], ignore_index=True)   # 새로 등록한 환자가 맨 위
    return df


def registered():
    f = REG / "patients.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def register(rec):
    """새 환자 1명 저장 (같은 등록번호가 있으면 바꿔 씀). rec: 등록번호·환자명·나이·HR·HER2·인종·폐경·기타·inputs."""
    import datetime as dt
    REG.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "등록일": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    reg = [r for r in registered() if r["등록번호"] != rec["등록번호"]] + [rec]
    (REG / "patients.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")


def unregister(pid, uid):
    """새로 등록한 환자 1명 삭제 (본인이 등록한 환자만). 정보·영상·분석 결과·소견·대기자 명단에서 모두 지움.
    반환: 오류 문장 ("" = 성공). I-SPY2 공개 환자는 지울 수 없음."""
    import shutil
    reg = registered()
    rec = next((r for r in reg if r["등록번호"] == pid), None)
    if rec is None:
        return "새로 등록한 환자만 삭제할 수 있습니다."
    if rec.get("담당") != uid:
        return "본인이 등록한 환자만 삭제할 수 있습니다."
    (REG / "patients.json").write_text(json.dumps([r for r in reg if r["등록번호"] != pid], ensure_ascii=False,
                                                  indent=1), encoding="utf-8")
    for base in (REG, REG / "T1", CACHE / "pre", CACHE / "t3"):
        d = base / pid
        if d.resolve().parent == base.resolve() and d.is_dir():   # 등록번호에 경로 문자가 있어도 폴더 밖은 안 지움
            shutil.rmtree(d, ignore_errors=True)
    for ud in (NOTES.iterdir() if NOTES.exists() else []):
        f = ud / f"{pid}.json"
        if f.resolve().parent == ud.resolve():
            f.unlink(missing_ok=True)
    _queue_save(uid, [x for x in queue_load(uid) if x["pid"] != pid])
    return ""


def update_registered(pid, uid, fields):
    """새로 등록한 환자 정보 수정 (본인이 등록한 환자만, 등록번호·담당·등록일은 그대로). 반환: 오류 문장 ("" = 성공)."""
    reg = registered()
    i = next((k for k, r in enumerate(reg) if r["등록번호"] == pid), None)
    if i is None:
        return "새로 등록한 환자만 수정할 수 있습니다."
    if reg[i].get("담당") != uid:
        return "본인이 등록한 환자만 수정할 수 있습니다."
    fields = {k: v for k, v in fields.items() if k not in ("등록번호", "담당", "등록일")}
    reg[i] = {**reg[i], **fields}
    (REG / "patients.json").write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    return ""


# ── I-SPY2 환자 정보 수정·숨김 (원본 자료는 그대로, 의료진 계정별 화면에만 적용) ─────────────
OVR = CACHE / "overrides"
OVR_FIELDS = ("환자명", "나이", "HR", "HER2", "폐경", "인종", "기타")


def ovr_load(uid):
    f = OVR / f"{uid}.json"
    d = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    return {"edit": d.get("edit", {}), "hidden": d.get("hidden", [])}


def _ovr_save(uid, d):
    OVR.mkdir(parents=True, exist_ok=True)
    (OVR / f"{uid}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def ovr_edit(uid, pid, fields):
    d = ovr_load(uid)
    d["edit"][pid] = {k: fields[k] for k in OVR_FIELDS if k in fields}
    _ovr_save(uid, d)


def ovr_hide(uid, pid, hide=True):
    d = ovr_load(uid)
    d["hidden"] = [p for p in d["hidden"] if p != pid] + ([pid] if hide else [])
    _ovr_save(uid, d)
    if hide:
        _queue_save(uid, [x for x in queue_load(uid) if x["pid"] != pid])


def apply_overrides(df, uid):
    """환자 목록에 이 계정의 수정·숨김 반영 (I-SPY2 환자만; 새로 등록한 환자는 patients.json을 직접 고침)."""
    d = ovr_load(uid)
    if not d["edit"] and not d["hidden"]:
        return df
    df = df[~(df["등록번호"].isin(d["hidden"]) & (df["구분"] != "등록"))].copy()
    for pid, f in d["edit"].items():
        i = df.index[(df["등록번호"] == pid) & (df["구분"] != "등록")]
        for k, v in f.items():
            if k in df.columns:
                df.loc[i, k] = v
        if len(i):
            df.loc[i, "아형"] = subtype_of(df.loc[i[0], "HR"], df.loc[i[0], "HER2"])
    return df


def clear_results(pid, which=("pre", "t3")):
    """영상을 다시 올렸을 때 예전 분석 결과 삭제 (다시 분석하도록)."""
    import shutil
    for w in which:
        base = CACHE / w; d = base / pid
        if d.resolve().parent == base.resolve() and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _reg_inputs(pid):
    for r in registered():
        if r["등록번호"] == pid and r.get("inputs"):
            i = r["inputs"]
            return {"dce": {int(t): Path(v) for t, v in i["dce"].items()}, "mask": Path(i["mask"]),
                    "t_pre": int(i["t_pre"]), "t_early": int(i["t_early"]), "t_late": int(i["t_late"]),
                    "spacing": tuple(i["spacing"]), "flip": bool(i["flip"]),
                    "w3": tuple(Path(x) for x in i["w3"]) if i.get("w3") else None}
    return None


_IDX = None


def demo_inputs(pid):
    """I-SPY2 공개 자료(또는 웹에서 등록한 환자의 저장 영상)에서 영상·마스크·시점·spacing (없으면 None)."""
    global _IDX
    r = _reg_inputs(pid)
    if r is not None:
        return r
    if _IDX is None:
        meta = pd.read_csv(cfg.METADATA_CSV); meta[cfg.COL_PID] = meta[cfg.COL_PID].astype(str)
        _IDX = (pp._index(pp.EXTRACT_DIR), meta.set_index(cfg.COL_PID))
    idx, meta = _IDX
    if pid not in meta.index:
        return None
    r = meta.loc[pid]
    dce = {t: idx[cfg.DCE_TP_PATTERN.format(pid=pid, t=t)][0] for t in range(int(r["n_times"]))
           if cfg.DCE_TP_PATTERN.format(pid=pid, t=t) in idx}
    mk = idx.get(cfg.MASK_PATTERN.format(pid=pid))
    if not dce or not mk:
        return None
    t1 = cfg.DATA_INTERIM / "T1_dicom" / pid
    return {"dce": dce, "mask": mk[0], "t_pre": int(r["pre"]), "t_early": int(r["post_early"]),
            "t_late": int(r["post_late"]),
            "spacing": (float(r["slice_thick"]), float(r["xy_spacing"]), float(r["xy_spacing"])),
            "w3": (t1 / "dce", t1 / "mask") if (t1 / "dce" / ".done").exists() else None, "flip": True}


# ── 분석 실행·저장 ───────────────────────────────────────────────
def run(pid, mode, inputs, clin, save=True, progress=None):
    """분석 1건. save=True면 Web/cache/<모드>/<등록번호>/에 결과를 남김 (I-SPY2 자료로 분석한 경우).
    progress = f(0~1, 단계 이름). 3주 차 모드는 저장된 치료 전 결과가 있으면 그것을 이어 씀 (같은 I-SPY2 입력일 때)."""
    inf = import_module("21_infer")
    d = CACHE / mode / pid
    work = d / "work" if save else None
    if save:
        work.mkdir(parents=True, exist_ok=True)
    args = (pid, inputs["dce"], inputs["mask"], inputs["t_pre"], inputs["t_early"], inputs["t_late"],
            inputs["spacing"], clin)
    if mode == "pre":
        res = inf.predict_pre(*args, work=work, progress=progress)
    else:
        pre = load_saved(pid, "pre") if save else None
        if pre is not None and not Path(pre["work"]).exists():
            pre = None
        res = inf.predict_t3(*args, *inputs["w3"], work=work, progress=progress, pre=pre)
    res["inputs"] = inputs
    if save:
        with open(d / "result.pkl", "wb") as f:
            pickle.dump(res, f)
        (d / "summary.json").write_text(json.dumps({"pid": pid, "mode": mode, "prob": float(res["prob"])}),
                                        encoding="utf-8")
    return res


def load_saved(pid, mode):
    f = CACHE / mode / pid / "result.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def load_note(pid, uid):
    """의사 소견은 의료진 계정별로 따로 저장 (다른 계정에서는 보이지 않음)."""
    f = NOTES / uid / f"{pid}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"text": "", "saved": ""}


def save_note(pid, text, uid):
    import datetime as dt
    (NOTES / uid).mkdir(parents=True, exist_ok=True)
    j = {"text": text, "saved": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    (NOTES / uid / f"{pid}.json").write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return j


def saved_probs():
    """저장된 분석 결과 → {(등록번호, 모드): 확률}."""
    out = {}
    for f in CACHE.glob("*/*/summary.json"):
        try:
            j = json.loads(f.read_text(encoding="utf-8")); out[(j["pid"], j["mode"])] = j["prob"]
        except Exception:  # noqa: BLE001 — 쓰는 중인 파일은 건너뜀
            pass
    return out


# ── 화면 표시용 배열 ─────────────────────────────────────────────
def phase_names(n, t_pre, t_early, t_late=None):
    """조영 시점 이름: 조영 전 / 1차 / 2차(초기) / ... / n차(후기)."""
    out = []
    for t in range(n):
        if t == t_pre:
            out.append("조영 전"); continue
        tag = {t_early: " (초기)", t_late: " (후기)"}.get(t, "")
        out.append(f"{t - t_pre}차{tag}")
    return out


def view_arrays(work, dce_items, t_pre, t_early, mask_path, spacing):
    """모든 조영 시점 영상(phases, float16) + 조영 초기(raw)·차이(sub) + 종양 영역·habitat·SER을 같은 격자로.
    dce_items = ((시점, 경로), ...). numpy 축 = (좌우, 앞뒤, 단면), 화소 크기 (xy, xy, 두께)."""
    import SimpleITK as sitk  # noqa: N813
    dce = dict(dce_items)
    raw = pp._read(Path(dce[t_early]), tuple(spacing), sitk, sitk.sitkFloat32)
    pre = pp._read(Path(dce[t_pre]), tuple(spacing), sitk, sitk.sitkFloat32)
    mk = sitk.Cast(sitk.ReadImage(str(mask_path), sitk.sitkUInt8) > 0, sitk.sitkUInt8)
    mk.SetSpacing(tuple(spacing)); mk.SetOrigin((0.0, 0.0, 0.0)); mk.SetDirection(pp.IDENTITY)
    w = Path(work)

    def to_raw(im, nn):
        return sitk.GetArrayFromImage(sitk.Resample(
            im, raw, sitk.Transform(), sitk.sitkNearestNeighbor if nn else sitk.sitkLinear, 0.0,
            sitk.sitkUInt8 if nn else sitk.sitkFloat32))

    a_raw, a_pre = sitk.GetArrayFromImage(raw), sitk.GetArrayFromImage(pre)
    out = {"raw": a_raw, "sub": np.clip(a_raw - a_pre, 0, None) if a_pre.shape == a_raw.shape else a_raw,
           "mask": sitk.GetArrayFromImage(mk).astype(bool),
           "hab": to_raw(sitk.ReadImage(str(w / "hab3.nii.gz")), True),
           "ser": to_raw(sitk.ReadImage(str(w / "ser.nii.gz")), False),
           "vox": (float(spacing[1]), float(spacing[2]), float(spacing[0]))}
    ph = []
    for t in sorted(dce):
        a = a_raw if t == t_early else a_pre if t == t_pre else sitk.GetArrayFromImage(
            pp._read(Path(dce[t]), tuple(spacing), sitk, sitk.sitkFloat32))
        ph.append(a.astype(np.float16) if a.shape == a_raw.shape else np.zeros(a_raw.shape, np.float16))
    out["phases"], out["t_pre"], out["t_early"] = np.stack(ph), sorted(dce).index(t_pre), sorted(dce).index(t_early)
    out["feret_mm"] = feret(out["mask"], out["vox"])
    return out


def view_arrays_w3(dce_dir, mask_dir, n_times, post_early):
    """3주 차 원본(DICOM) 조영 전·초기 영상과 종양 영역 → view_arrays와 같은 축 순서 (좌우, 앞뒤, 단면)."""
    t1m = import_module("10_t1_features")
    arr, zs, sp, nrm = t1m.read_dce(Path(dce_dir))
    m, _, _ = t1m.read_mask(Path(mask_dir), zs, nrm, arr.shape[1:], sp[0])
    T = arr.shape[0]; e = int(np.clip(round(post_early * (T - 1) / max(n_times - 1, 1)), 1, T - 1))
    tr = lambda a: np.transpose(a, (2, 1, 0))  # noqa: E731  (단면, 행, 열) → (열, 행, 단면)
    raw = tr(arr[e].astype(np.float32))
    mk = tr(m.astype(bool))
    vox = (float(sp[2]), float(sp[1]), float(sp[0]))
    return {"raw": raw, "sub": np.clip(raw - tr(arr[0].astype(np.float32)), 0, None), "mask": mk, "vox": vox,
            "phases": np.stack([tr(a).astype(np.float16) for a in arr]), "t_pre": 0, "t_early": e,
            "feret_mm": feret(mk, vox)}


def feret(mask, vox):
    """종양 최대 직경(mm) — 가장 큰 연결 병변의 표면 화소 사이 최대 거리."""
    from scipy import ndimage as ndi
    from scipy.spatial.distance import pdist
    if not mask.any():
        return float("nan")
    lab, n = ndi.label(mask)                    # 가장 큰 병변 기준 (흩어진 작은 조각이 직경을 부풀리지 않게)
    if n > 1:
        mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    edge = mask & ~ndi.binary_erosion(mask)
    pts = np.argwhere(edge) * np.array(vox)
    try:                                        # 최대 거리는 볼록 껍질 꼭짓점 사이에 있음 → 정확하고 빠름
        from scipy.spatial import ConvexHull
        pts = pts[ConvexHull(pts).vertices]
    except Exception:  # noqa: BLE001 — 납작한 영역 등
        pass
    return float(pdist(pts).max()) if len(pts) > 1 else 0.0


def tumor_stats(V):
    """종양 영역 수치: 부피, 시점별 평균 조영 증강률(%), 최고 증강률, 조영 곡선 유형, SER·habitat 구성."""
    m = V["mask"]; vv = float(np.prod(V["vox"]))
    o = {"vol_cm3": float(m.sum()) * vv / 1000, "feret_mm": V.get("feret_mm", float("nan"))}
    if m.any() and "phases" in V:
        ph = V["phases"]; p0 = float(ph[V["t_pre"]][m].astype(np.float32).mean())
        curve = [(float(ph[t][m].astype(np.float32).mean()) - p0) / max(p0, 1e-6) * 100 for t in range(len(ph))]
        o["curve"] = curve
        e = V["t_early"]; o["early_pct"] = curve[e]; o["peak_pct"] = max(curve); o["peak_t"] = int(np.argmax(curve))
        late = curve[-1]; o["late_change"] = (late - curve[e]) / max(abs(curve[e]), 1e-6) * 100
        o["curve_type"] = ("유출형" if o["late_change"] < -10 else
                           "지속 상승형" if o["late_change"] > 10 else "평탄형")
        rel = (ph[e][m].astype(np.float32) - ph[V["t_pre"]][m].astype(np.float32)) / max(p0, 1e-6)
        o["enh_frac"] = float((rel > 0.7).mean() * 100)          # 초기 증강 70% 넘는 종양 비율 (FTV 기준)
    if V.get("ser") is not None and m.any():
        s = V["ser"][m]; s = s[s > 0]
        if len(s):
            o["ser_mean"] = float(np.mean(s)); o["washout_frac"] = float((s > 1.1).mean() * 100)
    return o


def plane(V, key, view, k, flip):
    """2D 단면 (화면 행, 열)과 세로/가로 화소 비. view = 축상·시상·관상·MIP."""
    a = V[key]
    vx, vy, vz = V["vox"]
    if view == "MIP":
        img = (a.max(axis=2) if key != "mask" else a.any(axis=2)).T
        return (np.flipud(img) if flip else img), vy / vx
    if view == "축상":
        img = a[:, :, k].T
        return (np.flipud(img) if flip else img), vy / vx
    if view == "시상":
        img = a[k, :, :].T                     # (단면, 앞뒤)
        img = img[:, ::-1] if flip else img     # 앞쪽이 왼쪽
        return np.flipud(img), vz / vy          # 위쪽 단면이 위로
    img = a[:, k, :].T                          # 관상: (단면, 좌우)
    return np.flipud(img), vz / vx


def n_slices(V, view):
    return {"축상": V["raw"].shape[2], "시상": V["raw"].shape[0], "관상": V["raw"].shape[1]}.get(view, 1)


def center(V, view):
    m = V["mask"]
    ax = {"축상": (0, 1), "시상": (1, 2), "관상": (0, 2)}.get(view)
    if ax is None or not m.any():
        return n_slices(V, view) // 2
    return int(np.argmax(m.sum(ax)))


# ── 3D 표면 ─────────────────────────────────────────────────────
def surfaces(V, color_by="hab"):
    """3D 표시용: 주변 조직 볼륨(조영 전후 차이) + 종양 표면(꼭짓점 색 = habitat 또는 SER). 좌표 mm."""
    from scipy import ndimage as ndi
    from skimage import measure
    vox = np.array(V["vox"])
    out = {}
    # 주변 조직 볼륨 (3D MIP 느낌): 조영 전후 차이 영상을 약 2.5 mm 격자로 줄여 값 0~255
    if V.get("sub") is not None:
        g = np.maximum(1, np.round(2.5 / vox)).astype(int)
        sv = ndi.uniform_filter(V["sub"].astype(np.float32), size=tuple(g))[::g[0], ::g[1], ::g[2]]
        lo, hi = np.percentile(sv, 90), np.percentile(sv, 99.9)
        val = np.clip((sv - lo) / max(hi - lo, 1e-6), 0, 1)
        X, Y, Z = np.meshgrid(*[(np.arange(n) * g[d] * vox[d]).astype(np.float32) for d, n in enumerate(sv.shape)],
                              indexing="ij")
        out["vol"] = (X.ravel(), Y.ravel(), Z.ravel(), (val * 255).astype(np.uint8).ravel())
    m = V["mask"]
    if m.any():
        sl = ndi.find_objects(m.astype(np.uint8))[0]
        sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
        sub = m[sl]
        sm = ndi.gaussian_filter(sub.astype(np.float32), sigma=0.8)
        v, fc, _, _ = measure.marching_cubes(sm, 0.5, spacing=tuple(vox))
        off = np.array([s.start for s in sl]) * vox
        idx = np.clip(np.round(v / vox).astype(int), 0, np.array(sub.shape) - 1)
        # 꼭짓점 → 가장 가까운 종양 안 화소의 값
        val = None
        if color_by in V and V.get(color_by) is not None:
            src = V[color_by][sl]
            ref = sub & (src > 0) if color_by == "hab" and (sub & (src > 0)).any() else sub
            _, near = ndi.distance_transform_edt(~ref, return_indices=True)
            val = src[tuple(near[d][idx[:, 0], idx[:, 1], idx[:, 2]] for d in range(3))]
        out["tumor"] = (v + off, fc, val)
    return out


# ── 진료 대기자 (의료진 계정별, 최대 5명) ──────────────────────────────
QUEUE = CACHE / "queue"


def queue_load(uid):
    f = QUEUE / f"{uid}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _queue_save(uid, q):
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / f"{uid}.json").write_text(json.dumps(q[:5], ensure_ascii=False), encoding="utf-8")


def queue_pick(uid, pid):
    """고른 환자를 1번 · 진료 중으로 (다른 진료 중 환자는 대기 중)."""
    q = [{"pid": x["pid"], "st": "wait" if x["st"] == "on" else x["st"]} for x in queue_load(uid) if x["pid"] != pid]
    _queue_save(uid, [{"pid": pid, "st": "on"}] + q)


def queue_status(uid, pid, st):
    """원을 눌렀을 때: 그 자리(순서 그대로)에서 상태만 바꿈. 진료 중은 한 명만 (다른 진료 중 환자는 대기 중)."""
    q = queue_load(uid)
    _queue_save(uid, [{"pid": x["pid"], "st": st if x["pid"] == pid else ("wait" if st == "on" and x["st"] == "on"
                                                                         else x["st"])} for x in q])


def queue_reset(uid):
    """로그인할 때: 진료 중 표시는 모두 대기 중으로 (선택한 환자 없음)."""
    q = queue_load(uid)
    if q:
        _queue_save(uid, [{"pid": x["pid"], "st": "wait" if x["st"] == "on" else x["st"]} for x in q])


def queue_remove(uid, pid, why="done"):
    """진료 완료(done) · 취소(cancel): 대기자 명단에서 빼고 기록만 남김 (cache/queue/<uid>_log.jsonl)."""
    import datetime as dt
    _queue_save(uid, [x for x in queue_load(uid) if x["pid"] != pid])
    with open(QUEUE / f"{uid}_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"pid": pid, "result": why, "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M")},
                           ensure_ascii=False) + "\n")


def queue_save(uid, q):
    """순서를 바꾼 대기자 명단 저장 (최대 5명)."""
    _queue_save(uid, q)
