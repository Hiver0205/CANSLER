"""
09_t1_download.py ─ 실험 3용: 치료 3주 차(T1) DCE 원본 + 분석 마스크 다운로드 (TCIA, CC BY 4.0)
==========================================================================================
같은 I-SPY2 임상시험·같은 환자의 T1 방문 영상. (BreastDCEDL은 T0만 변환 배포)

  대상 시리즈 (환자당 2개, StudyDesc가 *_MRI_T1 인 것):
    · "VOLSER: uni-lateral cropped: original DCE" (없으면 "bi-lateral: original DCE")
    · "VOLSER: uni-lateral cropped: Analysis Mask" (없으면 "bi-lateral: Analysis Mask")
  출처: NBIA 공개 API — getSeries(Collection=ISPY2 / ACRIN-6698) → getImage(SeriesInstanceUID)
  저장: Data/Interim/T1_dicom/<pid>/{dce,mask}/*.dcm   (zip은 검증·해제 후 삭제)
  이어받기: 해제 완료 표시(.done) 가 있는 시리즈는 건너뜀 → 중단돼도 재실행하면 이어서 진행
  로그:  Evaluate/09_t1_download_log.csv, 09_t1_manifest.csv

실행:  python 09_t1_download.py --workers 10        (--limit N 으로 시험)
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
import pandas as pd

cfg = import_module("00_config")
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
OUT = cfg.DATA_INTERIM / "T1_dicom"
VISIT = "T1"


def _get(url, timeout=600):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def build_manifest():
    s = pd.concat([pd.DataFrame(json.loads(_get(f"{API}/getSeries?Collection={c}&format=json")))
                   for c in ("ISPY2", "ACRIN-6698")], ignore_index=True)
    s = s[s["StudyDesc"].str.endswith(f"_MRI_{VISIT}")]
    ours = set(pd.read_csv(cfg.METADATA_CSV)[cfg.COL_PID].astype(str))
    s = s[s["PatientID"].isin(ours)]
    rows = []
    for kind, key in (("dce", "original DCE"), ("mask", "Analysis Mask")):
        uni = s[s.SeriesDescription.str.contains(f"uni-lateral cropped: {key}", regex=False)]
        bi = s[s.SeriesDescription.str.contains(f"bi-lateral: {key}", regex=False)]
        bi = bi[~bi.PatientID.isin(uni.PatientID)]
        sel = pd.concat([uni, bi]).sort_values("FileSize", ascending=False).drop_duplicates("PatientID")
        for _, r in sel.iterrows():
            rows.append({"pid": r.PatientID, "kind": kind, "uid": r.SeriesInstanceUID,
                         "desc": r.SeriesDescription, "size": int(r.FileSize), "n_img": int(r.ImageCount)})
    m = pd.DataFrame(rows)
    both = m.groupby("pid").kind.nunique()
    m = m[m.pid.isin(both[both == 2].index)]            # DCE·마스크 둘 다 있는 환자만
    return m


def fetch(row, retries=5):
    dst = OUT / row["pid"] / row["kind"]; done = dst / ".done"
    if done.exists():
        return row["uid"], "skip", 0, 0.0
    t0 = time.time()
    for k in range(retries):
        try:
            data = _get(f"{API}/getImage?SeriesInstanceUID={row['uid']}")
            z = zipfile.ZipFile(io.BytesIO(data))
            bad = z.testzip()
            if bad:
                raise IOError(f"손상 {bad}")
            dcm = [n for n in z.namelist() if n.lower().endswith(".dcm")]
            if len(dcm) < row["n_img"]:
                raise IOError(f"이미지 부족 {len(dcm)}/{row['n_img']}")
            dst.mkdir(parents=True, exist_ok=True)
            for n in dcm:
                (dst / n.split("/")[-1]).write_bytes(z.read(n))
            done.write_text(row["uid"])
            return row["uid"], "OK", len(data), time.time() - t0
        except Exception as e:  # noqa: BLE001
            err = str(e); time.sleep(10 * (k + 1))
    return row["uid"], f"실패:{err}", 0, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="환자 수 제한 (시험용)")
    args = ap.parse_args()
    cfg.ensure_dirs(); OUT.mkdir(parents=True, exist_ok=True)
    mpath = cfg.EVAL_DIR / "09_t1_manifest.csv"
    m = pd.read_csv(mpath) if mpath.exists() else build_manifest()
    m.to_csv(mpath, index=False)
    if args.limit:
        m = m[m.pid.isin(m.pid.unique()[: args.limit])]
    print(f"[i] T1 대상 환자 {m.pid.nunique()}명 · 시리즈 {len(m)}개 · 원본 {m['size'].sum()/1e9:.1f} GB "
          f"· 동시 {args.workers}개", flush=True)
    log, got, t0 = [], 0, time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, r): r for r in m.to_dict("records")}
        for i, f in enumerate(as_completed(futs), 1):
            uid, st, nbytes, sec = f.result(); r = futs[f]
            got += nbytes
            log.append({"pid": r["pid"], "kind": r["kind"], "uid": uid, "status": st,
                        "mb": nbytes / 1e6, "sec": round(sec, 1)})
            if i % 20 == 0 or st.startswith("실패"):
                el = time.time() - t0
                print(f"  [{i}/{len(m)}] {r['pid']} {r['kind']} {st} · 누적 {got/1e9:.1f} GB · "
                      f"{got/1e6/max(el,1):.1f} MB/s · 경과 {el/60:.0f}분", flush=True)
            if i % 50 == 0:
                pd.DataFrame(log).to_csv(cfg.EVAL_DIR / "09_t1_download_log.csv", index=False)
    lg = pd.DataFrame(log); lg.to_csv(cfg.EVAL_DIR / "09_t1_download_log.csv", index=False)
    print("\n" + lg.status.str.split(":").str[0].value_counts().to_string())
    ok = lg[lg.status.isin(["OK", "skip"])].groupby("pid").kind.nunique()
    print(f"[✓] DCE+마스크 모두 확보 {int((ok == 2).sum())}명 · {(time.time()-t0)/60:.0f}분 · {OUT}")
    print("    실패 시리즈는 재실행하면 이어받음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
