"""
run_demo.py ─ 가상 데이터로 BPR 전체 흐름 실행 (실제 환자 자료 없이)
==================================================================
1) 학습된 모델 확인 — 로지스틱 회귀·habitat 모델은 저장소 Models/에 포함,
   CNN 가중치(약 300 MB)는 GitHub Release에서 내려받음 (--no-cnn이면 CNN 없이 로지스틱 회귀만)
2) 가상 유방 DCE-MRI 생성 (demo/make_synthetic.py, I-SPY2와 같은 파일 형식)
3) 실제 추론 코드(Code/10) Web/21)·22))로 환자마다 전처리 → 특징 추출 → 예측, 결과 표 출력·저장
4) --web: 같은 가상 환자로 웹앱(Streamlit) 실행 — 데모 계정으로 로그인

  python demo/run_demo.py              # 1~3 (CPU에서 환자당 1~3분)
  python demo/run_demo.py --web        # 1~3 후 웹앱 → http://localhost:8501
  python demo/run_demo.py --no-cnn     # CNN 가중치 내려받기 없이 (로지스틱 회귀만)
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CNN_URL = "https://github.com/Hiver0205/CANSLER/releases/download/models-v1/bpr-cnn-weights-v1.zip"
CNN_FILES = ["cnn_t6_tta_s3/seed0.pt", "cnn_t6_tta_s3/seed1.pt", "cnn_t6_tta_s3/seed2.pt", "cnn_t6_3d/seed0.pt",
             "cnn_t6_T1/seed0.pt"]
DEMO_USER = ("demo", "demo1234", "246810")          # 웹 데모 계정 (아이디, 비밀번호, 간편 비밀번호)


def say(msg):
    print(msg, flush=True)


def get_cnn(models):
    if all((models / f).exists() for f in CNN_FILES):
        return True
    zp = models / "bpr-cnn-weights-v1.zip"
    say(f"[i] CNN 가중치 내려받는 중 (약 300 MB): {CNN_URL}")
    try:
        shown = [-1]

        def hook(n, bs, total):                            # 10%마다 한 줄
            if total > 0 and int(min(n * bs / total, 1) * 10) > shown[0]:
                shown[0] = int(min(n * bs / total, 1) * 10); say(f"    {shown[0] * 10}%")
        urllib.request.urlretrieve(CNN_URL, zp, hook)
        with zipfile.ZipFile(zp) as z:
            z.extractall(models)
        zp.unlink()
        return all((models / f).exists() for f in CNN_FILES)
    except Exception as e:  # noqa: BLE001
        say(f"[!] CNN 가중치를 받지 못했습니다 ({e}) → 로지스틱 회귀만으로 진행")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="가상 환자 수 (최대 6)")
    ap.add_argument("--no-cnn", action="store_true", help="CNN 없이 로지스틱 회귀 앙상블만")
    ap.add_argument("--web", action="store_true", help="분석 후 웹앱 실행")
    ap.add_argument("--port", type=int, default=8501)
    a = ap.parse_args()
    t0 = time.time()
    models = REPO / "Models"
    for f in ("ens_main_lr.joblib", "ens_t3_lr.joblib", "habitat_model.npz"):
        if not (models / f).exists():
            sys.exit(f"[!] {models / f} 없음 — 저장소를 다시 받아 주세요.")
    cnn = False if a.no_cnn else get_cnn(models)

    # 데모 데이터 루트 = 저장소 (Data/·Evaluate/는 .gitignore). 모든 코드가 BREASTCANCER_ROOT를 기준으로 읽음.
    os.environ["BREASTCANCER_ROOT"] = str(REPO)
    os.environ["BPR_DEMO"] = "1"
    feat = REPO / "Data" / "Features"; feat.mkdir(parents=True, exist_ok=True)
    shutil.copy2(models / "habitat_model.npz", feat / "habitat_model.npz")
    meta = REPO / "Data" / "Original" / "BreastDCEDL_ISPY2_metadata.csv"
    if not meta.exists():
        say("[1/3] 가상 유방 MRI 생성")
        subprocess.run([sys.executable, str(REPO / "demo" / "make_synthetic.py"), "--root", str(REPO),
                        "--n", str(a.n)], check=True)
    else:
        say(f"[1/3] 가상 유방 MRI 있음: {meta.parent.parent}")

    sys.path.insert(0, str(REPO / "Code"))
    import bpr_modules  # noqa: F401,E402 — 원래 모듈 이름으로 저장소 파일을 찾게 함
    import bpr_core as core  # noqa: E402
    from importlib import import_module  # noqa: E402
    inf = import_module("21_infer")
    if not cnn:
        inf.MAIN_CNN = ()                                  # 로지스틱 회귀 14개만 평균
    say(f"[2/3] 예측 모델: 로지스틱 회귀 14개{' + CNN 2개' if cnn else ' (CNN 없음)'}")

    say("[3/3] 환자별 분석 (전처리 → 동역학·주변부·시간곡선 → 질감(PyRadiomics) → habitat → 예측)")
    P = core.patients()
    rows = []
    for _, p in P.iterrows():
        pid = p["등록번호"]; t1 = time.time()
        inp = core.demo_inputs(pid)
        r = core.run(pid, "pre", inp, core.to_clinical(p))
        e = inf.explain(r["features"]).sort_values(key=abs, ascending=False)
        rows.append({"환자": pid, "아형": p["아형"], "특징 수": len(r["features"]),
                     "pCR 확률(%)": round(r["prob"] * 100, 1),
                     "가장 큰 요인": f"{e.index[0]} ({'+' if e.iloc[0] > 0 else '−'})", "시간(s)": round(time.time() - t1)})
        say(f"    {pid} {p['아형']:<9} pCR 확률 {r['prob'] * 100:5.1f}%  특징 {len(r['features'])}개  "
            f"({time.time() - t1:.0f}s)")
    import pandas as pd  # noqa: E402
    out = REPO / "Evaluate"; out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / "demo_results.csv", index=False, encoding="utf-8-sig")
    say("\n" + df.to_string(index=False))
    say(f"\n[✓] 완료 {time.time() - t0:.0f}s · 결과 {out / 'demo_results.csv'}")
    say("    (가상 영상이므로 확률 값 자체에는 임상적 의미가 없습니다)")

    if a.web:
        auth = import_module("auth")
        uid, pw, pin = DEMO_USER
        if not auth.check_pin(uid, pin):
            auth.create_user(uid, "데모", "BPR 데모 병원", "유방외과", pw, pin)
        web = Path(bpr_modules.WEB_APP)
        say(f"\n[웹] http://localhost:{a.port}  ·  아이디 {uid}  ·  비밀번호 {pw}  ·  간편 비밀번호 {pin}")
        os.chdir(web.parent)                               # 화면 설정(.streamlit/config.toml)
        from streamlit.web import cli as stcli  # noqa: E402
        sys.argv = ["streamlit", "run", str(web), "--server.port", str(a.port), "--server.headless", "true",
                    "--browser.gatherUsageStats", "false"]
        sys.exit(stcli.main())


if __name__ == "__main__":
    main()
