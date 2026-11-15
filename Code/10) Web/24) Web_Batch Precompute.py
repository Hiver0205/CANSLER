"""
precompute.py ─ 워크리스트의 AI 예측을 미리 채우는 일괄 분석 (I-SPY2 자료)
==================================================================
기본: 내부 검증 환자 198명, 치료 전·3주 차 두 모드. 이미 분석한 환자는 건너뜀.
결과는 Web/cache/<모드>/<등록번호>/ (환자 단위 자료 — GitHub에 올리지 않음).

실행:  python Web/precompute.py [--all] [--mode pre|t3|both]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpr_core as core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="학습에 쓴 환자까지 982명 전체")
ap.add_argument("--mode", default="both", choices=["pre", "t3", "both"])
a = ap.parse_args()
P = core.patients()
P = P if a.all else P[P["구분"] == "검증"]
modes = ["pre", "t3"] if a.mode == "both" else [a.mode]
done = core.saved_probs()
t0 = time.time()
for mode in modes:
    for n, (_, p) in enumerate(P.iterrows(), 1):
        pid = p["등록번호"]
        if (pid, mode) in done or (mode == "t3" and not p["3주차"]):
            continue
        inp = core.demo_inputs(pid)
        if inp is None or (mode == "t3" and inp["w3"] is None):
            print(f"[{mode}] {pid} 영상 없음", flush=True); continue
        try:
            r = core.run(pid, mode, inp, core.to_clinical(p))
            print(f"[{mode}] {n}/{len(P)} {pid} {r['prob'] * 100:.0f}%  ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{mode}] {pid} 실패: {e}", flush=True)
print("완료", f"{time.time() - t0:.0f}s")
