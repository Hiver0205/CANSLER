"""
01_explore_metadata.py ─ 메타데이터·라벨·임상변수 점검
=====================================================
pCR 분포, 아형별 pCR률, 임상변수 결측, 내장 test 분할을 확인한다.
산출물 → Evaluate/01_label_summary.csv, 01_pcr_by_subtype.png

실행:  python 01_explore_metadata.py
"""
from __future__ import annotations
import sys
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")


def main() -> int:
    cfg.ensure_dirs()
    if not cfg.METADATA_CSV.exists():
        print(f"[!] metadata 없음: {cfg.METADATA_CSV}")
        return 1
    df = pd.read_csv(cfg.METADATA_CSV)
    print(f"[i] {df.shape[0]}명 × {df.shape[1]}변수")

    # pCR
    y = df[cfg.COL_PCR].astype(int)
    print(f"\n[pCR] {int(y.sum())}/{len(y)} = {y.mean():.1%} (비pCR {int((1-y).sum())})")

    # 아형별
    print("\n[아형별 pCR률]")
    rows = []
    for st, g in df.groupby(cfg.COL_SUBTYPE):
        lab = cfg.SUBTYPE_LABEL.get(st, st)
        rows.append({"subtype": lab, "n": len(g), "pcr": int(g[cfg.COL_PCR].sum()),
                     "pcr_rate": g[cfg.COL_PCR].mean()})
        print(f"  {lab:10s} n={len(g):4d}  pCR={int(g[cfg.COL_PCR].sum()):4d}  ({g[cfg.COL_PCR].mean():.1%})")
    summ = pd.DataFrame(rows).sort_values("pcr_rate", ascending=False)
    summ.loc[len(summ)] = ["ALL", len(df), int(y.sum()), y.mean()]
    summ.to_csv(cfg.EVAL_DIR / "01_label_summary.csv", index=False)

    # 임상변수 결측
    print("\n[임상변수 결측]")
    for c in cfg.CLINICAL_COLS:
        if c in df.columns:
            print(f"  {c:12s} 결측 {int(df[c].isna().sum()):3d} / 고유 {df[c].nunique()}")
        else:
            print(f"  {c:12s} [!] 컬럼 없음")

    # test 분할
    if cfg.COL_SPLIT in df.columns:
        print("\n[내장 분할 test]")
        print(df.groupby(cfg.COL_SPLIT)[cfg.COL_PCR].agg(["size", "mean"]).round(3).to_string())

    # 그림
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            pass
        plt.rcParams["axes.unicode_minus"] = False
        plot = summ[summ.subtype != "ALL"]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(plot.subtype, plot.pcr_rate * 100,
                      color=["#F58518", "#54A24B", "#4C78A8"][:len(plot)])
        ax.axhline(y.mean() * 100, ls="--", color="#E45756", lw=1.1, label=f"전체 {y.mean()*100:.1f}%")
        for b, r, n in zip(bars, plot.pcr_rate, plot.n):
            ax.text(b.get_x()+b.get_width()/2, r*100+0.6, f"{r*100:.1f}%\n(n={n})", ha="center", fontsize=9)
        ax.set_ylabel("pCR률 (%)"); ax.set_title("아형별 pCR률"); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(cfg.EVAL_DIR / "01_pcr_by_subtype.png", dpi=150)
        print(f"\n[✓] Evaluate/01_label_summary.csv, 01_pcr_by_subtype.png")
    except Exception as e:  # noqa: BLE001
        print(f"[i] 그림 생략: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
