"""
04c_cnn_finetune.py ─ G4: CNN 미세조정 (GPU) + 누수 없는 결합 평가
==================================================================
고정 CNN 특징(04b)과 달리 CNN 자체를 pCR 라벨로 학습한다. 라벨을 쓰므로 누수 관리가 핵심:

  · 바깥 분할 = 06과 동일 (StratifiedGroupKFold(5, shuffle, random_state=r), 개발셋만, 홀드아웃 미사용)
  · 바깥 테스트 환자의 CNN 점수 = 바깥 학습분 전체로 학습한 CNN의 예측
  · 바깥 학습 환자의 CNN 점수   = 바깥 학습분 안의 안쪽 3-fold OOF 예측 (자기 자신을 본 CNN 점수 금지)
  · 그 점수를 특징(cnn__score)으로 06의 build()와 같은 파이프라인에 넣어 임상·영상과 결합

입력: Data/Interim/<pid>/pe_early.nii.gz, pe.nii.gz, ser.nii.gz, mask.nii.gz (03), analysis_table.csv (05)
  슬라이스: 종양 면적 ≥ 최대의 10%인 축상 슬라이스 중 최대 MAX_SLICES장(균등 간격)
  환자 점수 = 슬라이스 logit 평균
모델: ImageNet ResNet18 전체 미세조정, AdamW, BCE(pos_weight), 증강(뒤집기·90°회전·밝기·크롭)
출력: Evaluate/04c_cnn_oof_long.csv (pid, repeat, fold, model, prob), 04c_cnn_summary.csv

실행:  python 04c_cnn_finetune.py --smoke          # 1반복·1폴드·2에폭 동작 확인
       python 04c_cnn_finetune.py --repeats 5      # 본 실행
"""
from __future__ import annotations
import argparse
import sys
import time
from importlib import import_module
import numpy as np
import pandas as pd

cfg = import_module("00_config")
tr6 = import_module("06_train_cv")      # build(), cols_of(), 접두어 — 06과 동일 파이프라인

PE_MAX, SER_MAX, MIN_AREA_FRAC = 400.0, 3.0, 0.10
CH_MODE = "pe3"   # "pe3" = [초기PE, 후기PE, SER] / "t6" = 고정 6시점 곡선 (J2, --channels로 설정)
DIM3, DEPTH3, SIZE3 = False, 32, 64   # J3: 3D CNN (--dim3), 볼륨 크기 (깊이, 평면)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# ── 데이터: 환자별 슬라이스를 한 번만 만들어 메모리에 (fp16) ──────────────
def load_all(pids, size, max_slices, sitk, torch, root=None, offset=0):
    import torch.nn.functional as F
    xs, owner = [], []
    root = root or cfg.DATA_INTERIM
    for i0, pid in enumerate(pids):
        i = i0 + offset
        d = root / pid
        keys = ("pe_early", "pe", "ser") if CH_MODE == "pe3" else tuple(f"pe_t{c}" for c in range(6))
        if not (d / "mask.nii.gz").exists():   # 지도 없음(예: T1 마스크 비어 있음) → 빈 입력으로 코호트·분할 유지
            c_in = len(keys)
            shp = (1, c_in, DEPTH3, SIZE3, SIZE3) if DIM3 else (1, c_in, size, size)
            xs.append(torch.zeros(shp, dtype=torch.float16)); owner.append(i)
            print(f"  [!] {pid}: 지도 없음 → 빈 입력", flush=True)
            continue
        a = {k: sitk.GetArrayFromImage(sitk.ReadImage(str(d / f"{k}.nii.gz"))) for k in keys + ("mask",)}
        m = a["mask"] > 0
        area = m.sum(axis=(0, 1))
        ks = np.where(area >= MIN_AREA_FRAC * area.max())[0]
        if len(ks) > max_slices:
            ks = ks[np.linspace(0, len(ks) - 1, max_slices).round().astype(int)]
        if CH_MODE == "pe3":
            ch = np.stack([np.clip(a["pe_early"], 0, PE_MAX) / PE_MAX, np.clip(a["pe"], 0, PE_MAX) / PE_MAX,
                           np.clip(a["ser"], 0, SER_MAX) / SER_MAX], 0).astype(np.float32)
        else:   # J2: 고정 6시점 증강 곡선 (03g)
            ch = np.stack([np.clip(a[k], 0, PE_MAX) / PE_MAX for k in keys], 0).astype(np.float32)
        if DIM3:   # J3: 종양 crop 전체를 3D 볼륨 하나로 (C, slice, A, B) → 정사각 패딩 → (D3, S3, S3)
            v = np.transpose(ch, (0, 3, 1, 2)); c_in, dd, h, w = v.shape; s = max(h, w)
            pad = np.zeros((c_in, dd, s, s), np.float32)
            pad[:, :, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = v
            t = F.interpolate(torch.from_numpy(pad)[None], size=(DEPTH3, SIZE3, SIZE3),
                              mode="trilinear", align_corners=False)
            xs.append(t.half()); owner.append(i)
            if (i0 + 1) % 200 == 0:
                print(f"  볼륨 적재 {i0+1}/{len(pids)}", flush=True)
            continue
        sl = np.transpose(ch[:, :, :, ks], (3, 0, 1, 2))
        n, c_in, h, w = sl.shape; s = max(h, w)
        pad = np.zeros((n, c_in, s, s), np.float32)
        pad[:, :, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = sl
        t = F.interpolate(torch.from_numpy(pad), size=(size, size), mode="bilinear", align_corners=False)
        xs.append(t.half()); owner += [i] * n
        if (i0 + 1) % 200 == 0:
            print(f"  슬라이스 적재 {i0+1}/{len(pids)}", flush=True)
    return torch.cat(xs), np.asarray(owner)


def augment(x, torch):
    for ax in ((-1, -2, -3) if x.dim() == 5 else (-1, -2)):    # 3D면 깊이 축도 뒤집기
        if torch.rand(1).item() < 0.5:
            x = x.flip(ax)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,)).item()), dims=(-2, -1))
    scale = 1.0 + 0.2 * (torch.rand((x.shape[0],) + (1,) * (x.dim() - 1), device=x.device) - 0.5)  # 밝기 ±10%
    return (x * scale).clamp(0, 1)


def make_model(torch):
    import torchvision
    n_in = 3 if CH_MODE == "pe3" else 6
    if DIM3:   # J3: Kinetics 사전학습 3D ResNet18, stem을 n_in 채널로 확장
        m = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.KINETICS400_V1)
        c0 = m.stem[0]
        if n_in != 3:
            w = c0.weight.data.mean(1, keepdim=True).repeat(1, n_in, 1, 1, 1) * (3.0 / n_in)
            m.stem[0] = torch.nn.Conv3d(n_in, 64, kernel_size=c0.kernel_size, stride=c0.stride,
                                        padding=c0.padding, bias=False)
            m.stem[0].weight.data = w
        m.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(m.fc.in_features, 1))
        return m
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    if n_in != 3:   # 첫 층을 n_in 채널로: RGB 가중치 평균을 복제하고 크기 보존(×3/n_in)
        w = m.conv1.weight.data.mean(1, keepdim=True).repeat(1, n_in, 1, 1) * (3.0 / n_in)
        m.conv1 = torch.nn.Conv2d(n_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.conv1.weight.data = w
    m.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(m.fc.in_features, 1))
    return m


def train_predict(X, owner, y_pat, train_pats, pred_pats, args, torch, seed):
    """train_pats 환자 슬라이스로 학습 → pred_pats 환자별 점수(슬라이스 logit 평균)."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    mean, std = torch.tensor(MEAN, device=dev), torch.tensor(STD, device=dev)
    tr_idx = np.where(np.isin(owner, train_pats))[0]
    ys = torch.tensor(y_pat[owner[tr_idx]], dtype=torch.float32)
    pos = float(ys.mean())
    model = make_model(torch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps = args.epochs * int(np.ceil(len(tr_idx) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for ep in range(args.epochs):
        perm = np.random.permutation(len(tr_idx))
        for b in range(0, len(perm), args.batch):
            j = tr_idx[perm[b:b + args.batch]]
            xb = augment(X[j].to(dev).float(), torch)
            yb = ys[perm[b:b + args.batch]].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model((xb - mean) / std).squeeze(1), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sched.step()
    model.eval(); out = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for p in pred_pats:
            j = np.where(owner == p)[0]
            xb = X[j].to(dev).float()
            if getattr(args, "tta", False):   # K3: 8방향(회전 4 × 뒤집기 2) 예측 평균
                views = [torch.rot90(v, k, dims=(-2, -1)) for v in (xb, xb.flip(-1)) for k in range(4)]
                lg = torch.stack([model((v - mean) / std).squeeze(1).float() for v in views]).mean(0)
            else:
                lg = model((xb - mean) / std).squeeze(1).float()
            out[p] = float(lg.mean())
    del model; torch.cuda.empty_cache()
    return out


def train_predict_multi(X, owner, y_pat, train_pats, pred_pats, args, torch, seed):
    """K3: 시드 여러 개로 학습한 모델의 logit 평균 (학습 잡음 감소)."""
    outs = [train_predict(X, owner, y_pat, train_pats, pred_pats, args, torch, seed + 7919 * s)
            for s in range(getattr(args, "seeds", 1))]
    return {p: float(np.mean([o[p] for o in outs])) for p in outs[0]}


def final_holdout(df, args, sitk, torch):
    """개발셋 전체로 CNN 학습(시드 평균) → 홀드아웃 환자 점수(logit). 홀드아웃 라벨은 학습에 쓰지 않음."""
    dev = df[~df["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)
    hold = df[df["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)
    pids = list(dev[cfg.COL_PID]) + list(hold[cfg.COL_PID])
    root = (cfg.DATA_INTERIM / args.root) if args.root else None
    X, owner = load_all(pids, args.size, args.max_slices, sitk, torch, root=root)
    y_all = np.concatenate([dev["pCR"].to_numpy().astype(int), np.zeros(len(hold), int)])  # 홀드아웃 자리표시(미사용)
    tr = np.arange(len(dev)); te = np.arange(len(dev), len(pids))
    print(f"[i] 최종 학습: 개발 {len(dev)}명 → 홀드아웃 {len(hold)}명 예측 · 시드 {args.seeds} · TTA {args.tta}", flush=True)
    s = train_predict_multi(X, owner, y_all, tr, te, args, torch, seed=cfg.RANDOM_STATE)
    out = pd.DataFrame({cfg.COL_PID: [pids[i] for i in te], "cnn_logit": [s[i] for i in te]})
    sfx = (("_t6" if CH_MODE == "t6" else "") + ("_3d" if DIM3 else "") + ("_tta" if args.tta else "")
           + (f"_s{args.seeds}" if args.seeds > 1 else "") + (f"_{args.root}" if args.root else ""))
    out.to_csv(cfg.EVAL_DIR / f"04c_final{sfx}.csv", index=False)
    print(f"[✓] Evaluate/04c_final{sfx}.csv ({len(out)}명)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--max-slices", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--external", action="store_true",
                    help="I-SPY1·Duke(04d)를 모든 CNN 학습에 추가 (평가는 I-SPY2 개발셋만)")
    ap.add_argument("--channels", choices=["pe3", "t6"], default="pe3",
                    help="pe3 = 3채널(기본) / t6 = 고정 6시점 곡선 (J2, 03g 필요)")
    ap.add_argument("--dim3", action="store_true", help="J3: 3D CNN (r3d_18, 환자당 볼륨 1개)")
    ap.add_argument("--tta", action="store_true", help="K3: 예측 시 8방향 증강 평균")
    ap.add_argument("--seeds", type=int, default=1, help="K3: 시드 수 (모델 평균)")
    ap.add_argument("--root", default=None, help="L2: 지도 상위 폴더 (예: T1 → Data/Interim/T1). "
                                                  "지정 시 개발셋을 T1 특징이 있는 환자로 제한 → 08과 같은 분할")
    ap.add_argument("--final", action="store_true",
                    help="최종: 개발셋 전체로 학습 → 홀드아웃 1회 예측 (CV 없음). Evaluate/04c_final*.csv")
    args = ap.parse_args()
    global CH_MODE, MEAN, STD, DIM3
    CH_MODE, DIM3 = args.channels, args.dim3
    if CH_MODE == "t6":
        if args.external:
            raise SystemExit("[!] t6는 외부 데이터 미지원 (Duke는 시점 3개)")
        MEAN = np.full((1, 6, 1, 1), 0.45, np.float32); STD = np.full((1, 6, 1, 1), 0.225, np.float32)
    if DIM3:   # 5D 정규화 (Kinetics 통계 근사), 배치 축소
        c = 3 if CH_MODE == "pe3" else 6
        MEAN = np.full((1, c, 1, 1, 1), 0.40, np.float32); STD = np.full((1, c, 1, 1, 1), 0.22, np.float32)
        if args.batch == 64:
            args.batch = 16
    if args.smoke:
        args.repeats, args.epochs = 1, 2
    import SimpleITK as sitk  # noqa: N813
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    assert torch.cuda.is_available(), "GPU 필요"
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(cfg.DATA_FEATURES / "analysis_table.csv"); df[cfg.COL_PID] = df[cfg.COL_PID].astype(str)
    if args.root:   # L2: 08 --source full 과 같은 코호트·순서 (analysis_table ⋈ t1_features, 개발셋)
        t1p = set(pd.read_csv(cfg.DATA_FEATURES / "t1_features.csv")[cfg.COL_PID].astype(str))
        df = df[df[cfg.COL_PID].isin(t1p)]
    if args.final:
        return final_holdout(df, args, sitk, torch)
    dev = df[~df["split"].isin(cfg.HOLDOUT_SPLITS)].reset_index(drop=True)   # 홀드아웃 미사용
    y, groups = dev["pCR"].to_numpy().astype(int), dev[cfg.COL_PID].to_numpy()
    print(f"[i] 개발 {len(dev)}명 · 반복 {args.repeats} · 에폭 {args.epochs} · {args.size}px · "
          f"슬라이스≤{args.max_slices} · GPU {torch.cuda.get_device_name(0)}")
    t0 = time.time()
    X, owner = load_all(list(groups), args.size, args.max_slices, sitk, torch,
                        root=(cfg.DATA_INTERIM / args.root) if args.root else None)
    y_all, ext_idx = y.copy(), np.array([], dtype=int)
    if args.external:   # 외부 환자: 인덱스 len(dev)부터, 학습에만 사용
        ext = pd.read_csv(cfg.DATA_FEATURES / "ext_labels.csv")
        Xe, oe = load_all(list(ext["pid"]), args.size, args.max_slices, sitk, torch,
                          root=cfg.DATA_INTERIM / "ext", offset=len(dev))
        X, owner = torch.cat([X, Xe]), np.concatenate([owner, oe])
        y_all = np.concatenate([y, ext["pCR"].to_numpy().astype(int)])
        ext_idx = np.arange(len(dev), len(dev) + len(ext))
        print(f"[i] 외부 학습 환자 {len(ext)}명 추가 (" +
              " · ".join(f"{s} {n}" for s, n in ext.source.value_counts().items()) + ")")
    print(f"[i] 슬라이스 {len(owner)}장 적재 ({X.element_size()*X.nelement()/1e9:.2f} GB) · {time.time()-t0:.0f}s")

    clin, vol, kin = tr6.cols_of(dev, ["clin__"]), tr6.cols_of(dev, ["vol__"]), tr6.cols_of(dev, [tr6.KIN])
    tch = tr6.cols_of(dev, [tr6.TC, tr6.HAB])          # 현재 최고 단일 조합: 전체 곡선 + habitat
    specs = {   # 이름 → (항상 유지 열, L1 대상 열)  — cnn__score는 항상 유지 쪽
        "clinical":         (clin, []),
        "combined_vol_kin": (clin + vol, kin),
        "cvk_cnn":          (clin + vol + ["cnn__score"], kin),
        "img_vol_kin":      (vol, kin),
        "img_vol_kin_cnn":  (vol + ["cnn__score"], kin),
        "cv_tc_hab":        (clin + vol, tch),
        "cv_tc_hab_cnn":    (clin + vol + ["cnn__score"], tch),
        "img_vol_tc_hab":   (vol, tch),
        "img_vol_tc_hab_cnn": (vol + ["cnn__score"], tch),
    }
    rows, frows = [], []
    for r in range(args.repeats):
        cv = StratifiedGroupKFold(cfg.CV_FOLDS, shuffle=True, random_state=r)   # 06과 동일
        for f, (tr, te) in enumerate(cv.split(dev, y, groups)):
            if args.smoke and f > 0:
                break
            ts = time.time()
            score = np.full(len(dev), np.nan)
            # 바깥 학습분: 안쪽 OOF 점수
            inner = StratifiedKFold(args.inner, shuffle=True, random_state=1000 * r + f)
            for itr, ite in inner.split(tr, y[tr]):
                s = train_predict_multi(X, owner, y_all, np.concatenate([tr[itr], ext_idx]), tr[ite], args,
                                        torch, seed=10_000 * r + 100 * f + len(itr))
                for p, v in s.items():
                    score[p] = v
            # 바깥 테스트분: 바깥 학습분 전체(+외부)로 학습한 CNN
            s = train_predict_multi(X, owner, y_all, np.concatenate([tr, ext_idx]), te, args, torch,
                                    seed=10_000 * r + 100 * f)
            for p, v in s.items():
                score[p] = v
            d = dev.copy(); d["cnn__score"] = score
            res = {"cnn_only": score[te]}
            for name, (base, im) in specs.items():
                est = tr6.build(base, im, inner_seed=1000 * r + f)
                est.fit(d.iloc[tr][base + im], y[tr])
                res[name] = est.predict_proba(d.iloc[te][base + im])[:, 1]
            for name, p in res.items():
                rows += [{"pid": pid, "repeat": r, "fold": f, "model": name, "prob": float(v)}
                         for pid, v in zip(groups[te], p)]
                frows.append({"repeat": r, "fold": f, "model": name, "n_train": len(tr), "n_test": len(te),
                              "auc": roc_auc_score(y[te], p)})
            a = {k: roc_auc_score(y[te], v) for k, v in res.items()}
            print(f"  반복 {r+1} 폴드 {f+1}: CNN단독 {a['cnn_only']:.3f} · 임상+부피+동역학 {a['combined_vol_kin']:.3f}"
                  f" → +CNN {a['cvk_cnn']:.3f} · {time.time()-ts:.0f}s", flush=True)

    long = pd.DataFrame(rows); fa = pd.DataFrame(frows)
    sfx = (("_ext" if args.external else "") + ("_t6" if CH_MODE == "t6" else "") + ("_3d" if DIM3 else "")
           + ("_tta" if args.tta else "") + (f"_s{args.seeds}" if args.seeds > 1 else "")
           + (f"_{args.root}" if args.root else "")
           + ("_smoke" if args.smoke else ""))
    long.to_csv(cfg.EVAL_DIR / f"04c_cnn_oof_long{sfx}.csv", index=False)
    fa.to_csv(cfg.EVAL_DIR / f"04c_cnn_fold_auc{sfx}.csv", index=False)

    # 요약: 반복별 pooled AUC, ΔAUC (corrected t-test는 07 함수 재사용)
    ev = import_module("07_evaluate")
    ymap = dict(zip(groups, y)); sub = dict(zip(groups, dev["subtype"]))
    per = long.groupby(["model", "repeat"]).apply(
        lambda g: roc_auc_score(g.pid.map(ymap), g.prob), include_groups=False).unstack(0)
    print("\n=== G4 결과 (개발셋, 반복별 pooled AUC 평균) ===")
    print(per.mean().round(3).to_string())
    wide = fa.pivot_table(index=["repeat", "fold"], columns="model", values="auc")
    out = []
    for a, b in [("combined_vol_kin", "cvk_cnn"), ("img_vol_kin", "img_vol_kin_cnn"), ("clinical", "cvk_cnn"),
                 ("cv_tc_hab", "cv_tc_hab_cnn"), ("img_vol_tc_hab", "img_vol_tc_hab_cnn")]:
        dr = (per[b] - per[a]).to_numpy()
        pv = ev.corrected_ttest((wide[b] - wide[a]).to_numpy(), fa.n_train.mean(), fa.n_test.mean()) \
            if len(wide) > 1 else float("nan")
        out.append({"from": a, "to": b, "delta_mean": dr.mean(), "frac_pos": (dr > 0).mean(), "p": pv})
        print(f"  {a} → {b}: ΔAUC {dr.mean():+.3f} · 양수 반복 {(dr > 0).mean():.0%} · p={pv:.4f}")
    print("\n[아형별 AUC (반복 평균)]")
    long["subtype"] = long.pid.map(sub); long["y"] = long.pid.map(ymap)
    for name in ("cnn_only", "img_vol_kin", "img_vol_kin_cnn", "combined_vol_kin", "cvk_cnn",
                 "cv_tc_hab", "cv_tc_hab_cnn", "img_vol_tc_hab", "img_vol_tc_hab_cnn"):
        g = long[long.model == name]
        txt = " · ".join(f"{cfg.SUBTYPE_LABEL.get(st, st)} "
                         f"{np.mean([roc_auc_score(x.y, x.prob) for _, x in gs.groupby('repeat') if x.y.nunique() == 2]):.3f}"
                         for st, gs in g.groupby("subtype"))
        print(f"  {name:17s} {txt}")
    pd.DataFrame(out).to_csv(cfg.EVAL_DIR / f"04c_cnn_summary{sfx}.csv", index=False)
    print(f"\n[✓] 총 {time.time()-t0:.0f}s · Evaluate/04c_cnn_*{sfx}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
