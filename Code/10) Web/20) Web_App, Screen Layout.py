"""
BPR 웹 도구 (Streamlit) ─ I-SPY2 MRI 기반 pCR 예측 판독 화면
==================================================================
왼쪽 패널(화살표로 접고 펼침): 선택한 환자 정보 → 환자 목록(등록번호·환자명 검색, 아형 필터) → 분석하기
가운데: 유방 MRI 2D(축상·시상·관상·MIP, 조영 시점 막대)·3D(종양 표면 + 주변 조직) + 정량 수치
오른쪽: pCR 가능성(% + 막대) · 판단 근거 · 의사 소견 · 결과 CSV 저장
환자 목록은 I-SPY2 메타데이터 (기본: 학습에 쓰지 않은 내부 검증 환자 198명, split 1·2).
영상은 I-SPY2 공개 자료에서 불러오거나 TCIA I-SPY2 검사 zip(조영 MRI + Analysis Mask)을 올림.
추론은 Code1/21_infer.py, 공통 로직은 Web/bpr_core.py. I-SPY2 자료로 분석한 결과는 Web/cache/에 남아
다시 열 때 바로 보임 (Web/precompute.py로 미리 일괄 분석 가능).

실행:  Web/run_web.bat   (BreastCancer conda 환경)
"""
from __future__ import annotations
import io
import sys
import tempfile
import zipfile
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bpr_core as core  # noqa: E402  (torch를 먼저 불러옴 — DLL 충돌 방지)
import auth  # noqa: E402  (의료진 계정)
import time  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

cfg, inf, dio = core.cfg, import_module("21_infer"), import_module("23_dicom_io")
WEB = core.WEB
CRIMSON, CORAL, PINK, INK, MUTED = "#b40e26", "#f26375", "#fdc6cb", "#1f1e22", "#6b6570"
SUBTYPE_RATE = core.SUBTYPE_RATE
AUC = {"pre": ("치료 전 MRI로 예측", 0.757), "t3": ("치료 전 + 3주 차 MRI로 예측", 0.816)}
MODES = {"치료 전 MRI": "pre", "3주 차 MRI": "t3"}     # 가운데 패널 선택 → 보여 줄 영상 + 예측 모델
CHIP = {"HER2+": ("#fbf1f3", CRIMSON), "TNBC": ("#efeaf6", "#5b3a8c"), "HR+HER2−": ("#eaf2f7", "#22577a")}
MUTED_L = MUTED

st.set_page_config(page_title="BPR · Breast pCR Response", layout="wide", initial_sidebar_state="expanded")

ss = st.session_state
for _k, _v in (("uid", None), ("tok", ""), ("boot_checked", False), ("print_nonce", 0), ("logout_note", ""),
               ("dev_tok", ""), ("dev_clear", False), ("full_login", False), ("acct_ok", 0)):
    ss.setdefault(_k, _v)
if ss.uid and auth.session_user(ss.tok) is None:          # 30분 동안 쓰지 않으면 자동 로그아웃
    ss.uid, ss.tok, ss.logout_note = None, "", "30분 동안 사용하지 않아 자동으로 로그아웃되었습니다."
    ss.sel = None
ME = auth.user(ss.uid) if ss.uid else {}
DARK = (ME.get("theme") == "dark") if ss.uid else (st.context.theme.type == "dark")
if DARK:
    CRIMSON, INK, MUTED = "#e0405a", "#ece8ea", "#a59ea4"
    CARD, LINE, SOFT, HOVER, SEL = "#1c191d", "#352f35", "#2a252b", "#241f25", "#33202a"
    PILL, PILLTX, POSBG, POSBD, NEGBG, NEGBD = "#2d282e", "#d6d0d4", "#3a1f27", "#7a3341", "#262227", "#4a434a"
    CHIP = {"HER2+": ("#3a1f27", "#ff8296"), "TNBC": ("#2c2438", "#c4a8ee"), "HR+HER2−": ("#1d2c37", "#8cc3e8")}
else:
    CARD, LINE, SOFT, HOVER, SEL = "#fff", "#e7e2e4", "#f3eef0", "#fff8f9", "#fbf1f3"
    PILL, PILLTX, POSBG, POSBD, NEGBG, NEGBD = "#f3f0f1", "#4a4550", "#fff5f6", "#f2b8c0", "#faf9fa", "#dcd6d9"
st.markdown(f"""<style>
.block-container {{padding-top: 2.2rem; max-width: 1640px}}
section[data-testid="stSidebar"] {{width: 380px !important}}
section[data-testid="stSidebar"] .block-container {{padding-top: 1.2rem}}
.bpr-card {{background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; margin-bottom:12px}}
.bpr-prob {{font-size:64px; font-weight:700; color:{CRIMSON}; line-height:1}}
.bpr-muted {{color:{MUTED}; font-size:13px}}
.bpr-sec {{font-size:13px; font-weight:700; color:{MUTED}; letter-spacing:.04em; margin:2px 0 6px}}
.bpr-tag-up {{background:{SEL}; color:{CRIMSON}; border-radius:999px; padding:1px 8px; font-size:12px; font-weight:600}}
.bpr-tag-dn {{background:{PILL}; color:{PILLTX}; border-radius:999px; padding:1px 8px; font-size:12px; font-weight:600}}
.bpr-chip {{display:inline-block; border-radius:999px; padding:1px 10px; font-size:12px; font-weight:600}}
.bpr-pt {{background:{CARD}; border:1px solid {LINE}; border-left:5px solid {CRIMSON}; border-radius:12px;
          padding:12px 14px 8px; margin-bottom:6px}}
.bpr-pt .nm {{font-size:19px; font-weight:700}}
.bpr-pt .id {{color:{MUTED}; font-size:12.5px; margin-bottom:8px}}
.bpr-pt .row {{display:flex; justify-content:space-between; padding:3px 0; font-size:13.5px;
               border-top:1px dashed {LINE}}}
.bpr-pt .row span:first-child {{color:{MUTED}}}
.bpr-head {{display:flex; flex-wrap:wrap; gap:6px 18px; align-items:baseline; margin-bottom:6px}}
.bpr-head .nm {{font-size:24px; font-weight:700}}
.bpr-head .k {{color:{MUTED}; font-size:13px}}
[data-testid="stSidebarHeader"] {{height:auto; padding-top:8px; padding-bottom:0}}
[data-testid="stSidebarUserContent"] {{padding-top:0 !important}}
[data-testid="stSidebarUserContent"] [data-testid="stImage"] {{margin-top:-46px}}   /* CANSLER 로고를 위로 */
img.stLogo, [data-testid="stHeaderLogo"] {{height:72px !important; max-width:300px !important; width:auto !important}}
[data-testid="stHeader"] {{height:88px}}
.stApp:has(section[data-testid="stSidebar"][data-collapsed="true"]) .block-container {{padding-top:5.6rem}}
.bpr-res-head {{position:relative; overflow:hidden; container-type:inline-size; padding:14px 16px 15px; border-radius:14px; color:#fff;
                background:linear-gradient(125deg, {CRIMSON} 0%, #d6304a 55%, {CORAL} 100%);
                box-shadow:0 6px 16px rgba(180,14,38,.22)}}
.bpr-res-head::after {{content:""; position:absolute; right:-28px; top:-34px; width:120px; height:120px; border-radius:999px;
                       background:rgba(255,255,255,.10)}}
.bpr-res-head .top {{display:flex; align-items:center; gap:8px; flex-wrap:wrap; position:relative; z-index:1}}
.bpr-res-head .ic {{flex:none; border-radius:999px; padding:2px 9px; background:rgba(255,255,255,.95); color:{CRIMSON};
                    font-size:11px; font-weight:800; letter-spacing:.03em}}
.bpr-res-head .s {{font-size:12px; opacity:.95; min-width:0}}
.bpr-res-head .t {{position:relative; z-index:1; margin-bottom:7px; font-size:clamp(11px, 5.3cqi, 19px); font-weight:800;
                   line-height:1.3; letter-spacing:-.01em; white-space:nowrap}}
.bpr-hero {{text-align:center; padding:70px 20px 34px}}
.bpr-hero .sl {{font-size:34px; font-weight:800; letter-spacing:-.01em; margin-top:26px; line-height:1.3}}
.bpr-hero .sl em {{font-style:normal; color:{CRIMSON}}}
.bpr-hero .sub {{color:{MUTED}; font-size:15px; margin-top:10px; letter-spacing:.02em}}
[data-testid="stMainMenu"], [data-testid="stToolbarActions"] {{display:none !important}}
.st-key-topbar {{position:fixed; top:12px; right:22px; z-index:999990; width:auto !important;
                background:{CARD}; border:1px solid {LINE}; border-radius:999px; padding:2px 4px 2px 14px}}
.st-key-bridge {{display:none}}
.bpr-login {{max-width:440px; margin:0 auto}}
@media print {{
  body:not(.bpr-print-ok) [data-testid="stApp"] {{display:none !important}}
  body:not(.bpr-print-ok)::before {{content:"인쇄는 BPR 화면 오른쪽 위 ⋮ 메뉴의 'Print'에서 간편 비밀번호를 입력한 뒤 할 수 있습니다.";
                                    display:block; padding:40px; font-size:16px}}
  section[data-testid="stSidebar"], header, .st-key-topbar, [data-testid="stSidebarCollapsedControl"],
  .bpr-noprint, [data-testid="stFileUploader"], .stButton, .stDownloadButton {{display:none !important}}
  .block-container {{max-width:100% !important; padding:10px !important}}
}}
.st-key-m_print button, .st-key-m_setting button {{min-height:50px !important; padding:10px 22px !important;
                                                   justify-content:flex-start !important; border-radius:10px !important}}
.st-key-m_print button p, .st-key-m_setting button p {{font-size:17px !important; font-weight:600}}
.st-key-m_print button span[data-testid="stIconMaterial"], .st-key-m_setting button span[data-testid="stIconMaterial"]
    {{font-size:24px !important}}
.st-key-m_print button:hover, .st-key-m_setting button:hover {{background:{SOFT} !important}}
.bpr-title {{display:flex; align-items:center; gap:10px; margin:0 0 12px}}
.bpr-title .ic {{flex:none; width:30px; height:30px; border-radius:9px; display:flex; align-items:center;
                justify-content:center; background:linear-gradient(135deg, {CRIMSON}, {CORAL}); color:#fff;
                box-shadow:0 3px 8px rgba(180,14,38,.25)}}
.bpr-title .ic svg {{width:17px; height:17px}}
.bpr-title .tx {{font-size:17px; font-weight:800; color:{INK}; letter-spacing:-.01em; line-height:1.2}}
.bpr-title .sub {{display:block; font-size:12px; font-weight:500; color:{MUTED}; margin-top:2px}}
.bpr-gauge {{width:176px; height:176px; border-radius:999px; margin:6px auto 12px; position:relative;
            display:flex; align-items:center; justify-content:center; box-shadow:0 6px 18px rgba(180,14,38,.16)}}
.bpr-gauge .in {{width:136px; height:136px; border-radius:999px; background:{CARD}; display:flex; flex-direction:column;
                align-items:center; justify-content:center}}
.bpr-gauge .v {{font-size:46px; font-weight:800; color:{CRIMSON}; line-height:1}}
.bpr-gauge .v span {{font-size:20px; margin-left:1px}}
.bpr-gauge .l {{font-size:12.5px; font-weight:700; color:{MUTED}; margin-top:6px; letter-spacing:.01em}}
.bpr-gauge .avg {{position:absolute; left:50%; top:-2px; width:5px; height:24px; margin-left:-2.5px; background:#fff;
                 border:1.5px solid #3a3438; border-radius:3px; transform-origin:2.5px 90px; z-index:2}}
.bpr-card.res {{container-type:inline-size}}
.bpr-bigq {{display:flex; align-items:center; justify-content:center; gap:7px; font-size:clamp(13px, 5.6cqi, 18px);
           font-weight:800; white-space:nowrap; word-break:keep-all;
           color:{INK}; text-align:center; letter-spacing:-.01em; margin-bottom:12px}}
.bpr-q2 {{position:relative; flex:none; display:inline-flex; align-items:center; justify-content:center; width:21px;
         height:21px; border-radius:999px; border:1.8px solid {MUTED}; color:{MUTED}; font-size:12.5px;
         font-weight:800; cursor:help}}
.bpr-q2:hover {{border-color:{CRIMSON}; color:{CRIMSON}}}
.bpr-q2 .tip {{white-space:normal; word-break:keep-all; overflow-wrap:break-word; visibility:hidden; opacity:0; transition:opacity .15s; position:absolute; top:150%; right:-12px; width:300px;
              background:#26212a; color:#f6f1f3; font-size:12.8px; font-weight:400; line-height:1.6; text-align:left;
              letter-spacing:0; padding:13px 15px; border-radius:12px; box-shadow:0 10px 26px rgba(0,0,0,.28); z-index:1000}}
.bpr-q2 .tip b {{color:#fff}}
.bpr-q2.up .tip {{top:auto; bottom:150%; right:-4px; width:270px}}
.bpr-q2.left .tip {{right:auto; left:-12px}}
.bpr-q2.sm {{width:18px; height:18px; font-size:11px}}
.bpr-q2:hover .tip {{visibility:visible; opacity:1}}
.bpr-cmp {{display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; background:{SOFT}; border-radius:10px;
          padding:10px 12px; font-size:14.5px; color:{INK}}}
.bpr-cmp b.w {{color:{CRIMSON}; font-size:15px; white-space:nowrap; flex:none; margin-left:auto}}
.bpr-cmp > span {{white-space:nowrap; min-width:0; flex:0 1 auto; font-size:14px}}
.bpr-kv2 {{display:flex; justify-content:space-between; align-items:center; gap:10px; padding:9px 2px; font-size:14.5px;
          border-top:1px dashed {LINE}; color:{INK}}}
.bpr-kv2 span {{color:{MUTED}}}
.bpr-kv3 {{display:flex; justify-content:space-between; align-items:center; gap:10px; padding:9px 0;
          border-bottom:1px dashed {LINE}; font-size:14.5px; color:{INK}}}
.bpr-kv3 .k {{display:flex; align-items:center; gap:6px; color:{MUTED}}}
.bpr-kv3 b {{font-size:15.5px; white-space:nowrap}}
.bpr-kv {{display:flex; justify-content:space-between; gap:10px; padding:4px 0}}
[data-testid="stSidebarLogo"] {{display:none !important}}
[class*="st-key-prow_"], [class*="st-key-qcard_"] {{position:relative}}
[class*="st-key-qcard_"] [data-testid="stElementContainer"]:has(button) {{position:absolute !important; left:0; top:0;
    width:100% !important; height:100% !important; z-index:5; margin:0 !important}}
[class*="st-key-qcard_"] [data-testid="stElementContainer"]:has(button) > div {{width:100% !important; height:100% !important}}
[class*="st-key-qcard_"] .stButton, [class*="st-key-qcard_"] button {{width:100% !important; height:100% !important;
                                                                     opacity:0; cursor:pointer}}
[class*="st-key-qcard_"]:hover .bpr-row {{border-color:{CORAL}; background:{HOVER}}}
[class*="st-key-qrow_"] {{align-items:center}}
[class*="st-key-qmenu_"] button {{padding:0 2px !important; min-height:28px !important; width:22px !important}}
[class*="st-key-qst_"] button {{width:24px !important; min-width:24px; height:24px !important; min-height:24px !important;
    padding:0 !important; border-radius:999px !important; font-size:0 !important}}
[class*="st-key-qst_"] button * {{display:none !important}}
[class*="st-key-qst_on_"] button {{background:#22c55e !important; border:2px solid #16a34a !important;
                                   box-shadow:0 0 8px rgba(34,197,94,.75)}}
[class*="st-key-qst_op_"] button {{background:#f5b301 !important; border:2px solid #d99a00 !important;
                                   box-shadow:0 0 8px rgba(245,179,1,.7)}}
[class*="st-key-qst_wait_"] button {{background:transparent !important; border:2px solid {MUTED} !important}}
.bpr-q {{display:inline-flex; width:20px; height:20px; border-radius:999px; align-items:center; justify-content:center;
         font-size:11px; font-weight:700; background:{PILL}; color:{PILLTX}; margin-right:6px; flex:none}}
.bpr-legend {{font-size:11.5px; color:{MUTED}; display:flex; gap:10px; margin:-2px 0 6px}}
.bpr-dot {{display:inline-block; width:9px; height:9px; border-radius:999px; margin-right:3px; vertical-align:-1px}}
[class*="st-key-prow_"] [data-testid="stElementContainer"]:has(button) {{position:absolute !important; left:0; top:0;
    width:100% !important; height:100% !important; z-index:5; margin:0 !important}}
[class*="st-key-prow_"] [data-testid="stElementContainer"]:has(button) > div {{width:100% !important; height:100% !important}}
[class*="st-key-prow_"] .stButton, [class*="st-key-prow_"] button {{width:100% !important; height:100% !important;
                                                                   opacity:0; cursor:pointer}}
.bpr-row {{display:flex; justify-content:space-between; align-items:center; gap:8px; padding:9px 10px 11px; margin:0 0 7px;
           background:{CARD}; border:1px solid {LINE}; border-radius:10px; transition:all .12s}}
[class*="st-key-prow_"]:hover .bpr-row {{border-color:{CORAL}; background:{HOVER}}}
.bpr-row.on {{border:1.5px solid {CRIMSON}; background:{SEL}; box-shadow:inset 4px 0 0 {CRIMSON}}}
.bpr-row .nm {{font-weight:700; font-size:14px; line-height:1.25; white-space:nowrap}}
.bpr-row .id {{color:{MUTED}; font-size:11px; white-space:nowrap}}
.bpr-row .r {{display:flex; flex-direction:column; gap:4px; align-items:flex-end}}
.bpr-row .r > div {{display:flex; gap:3px; white-space:nowrap}}
.bpr-pill {{display:inline-block; border-radius:6px; padding:1px 6px; font-size:11.5px; font-weight:600;
            background:{PILL}; color:{PILLTX}}}
.bpr-rc {{display:inline-block; border-radius:6px; padding:0 5px; font-size:11px; font-weight:700; border:1px solid}}
.bpr-rc.pos {{color:{CRIMSON}; border-color:{POSBD}; background:{POSBG}}}
.bpr-rc.neg {{color:{MUTED}; border-color:{NEGBD}; background:{NEGBG}}}
.bpr-new {{display:inline-block; border-radius:4px; padding:0 4px; font-size:10px; font-weight:700; color:#fff;
           background:{CRIMSON}; margin-left:4px; vertical-align:middle}}
.bpr-big {{font-size:19px; font-weight:700; line-height:1.3}}
[data-testid="stMetricValue"] {{font-size:1.55rem}}
[data-testid="stMetric"] {{padding:10px 14px}}
.bpr-lh {{display:flex; justify-content:space-between; color:{MUTED}; font-size:12px; padding:0 10px 4px;
          border-bottom:1px solid {LINE}}}
</style>""", unsafe_allow_html=True)


# ── 데이터 (캐시) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def patients():
    return core.patients()


@st.cache_data(show_spinner=False, ttl=15)
def saved_probs():
    return core.saved_probs()


@st.cache_resource(show_spinner="I-SPY2 자료 색인 중 ...")
def demo_inputs(pid):
    return core.demo_inputs(pid)


# ── 업로드 (TCIA I-SPY2 검사 zip) ─────────────────────────────────
def split_study(data: bytes, prefix):
    """TCIA I-SPY2 검사 폴더 zip →(조영 MRI 파일 목록, 종양 영역 SEG 파일 목록 또는 None, 압축 해제 폴더)."""
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    g = dio.find_series(tmp)
    mr = [v for k, v in g.items() if k[0] == "MR"]
    segs = [v for k, v in g.items() if k[0] == "SEG"]
    return (max(mr, key=len) if mr else None), (segs[0] if segs else None), tmp


def ispy2_check(dce, seg, pid):
    """I-SPY2 형식 검사: 등록번호(PatientID)가 선택한 환자와 같고, 종양 영역이 I-SPY2 공식 Analysis Mask인지. 문제 문장 목록."""
    import pydicom
    err = []
    got = str(pydicom.dcmread(dce[0], stop_before_pixels=True).get("PatientID", "")).strip()
    if got != pid:
        err.append(f"업로드한 검사의 등록번호({got or '없음'})가 선택한 환자({pid})와 다릅니다.")
    if seg is None:
        err.append("종양 영역(Analysis Mask)이 없습니다. TCIA I-SPY2 검사의 조영 MRI와 Analysis Mask를 함께 zip으로 올려 주세요.")
    elif "analysis mask" not in str(pydicom.dcmread(seg[0], stop_before_pixels=True).get("SeriesDescription", "")).lower():
        err.append("종양 영역 파일이 I-SPY2 Analysis Mask가 아닙니다.")
    return err


@st.cache_resource(show_spinner="MRI 읽는 중 ...", max_entries=4)
def load_dicom(data: bytes):
    dce, seg, _ = split_study(data, "bpr_dcm_")
    if dce is None:
        return None, "zip 안에서 MRI 영상을 찾지 못했습니다. TCIA에서 받은 I-SPY2 검사 폴더 전체를 zip으로 올려 주세요."
    arr, zs, sp, nrm, times, flip = dio.read_dce_files(dce)
    e, l_, why = dio.choose_timepoints(times, arr.shape[0])
    return {"arr": arr, "zs": zs, "sp": sp, "nrm": nrm, "flip": flip, "e": e, "l": l_, "why": why, "seg": seg,
            "dce_files": dce}, ""


def w3_dirs(data: bytes):
    import shutil
    dce, seg, tmp = split_study(data, "bpr_w3_")
    if dce is None or seg is None:
        return None
    out = []
    for name, fs in (("d", dce), ("m", seg)):
        (tmp / name).mkdir()
        for i, f in enumerate(fs):
            shutil.copy(f, tmp / name / f"{i:05d}.dcm")
        out.append(tmp / name)
    return tuple(out)


def dicom_panel(pid, mode):
    f = st.file_uploader("치료 전 MRI 검사 (TCIA I-SPY2 검사 폴더를 zip으로 · 조영 MRI + Analysis Mask)", type="zip",
                         key=f"dz_{pid}")
    if f is None:
        return None
    D, note = load_dicom(f.getvalue())
    if D is None:
        st.error(note); return None
    bad = ispy2_check(D["dce_files"], D["seg"], pid)
    if bad:
        for x in bad:
            st.error(x)
        return None
    arr, sp, T = D["arr"], D["sp"], D["arr"].shape[0]
    e, l_ = D["e"], D["l"]
    st.success(f"I-SPY2 검사 확인 · 조영 MRI {T}회 촬영 · Analysis Mask 있음")
    with st.expander("고급 설정 (자동 판정을 바꿀 때만)"):
        st.caption(f"화소 {sp[1]:.2f}×{sp[2]:.2f} mm · 두께 {sp[0]:.2f} mm · 시점 선택: {D['why']}")
        c1, c2 = st.columns(2)
        e = int(c1.number_input("조영 초기 영상 (몇 번째 촬영)", 1, T - 1, e, key=f"de_{pid}"))
        l_ = int(c2.number_input("조영 후기 영상 (몇 번째 촬영)", 1, T - 1, l_, key=f"dl_{pid}"))
    mask = dio.read_seg(D["seg"], D["zs"], D["nrm"], arr.shape[1:], sp[0])
    w3 = None
    if mode == "t3":
        w3f = st.file_uploader("치료 3주 차 MRI 검사 (TCIA I-SPY2 T1 검사 zip · 조영 MRI + Analysis Mask)", type="zip",
                               key=f"w3_{pid}")
        if w3f is None:
            return None
        dce3, seg3, _ = split_study(w3f.getvalue(), "bpr_w3chk_")
        bad = ["3주 차: " + x for x in ispy2_check(dce3, seg3, pid)] if dce3 else ["3주 차 검사 zip에서 MRI 영상을 찾지 못했습니다."]
        if bad:
            for x in bad:
                st.error(x)
            return None
        w3 = w3_dirs(w3f.getvalue())
    err, warn = dio.validate(arr, sp, mask, e, l_)
    for w in warn:
        st.warning(w)
    if err:
        for x in err:
            st.error(x)
        return None
    out = Path(tempfile.mkdtemp(prefix="bpr_in_"))
    dce, mp = dio.write_inputs(arr, mask, out)
    return {"dce": dce, "mask": mp, "t_pre": 0, "t_early": e, "t_late": l_, "spacing": tuple(sp), "w3": w3,
            "flip": D["flip"], "upload": True}


# ── 영상 (캐시) ───────────────────────────────────────────────────
def _keys(res, mode, pid):
    inp = res["inputs"]
    work = res.get("work_pre") or (res["work"] if mode == "pre" else Path(res["work"]) / pid)
    return (str(work), tuple((int(t), str(p)) for t, p in sorted(inp["dce"].items())), int(inp["t_pre"]),
            int(inp["t_early"]), str(inp["mask"]), tuple(inp["spacing"]))


def _w3keys(inp):
    return str(inp["w3"][0]), str(inp["w3"][1]), max(inp["dce"]) + 1, int(inp["t_early"])


@st.cache_data(show_spinner="영상 준비 중 ...", max_entries=4)
def vol_pre(work, dce_items, t_pre, t_early, mask, spacing):
    return core.view_arrays(work, dce_items, t_pre, t_early, mask, spacing)


@st.cache_data(show_spinner="3주 차 영상 준비 중 ...", max_entries=4)
def vol_w3(dce_dir, mask_dir, n_times, post_early):
    return core.view_arrays_w3(dce_dir, mask_dir, n_times, post_early)


@st.cache_data(show_spinner=False, max_entries=8)
def stats(which, keys):
    return core.tumor_stats(vol_pre(*keys) if which == "pre" else vol_w3(*keys))


@st.cache_data(show_spinner="3D 표면 만드는 중 ...", max_entries=12)
def surf(which, keys, color_by, ver):
    V = vol_pre(*keys) if which == "pre" else vol_w3(*keys)
    return core.surfaces(V, color_by)


@st.cache_data(show_spinner=False, max_entries=8)
def window(which, keys, diff):
    """모든 조영 시점에 같은 밝기 기준 (시점을 넘겨도 조영 변화가 그대로 보이게)."""
    V = vol_pre(*keys) if which == "pre" else vol_w3(*keys)
    P = V["phases"][:, ::4, ::4, ::2].astype(np.float32)
    if diff:
        P = np.clip(P - P[V["t_pre"]], 0, None)
    return float(np.percentile(P, 1)), float(np.percentile(P, 99.8))


@st.cache_data(show_spinner="MIP 만드는 중 ...", max_entries=6)
def mip_frames(which, keys, flip):
    """조영 시점별 (조영 후 − 조영 전) 최대값 투영."""
    V = vol_pre(*keys) if which == "pre" else vol_w3(*keys)
    P, t0 = V["phases"], V["t_pre"]
    base = P[t0].astype(np.float32)
    out = []
    for i in range(len(P)):
        im, asp = core.plane({"img": np.clip(P[i].astype(np.float32) - base, 0, None), "vox": V["vox"]}, "img", "MIP", 0, flip)
        out.append(im)
    return out, asp


@st.cache_data(show_spinner=False)
def hab_labels():
    mdl = np.load(cfg.DATA_FEATURES / "habitat_model.npz")
    c = mdl["centers3"] * mdl["sd"] + mdl["mu"]                    # 원 단위: washin, washout, auc, early_frac
    speed = ["천천히 들어옴", "중간 속도로 들어옴", "빨리 들어옴"]
    return [speed[j] + (" · 빨리 빠짐" if c[j][1] < 0 else " · 오래 머묾") for j in range(3)]


def fig2d(frames, names, init, m2, layer, aspect, ov=None):
    """2D 영상 + 조영 시점 막대(브라우저 안에서 바뀌므로 끄는 동안 바로 보임). frames = 시점별 uint8 영상."""
    import plotly.graph_objects as go
    from skimage import measure
    H, W = frames[0].shape
    data = [go.Heatmap(z=frames[init], colorscale="gray", zmin=0, zmax=255, showscale=False, hoverinfo="skip",
                       zsmooth="best")]
    if m2 is not None and m2.any() and layer != "원본만":
        if layer == "종양 영역":
            data.append(go.Heatmap(z=np.where(m2, 1.0, np.nan), colorscale=[[0, CORAL], [1, CORAL]], opacity=0.33,
                                   showscale=False, hoverinfo="skip"))
        elif layer == "종양 내부 패턴" and ov is not None:
            cs = [[0, PINK], [0.25, PINK], [0.25, CORAL], [0.75, CORAL], [0.75, CRIMSON], [1, CRIMSON]]
            data.append(go.Heatmap(z=np.where(ov > 0, ov - 1.0, np.nan), colorscale=cs, zmin=0, zmax=2, opacity=0.85,
                                   showscale=False, hoverinfo="skip"))
        elif layer == "조영제 유출" and ov is not None:
            data.append(go.Heatmap(z=np.where(m2, ov, np.nan), colorscale="Turbo", zmin=0.5, zmax=1.5, opacity=0.8,
                                   showscale=False, hoverinfo="skip"))
        if layer in ("종양 영역", "윤곽"):
            xs, ys = [], []
            for c in measure.find_contours(m2.astype(float), 0.5):
                xs += list(c[:, 1]) + [None]; ys += list(c[:, 0]) + [None]
            data.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=CORAL, width=1.4), hoverinfo="skip"))
    fig = go.Figure(data)
    h = int(np.clip(700 * H * aspect / W, 380, 640))
    fig.update_layout(
        height=h + 90, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#0c0b0d", plot_bgcolor="#0c0b0d",
        showlegend=False, dragmode="pan",
        xaxis=dict(visible=False, range=[-0.5, W - 0.5], constrain="domain"),
        yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", scaleratio=aspect, constrain="domain"))
    if len(frames) > 1:
        fig.update_layout(sliders=[dict(
            active=init, pad=dict(t=12, b=10, l=20, r=20), len=1, x=0, y=0, yanchor="top",
            currentvalue=dict(prefix="조영제 투입 후 시점 · ", font=dict(color="#f3eef0", size=14), xanchor="left"),
            font=dict(color="#cfc8cc", size=11), bgcolor="#3a3438", activebgcolor=CRIMSON, bordercolor="#3a3438",
            tickcolor="#6b6570", transition=dict(duration=0),
            steps=[dict(method="restyle", args=[{"z": [f.tolist()]}, [0]], label=n) for f, n in zip(frames, names)])])
    return fig


def fig3d(S, color_by, show):
    import plotly.graph_objects as go
    data = []
    if "vol" in S and show:
        x, y, z, v = S["vol"]
        data.append(go.Volume(x=x, y=y, z=z, value=v, isomin=35, isomax=255, opacity=0.07, surface_count=14,
                              colorscale="Greys_r", showscale=False, hoverinfo="skip", name="주변 조직"))
    if "tumor" in S:
        v, f, val = S["tumor"]
        kw = dict(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], name="종양", hoverinfo="name",
                  lighting=dict(ambient=0.45, diffuse=0.8, specular=0.25, roughness=0.6), flatshading=False)
        if color_by == "hab" and val is not None:
            cols = np.array([CORAL, PINK, CORAL, CRIMSON])[np.clip(val.astype(int), 0, 3)]
            data.append(go.Mesh3d(vertexcolor=cols, **kw))
        elif color_by == "ser" and val is not None:
            data.append(go.Mesh3d(intensity=val, colorscale="Turbo", cmin=0.5, cmax=1.5, intensitymode="vertex",
                                  colorbar=dict(title="SER", thickness=10, len=0.5, tickfont=dict(color="#ccc"),
                                                title_font=dict(color="#ccc")), **kw))
        else:
            data.append(go.Mesh3d(color=CORAL, **kw))
    ax = dict(visible=False)
    fig = go.Figure(data)
    fig.update_layout(scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", bgcolor="#0c0b0d",
                                 camera=dict(eye=dict(x=0.0, y=-1.35, z=0.45), up=dict(x=0, y=0, z=1))),
                      paper_bgcolor="#0c0b0d", margin=dict(l=0, r=0, t=0, b=0), height=600, showlegend=False)
    return fig


# ── 판단 근거 묶음 ────────────────────────────────────────────────
GROUPS = [("clin__HER2", None), ("clin__HR", None), ("clin__age", "나이"), ("clin__menopause", "폐경 여부"),
          ("clin__race", "인종"), ("vol__", "종양 크기"), ("t1v__", "3주 차 종양 부피 변화"),
          ("t1c__", "3주 차 조영 양상"), ("t1h__", "3주 차 종양 내부 구성"), ("t1d__", "치료 전 대비 조영 변화"),
          ("t1R_", "3주 차 종양 질감"), ("kin_", "조영 증강 양상"), ("tc_", "시간에 따른 조영 변화"),
          ("hab", "종양 내부 구성"), ("peri_", "종양 주변 조직"), ("tcW_", "조영 변화의 공간 분포"),
          ("tcO_", "조영 변화의 공간 분포"), ("tcA_", "조영 변화의 공간 분포"), ("tcF_", "조영 변화의 공간 분포"),
          ("dceE_", "종양 내부 질감"), ("dceS_", "종양 내부 질감"), ("dce_", "종양 내부 질감")]


def group_name(feat, pt):
    for pfx, name in GROUPS:
        if feat.startswith(pfx):
            if pfx == "clin__HER2":
                return f"HER2 {pt['HER2'] or '정보 없음'}"
            if pfx == "clin__HR":
                return f"호르몬 수용체(HR) {pt['HR'] or '정보 없음'}"
            return name
    return "기타 영상 특징"


def get_result(pid, mode):
    if (pid, mode) not in ss.results:
        r = core.load_saved(pid, mode)
        if r is not None:
            ss.results[(pid, mode)] = r
    return ss.results.get((pid, mode))


def sign(v):
    return {"양성": "+", "음성": "−"}.get(v, "?")


def chip(sub):
    bg, fg = CHIP.get(sub, ("#eee", INK))
    return f'<span class="bpr-chip" style="background:{bg};color:{fg}">{sub}</span>'


# ── 상태 ─────────────────────────────────────────────────────────
if "results" not in ss:
    ss.sel, ss.results, ss.run_req, ss.run_inputs = None, {}, None, None


# ══ 의료진 로그인 · 브라우저 연결(세션 유지·기기 기억·화면 모드·인쇄) ══════════════
BRIDGE = st.components.v2.component(
    "bpr_bridge",
    html="<span></span>",
    js="""
export default function (component) {
  const { data, setTriggerValue } = component
  const d = data || {}
  const K = "bpr_tok", KD = "bpr_dev"
  if (d.get) {
    setTriggerValue("boot", { token: window.sessionStorage.getItem(K) || "",
                              device: window.localStorage.getItem(KD) || "" })
  }
  if (d.set) {
    window.sessionStorage.setItem(K, d.set)
  } else if (d.clear) {
    window.sessionStorage.removeItem(K)
  }
  if (d.dev_set) {
    window.localStorage.setItem(KD, d.dev_set)
  } else if (d.dev_clear) {
    window.localStorage.removeItem(KD)
  }
  if (d.theme) {
    const key = "stActiveTheme-" + window.location.pathname + "-v2"
    const want = JSON.stringify(d.theme === "dark" ? "Dark" : "Light")
    if (window.localStorage.getItem(key) !== want) {
      window.localStorage.setItem(key, want)
      window.location.reload()
    }
  }
  if (d.print && window.__bprPrinted !== d.print) {
    window.__bprPrinted = d.print
    document.body.classList.add("bpr-print-ok")
    window.addEventListener("afterprint", () => document.body.classList.remove("bpr-print-ok"), { once: true })
    setTimeout(() => window.print(), 400)
  }
}
""",
)


def _boot():
    """페이지를 열 때 브라우저에 남은 로그인 세션(같은 탭)과 기억한 기기(간편 비밀번호)를 읽음."""
    ss.boot_checked = True
    b = (ss.get("bpr_bridge") or {}).get("boot") or {}
    dev = b.get("device") or ""
    if auth.device_user(dev):
        ss.dev_tok = dev
    elif dev:
        ss.dev_clear = True                      # 만료·해제된 기기 표시는 지움
    uid = auth.session_user(b.get("token") or "")
    if uid and not ss.uid:
        ss.uid, ss.tok = uid, b["token"]


def bridge():
    with st.container(key="bridge"):
        BRIDGE(data={"get": not ss.boot_checked, "set": ss.tok, "clear": not ss.tok,
                     "dev_set": ss.dev_tok if not ss.dev_clear else "", "dev_clear": ss.dev_clear,
                     "theme": ME.get("theme") if ss.uid else None, "print": ss.print_nonce},
               key="bpr_bridge", on_boot_change=_boot)


SIGNUP_CODE_CHECK = True    # 기관 등록 코드 확인 (켜짐) — 코드는 .streamlit/secrets.toml 의 [auth] signup_code


def _signed_in(uid, tok, remember):
    """로그인 성공 처리. remember=True면 이 브라우저를 기억해 다음부터 간편 비밀번호로 로그인."""
    ss.uid, ss.tok, ss.logout_note = uid, tok, ""
    ss.sel, ss.results = None, {}                 # 로그인하면 어떤 환자도 선택하지 않은 상태로 시작
    core.queue_reset(uid)                          # 진료 중 표시는 대기 중으로
    if remember:
        if ss.dev_tok:
            auth.forget_device(ss.dev_tok)
        ss.dev_tok, ss.dev_clear = auth.remember_device(uid), False
    elif remember is False:
        auth.forget_device(ss.dev_tok)
        ss.dev_tok, ss.dev_clear = "", True
    st.rerun()


def pin_login():
    """기억한 기기: 간편 비밀번호 6자리만 입력."""
    uid = auth.device_user(ss.dev_tok)
    u = auth.user(uid)
    st.markdown(f'<div style="text-align:center;margin:6px 0 10px"><b style="font-size:18px">{u.get("name", "")}</b> '
                f'선생님<br><span class="bpr-muted">간편 비밀번호(숫자 6자리)를 입력해 주세요</span></div>',
                unsafe_allow_html=True)
    with st.form("pin_login", border=True):
        pin = st.text_input("간편 비밀번호", type="password", max_chars=6, key="l_pin", label_visibility="collapsed",
                            placeholder="● ● ● ● ● ●")
        if st.form_submit_button("로그인", type="primary", width="stretch", icon=":material/login:"):
            tok, err = auth.login(uid, pin, "pin")
            if err:
                st.error(err)
            else:
                _signed_in(uid, tok, None)
    if st.button("아이디·비밀번호로 로그인", type="tertiary", width="stretch", icon=":material/key:"):
        ss.full_login = True
        st.rerun()


def full_login():
    with st.form("login", border=True):
        uid = st.text_input("아이디", key="l_id")
        pw = st.text_input("비밀번호", type="password", key="l_pw")
        rem = st.checkbox("다음 로그인부터 간편 비밀번호로 로그인", key="l_rem",
                          help="이 브라우저를 기억해 다음부터 숫자 6자리만 입력합니다 (LogOut 뒤에도 유지). "
                               "공용 PC에서는 체크하지 마세요. 끄려면 Setting → 간편 비밀번호 → 해제.")
        if st.form_submit_button("로그인", type="primary", width="stretch", icon=":material/login:"):
            tok, err = auth.login(uid, pw)
            if err:
                st.error(err)
            else:
                _signed_in(uid.strip().lower(), tok, rem)


def login_page():
    import base64
    logo = base64.b64encode((WEB / "assets" / ("bpr-logo-wide-dark.png" if DARK else "bpr-logo-wide.png")).read_bytes()).decode()
    cl = base64.b64encode((WEB / "assets" / ("cansler-logo-dark.png" if DARK else "cansler-logo.png")).read_bytes()).decode()
    st.markdown(f'<img src="data:image/png;base64,{cl}" alt="CANSLER" style="position:fixed; right:28px; bottom:22px; '
                f'height:78px; opacity:.95; pointer-events:none; z-index:10">', unsafe_allow_html=True)   # 오른쪽 아래 회사 로고
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.markdown(f'''<div class="bpr-hero" style="padding:50px 0 18px"><img src="data:image/png;base64,{logo}"
          style="height:96px"><div class="sl" style="font-size:24px">치료를 시작하기 전에, <em>반응</em>을 먼저 봅니다</div>
          </div>''', unsafe_allow_html=True)
        if ss.logout_note:
            st.info(ss.logout_note)
        t1, t2 = st.tabs(["의료진 로그인", "계정 만들기"])
        with t1:
            if ss.dev_tok and not ss.full_login and auth.device_user(ss.dev_tok):
                pin_login()
            else:
                full_login()
        with t2, st.form("signup", border=True):
            import datetime as dt
            a, b = st.columns(2)
            name = a.text_input("이름 *")
            birth = b.date_input("생년월일", value=None, min_value=dt.date(1930, 1, 1), max_value=dt.date.today(),
                                 format="YYYY-MM-DD")
            st.markdown('<div class="bpr-muted" style="margin-top:4px">소속</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            hosp = a.text_input("병원", placeholder="예: 서울○○병원")
            dept = b.text_input("진료과", placeholder="예: 유방외과")
            a, b = st.columns(2)
            phone = a.text_input("전화번호", placeholder="010-1234-5678")
            email = b.text_input("이메일", placeholder="name@hospital.org")
            nid = st.text_input("아이디 * (영문 소문자·숫자 4자 이상)")
            a, b = st.columns(2)
            p1 = a.text_input("비밀번호 * (영문+숫자 8자 이상)", type="password")
            p2 = b.text_input("비밀번호 확인 *", type="password")
            pin = st.text_input("간편 비밀번호 * (숫자 6자리 · 간편 로그인과 인쇄할 때 씀)", type="password", max_chars=6)
            code = st.text_input("기관 등록 코드", type="password", help="관리자에게 받은 기관 등록 코드를 입력해 주세요.")
            if st.form_submit_button("계정 만들기", type="primary", width="stretch", icon=":material/person_add:"):
                want = str(st.secrets.get("auth", {}).get("signup_code", ""))
                if SIGNUP_CODE_CHECK and code.strip() != want:
                    st.error("기관 등록 코드가 맞지 않습니다.")
                elif not name.strip():
                    st.error("이름을 입력해 주세요.")
                elif p1 != p2:
                    st.error("비밀번호 확인이 맞지 않습니다.")
                else:
                    err = auth.create_user(nid, name, hosp, dept, p1, pin, birth or "", phone, email)
                    if err:
                        st.error(err)
                    else:
                        st.success("계정을 만들었습니다. '의료진 로그인' 탭에서 로그인해 주세요.")


if not ss.uid:
    login_page()
    bridge()
    st.stop()

ALL_RAW = patients()
ALL = ALL_RAW[(ALL_RAW["구분"] != "등록") | (ALL_RAW["담당"] == ss.uid)]    # 다른 의료진이 등록한 환자는 안 보임
if ss.sel is not None and ss.sel not in set(ALL["등록번호"]):
    ss.sel = None


# ══ 오른쪽 위: 이름 · LogOut · 더보기(인쇄 · Setting) ════════════════════════
@st.dialog("Print")
def print_dialog():
    st.caption("지금 보고 있는 화면을 인쇄합니다. 로그인 계정의 간편 비밀번호(숫자 6자리)를 입력해 주세요.")
    pin = st.text_input("간편 비밀번호", type="password", max_chars=6, key="pr_pin")
    if st.button("Print", type="primary", width="stretch", icon=":material/print:"):
        if auth.check_pin(ss.uid, pin):
            ss.print_nonce = time.time()
            st.rerun()
        st.error("간편 비밀번호가 맞지 않습니다.")


@st.dialog("Setting", width="medium")
def settings_dialog():
    org = " · ".join(x for x in (ME.get("hospital"), ME.get("dept")) if x) or "소속 미입력"
    st.markdown(f"**{ME.get('name', '')}** 선생님 · {org} · 아이디 `{ss.uid}`")
    t1, t2, t3 = st.tabs(["화면 모드", "간편 비밀번호", "비밀번호·개인정보"])
    with t1:
        th = st.segmented_control("화면 모드", ["기본 모드", "다크 모드"], required=True, label_visibility="collapsed",
                                  default="다크 모드" if DARK else "기본 모드", key="set_theme")
        if (th == "다크 모드") != DARK and st.button("화면 모드 적용", icon=":material/contrast:"):
            auth.set_theme(ss.uid, "dark" if th == "다크 모드" else "light")
            st.rerun()
    with t2:
        on = auth.device_user(ss.dev_tok) == ss.uid
        st.markdown("**간편 비밀번호 로그인 (이 브라우저)** &nbsp; "
                    + ('<span class="bpr-tag-up">등록됨</span>' if on else '<span class="bpr-tag-dn">해제됨</span>'),
                    unsafe_allow_html=True)
        st.caption("등록하면 다음 로그인부터 이 브라우저에서는 숫자 6자리만 입력합니다. 간편 비밀번호는 인쇄할 때도 씁니다.")
        if on and st.button("간편 비밀번호 로그인 해제", icon=":material/lock_open:"):
            auth.forget_device(ss.dev_tok)
            ss.dev_tok, ss.dev_clear = "", True
            st.rerun()
        if not on and st.button("간편 비밀번호 로그인 등록", type="primary", icon=":material/pin:"):
            ss.dev_tok, ss.dev_clear = auth.remember_device(ss.uid), False
            st.rerun()
        with st.form("pin_form", border=True):
            st.markdown("간편 비밀번호 바꾸기")
            a, b = st.columns(2)
            pw = a.text_input("비밀번호", type="password")
            pin = b.text_input("새 간편 비밀번호 (숫자 6자리)", type="password", max_chars=6)
            if st.form_submit_button("바꾸기"):
                err = auth.change_pin(ss.uid, pw, pin)
                st.error(err) if err else st.success("간편 비밀번호를 바꿨습니다.")
    with t3:
        if time.time() - ss.get("acct_ok", 0) > 300:          # 아이디·비밀번호를 다시 확인한 뒤 5분 동안 변경 가능
            st.caption("비밀번호와 개인정보를 바꾸려면 아이디와 비밀번호를 한 번 더 입력해 주세요.")
            with st.form("acct_check", border=True):
                a, b = st.columns(2)
                cid = a.text_input("아이디")
                cpw = b.text_input("비밀번호", type="password")
                if st.form_submit_button("확인", type="primary"):
                    if cid.strip().lower() == ss.uid and auth.check_password(ss.uid, cpw):
                        ss.acct_ok = time.time()
                        st.rerun(scope="fragment")
                    st.error("아이디 또는 비밀번호가 맞지 않습니다.")
            return
        import datetime as dt
        with st.form("profile", border=True):
            st.markdown("개인정보")
            a, b = st.columns(2)
            name = a.text_input("이름 *", value=ME.get("name", ""))
            bd = ME.get("birth") or ""
            birth = b.date_input("생년월일", value=dt.date.fromisoformat(bd) if bd else None,
                                 min_value=dt.date(1930, 1, 1), max_value=dt.date.today(), format="YYYY-MM-DD")
            a, b = st.columns(2)
            hosp = a.text_input("병원", value=ME.get("hospital", ""))
            dept = b.text_input("진료과", value=ME.get("dept", ""))
            a, b = st.columns(2)
            phone = a.text_input("전화번호", value=ME.get("phone", ""))
            email = b.text_input("이메일", value=ME.get("email", ""))
            if st.form_submit_button("개인정보 저장", type="primary"):
                err = auth.update_profile(ss.uid, name=name, birth=str(birth or ""), hospital=hosp, dept=dept,
                                          phone=phone, email=email)
                st.error(err) if err else st.success("개인정보를 저장했습니다.")
        with st.form("pw_form", border=True):
            st.markdown("비밀번호 바꾸기")
            old = st.text_input("지금 비밀번호", type="password")
            a, b = st.columns(2)
            n1 = a.text_input("새 비밀번호 (영문+숫자 8자 이상)", type="password")
            n2 = b.text_input("새 비밀번호 확인", type="password")
            if st.form_submit_button("비밀번호 바꾸기"):
                err = "새 비밀번호 확인이 맞지 않습니다." if n1 != n2 else auth.change_password(ss.uid, old, n1)
                st.error(err) if err else st.success("비밀번호를 바꿨습니다.")


def logout():
    """LogOut: 로그인 세션만 끝냄. '다음 로그인부터 간편 비밀번호로 로그인'을 체크했다면 다음엔 6자리만 입력."""
    auth.logout(ss.tok)
    for k in ("uid", "sel"):
        ss[k] = None
    ss.tok, ss.results, ss.logout_note, ss.full_login = "", {}, "", False
    ss.acct_ok = 0


def topbar():
    with st.container(key="topbar", horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f'<span class="bpr-muted"><b style="color:{INK}">{ME.get("name", "")}</b> 선생님</span>',
                    unsafe_allow_html=True)
        st.button("LogOut", icon=":material/logout:", type="secondary", key="logout_btn", on_click=logout)
        with st.popover("", icon=":material/more_vert:", type="tertiary", help="더보기"):
            if st.button("Print", icon=":material/print:", type="tertiary", width="stretch", key="m_print"):
                print_dialog()
            if st.button("Setting", icon=":material/settings:", type="tertiary", width="stretch", key="m_setting"):
                settings_dialog()


topbar()


# ══ 왼쪽 패널 ══════════════════════════════════════════════════════
def patient_card(p):
    grp = {"검증": "내부 검증 환자", "학습": "학습에 쓴 환자", "등록": "신규 등록 환자"}.get(p["구분"], "")
    vol = p["부피"]
    rows = [("나이", f"만 {p['나이'] or '?'}세"), ("아형", chip(p["아형"])),
            ("수용체", rc(p["HR"], "HR") + " " + rc(p["HER2"], "HER2")), ("폐경", p["폐경"] or "정보 없음"),
            ("인종", p["인종"] or "정보 없음"), ("치료 전 종양 부피", "—" if pd.isna(vol) else f"{vol:.1f} cm³"),
            ("MRI", f"치료 전 ● · 3주 차 {'●' if p['3주차'] else '—'}"), ("구분", grp)]
    if str(p.get("기타", "")).strip():
        rows.append(("기타", str(p["기타"]).replace("\n", "<br>")))
    st.markdown(f'<div class="bpr-pt"><div class="nm">{p["환자명"]}</div><div class="id">등록번호 {p["등록번호"]}</div>'
                + "".join(f'<div class="row"><span>{k}</span><b style="text-align:right">{v}</b></div>' for k, v in rows)
                + "</div>", unsafe_allow_html=True)


def rc(v, name):
    """수용체 표시 (양성 = 빨강 테두리, 음성 = 회색)."""
    return f'<span class="bpr-rc {"pos" if v == "양성" else "neg"}">{name}{sign(v)}</span>'


def pick(pid):
    ss.sel = pid
    core.queue_pick(ss.uid, pid)


ST_NAME = {"on": "진료 중", "op": "수술 중", "wait": "대기 중"}
ST_NEXT = {"on": "op", "op": "wait", "wait": "on"}   # 원: 그 자리에서 진료 중 → 수술 중 → 대기 중 순서


def leave_queue(pid, why):
    """진료 완료 · 취소 → 진료 대기자 명단에서 뺌 (환자 목록에는 그대로 남음)."""
    core.queue_remove(ss.uid, pid, why)
    if ss.sel == pid:
        ss.sel = None
    st.toast(("진료 완료" if why == "done" else "취소") + " · 대기자 명단에서 뺐습니다", icon=":material/check:")


def set_status(pid, st_):
    core.queue_status(ss.uid, pid, st_)
    if st_ == "on":
        ss.sel = pid
    elif ss.sel == pid:
        ss.sel = None


QUEUE_UI = st.components.v2.component(
    "bpr_queue",
    html='<div class="bq"></div>',
    css="""
.bq { display:flex; flex-direction:column; gap:6px; font-family:inherit; position:relative; }
.drop { position:absolute; left:4px; right:4px; height:5px; margin-top:-2.5px; border-radius:999px; background:#fff;
        border:1.5px solid var(--crimson); box-shadow:0 0 0 3px rgba(224,64,90,.18), 0 0 12px rgba(255,255,255,.9);
        pointer-events:none; display:none; z-index:9; }
.drop::before, .drop::after { content:""; position:absolute; top:50%; width:10px; height:10px; margin-top:-5px;
        border-radius:999px; background:#fff; border:2px solid var(--crimson); }
.drop::before { left:-6px; } .drop::after { right:-6px; }
.bq.dragging .drop { display:block; }
.row { display:flex; flex-wrap:wrap; align-items:center; gap:8px; padding:8px 8px 8px 10px; position:relative;
       background:var(--card); border:1px solid var(--line); border-radius:10px; cursor:pointer;
       user-select:none; -webkit-user-select:none; touch-action:none; transition:box-shadow .15s, transform .15s; }
.row:hover { border-color:var(--coral); background:var(--hover); }
.row.on { border:1.5px solid var(--crimson); background:var(--sel); box-shadow:inset 4px 0 0 var(--crimson); }
.row.drag { box-shadow:0 12px 28px rgba(0,0,0,.25); z-index:7; cursor:grabbing; transition:none; pointer-events:none;
            opacity:.88; }
.bq.dragging .row:not(.drag) { opacity:.75; }
.grip { flex:none; width:16px; margin:-8px 0 -8px -4px; align-self:stretch; display:flex; align-items:center;
        justify-content:center; color:var(--muted); font-size:15px; cursor:grab; letter-spacing:-2px; opacity:.7; }
.grip:hover { opacity:1; color:var(--crimson); }
.no { flex:none; width:20px; height:20px; border-radius:999px; display:flex; align-items:center; justify-content:center;
      font-size:11px; font-weight:700; background:var(--pill); color:var(--pilltx); }
.who { flex:1; min-width:0; }
.nm { font-weight:700; font-size:14px; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pill { display:inline-block; border-radius:6px; padding:1px 6px; font-size:11.5px; font-weight:600;
        background:var(--pill); color:var(--pilltx); }
.id { color:var(--muted); font-size:11px; white-space:nowrap; }
.chip { flex:none; border-radius:999px; padding:1px 9px; font-size:11.5px; font-weight:600; }
.dot { flex:none; width:22px; height:22px; padding:0; border-radius:999px; border:2px solid var(--muted);
       background:transparent; cursor:pointer; }
.dot.on { background:#22c55e; border-color:#16a34a; box-shadow:0 0 8px rgba(34,197,94,.75); }
.dot.op { background:#f5b301; border-color:#d99a00; box-shadow:0 0 8px rgba(245,179,1,.7); }
.more { flex:none; border:none; background:transparent; color:var(--muted); font-size:17px; line-height:1;
        cursor:pointer; padding:0 2px; }
.menu { display:none; flex-basis:100%; gap:6px; padding-top:6px; border-top:1px dashed var(--line); }
.menu.open { display:flex; }
.menu button { flex:1; border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:8px;
               padding:6px 8px; font-size:12.5px; font-weight:600; cursor:pointer; font-family:inherit; }
.menu button:hover { background:var(--soft); }
.menu button.done { color:#16a34a; }
""",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const d = data || {}
  const root = parentElement.querySelector(".bq")
  if (!root) return
  for (const [k, v] of Object.entries(d.colors || {})) root.style.setProperty("--" + k, v)
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]))
  const send = (o) => setTriggerValue("act", Object.assign({ n: Date.now() + Math.random() }, o))
  const rows = d.rows || []
  root.innerHTML = ""
  rows.forEach((r, i) => {
    const el = document.createElement("div")
    el.className = "row" + (r.sel ? " on" : "")
    el.dataset.pid = r.pid
    el.innerHTML = '<span class="grip" title="끌어서 순서 바꾸기">⠿</span><span class="no">' + (i + 1) + '</span>'
      + '<div class="who"><div class="nm">' + esc(r.name) + ' <span class="pill">만 ' + esc(r.age) + '세</span></div>'
      + '<div class="id">' + esc(r.pid) + '</div></div>'
      + '<span class="chip" style="background:' + r.bg + ';color:' + r.fg + '">' + esc(r.sub) + '</span>'
      + '<button class="dot ' + r.st + '" aria-label="상태"></button>'
      + '<button class="more" aria-label="더보기">⋮</button>'
      + '<div class="menu"><button class="done" data-a="done">✓ 진료 완료</button>'
      + '<button data-a="cancel">✕ 취소</button></div>'
    root.appendChild(el)
  })
  const renumber = () => root.querySelectorAll(".row").forEach((x, i) => { x.querySelector(".no").textContent = i + 1 })
  // 길게 누르기(0.35초) → 끌면 줄이 손을 따라오고, 들어갈 자리에 흰 선이 표시됨 → 놓으면 그 자리로
  const drop = document.createElement("div"); drop.className = "drop"; root.appendChild(drop)
  let timer = null, drag = null, cur = null, sx = 0, sy = 0, moved = false, idx = -1, others = [], c0 = 0
  const startOrder = rows.map((r) => r.pid).join()
  const place = (y) => {
    const rr = root.getBoundingClientRect()
    idx = others.length
    for (let k = 0; k < others.length; k++) {
      const b = others[k].getBoundingClientRect()
      if (y < b.top + b.height / 2) { idx = k; break }
    }
    let top
    if (idx < others.length) top = others[idx].getBoundingClientRect().top - rr.top - 3
    else { const b = others[others.length - 1].getBoundingClientRect(); top = b.bottom - rr.top + 3 }
    drop.style.top = top + "px"
  }
  const onMove = (e) => {
    if (timer && (Math.abs(e.clientX - sx) > 8 || Math.abs(e.clientY - sy) > 8)) { clearTimeout(timer); timer = null; moved = true }
    if (!drag) return
    e.preventDefault()
    drag.style.transform = "translate(10px, " + (e.clientY - sy) + "px) scale(0.98)"
    place(c0 + (e.clientY - sy))
  }
  const onUp = () => {
    window.removeEventListener("pointermove", onMove)
    window.removeEventListener("pointerup", onUp)
    window.removeEventListener("pointercancel", onUp)
    if (timer) {
      clearTimeout(timer); timer = null
      if (!moved && cur) send({ a: "open", pid: cur.dataset.pid })
    } else if (drag) {
      drag.style.transform = ""
      drag.classList.remove("drag"); root.classList.remove("dragging")
      if (idx >= 0 && idx < others.length) root.insertBefore(drag, others[idx])
      else if (others.length) others[others.length - 1].after(drag)
      renumber()
      drag = null
      const order = [...root.querySelectorAll(".row")].map((x) => x.dataset.pid)
      if (order.join() !== startOrder) send({ a: "order", order })
    }
    cur = null; moved = false
  }
  root.querySelectorAll(".row").forEach((el) => {
    const pid = el.dataset.pid
    el.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button")) return
      e.preventDefault()
      cur = el; sx = e.clientX; sy = e.clientY; moved = false
      const start = () => {
        timer = null; drag = el; el.classList.add("drag"); root.classList.add("dragging")
        const b = el.getBoundingClientRect(); c0 = b.top + b.height / 2
        others = [...root.querySelectorAll(".row")].filter((x) => x !== el); place(c0)
      }
      if (e.target.closest(".grip")) start()                // 손잡이: 바로 끌기
      else timer = setTimeout(start, 250)                     // 줄: 0.25초 길게 누르면 끌기
      window.addEventListener("pointermove", onMove, { passive: false })
      window.addEventListener("pointerup", onUp)
      window.addEventListener("pointercancel", onUp)
    })
    el.querySelector(".dot").addEventListener("click", (e) => { e.stopPropagation(); send({ a: "status", pid }) })
    el.querySelector(".more").addEventListener("click", (e) => { e.stopPropagation(); el.querySelector(".menu").classList.toggle("open") })
    el.querySelectorAll(".menu button").forEach((b) => b.addEventListener("click", (e) => {
      e.stopPropagation(); send({ a: b.dataset.a, pid })
    }))
  })
}
""",
)


def _queue_act():
    """진료 대기자 화면에서 온 동작: 열기 · 상태 · 순서 · 진료 완료 · 취소."""
    a = (ss.get("bpr_queue") or {}).get("act") or {}
    pid, kind = a.get("pid"), a.get("a")
    q = core.queue_load(ss.uid)
    if kind == "open" and pid:
        pick(pid)
    elif kind == "status" and pid:
        i = next((k for k, x in enumerate(q) if x["pid"] == pid), None)
        if i is None:
            return
        cur_st = q[i]["st"]
        nxt = ({"on": "op", "op": "wait", "wait": "on"} if i == 0 else {"on": "op", "op": "wait", "wait": "op"})[cur_st]
        set_status(pid, nxt)
    elif kind == "order":
        order = [p for p in a.get("order", []) if any(x["pid"] == p for x in q)]
        by = {x["pid"]: x for x in q}
        newq = [dict(by[p]) for p in order] + [x for x in q if x["pid"] not in order]
        for k, x in enumerate(newq):              # 진료 중은 1번만 가능 → 아래로 내려간 진료 중 환자는 대기 중
            if k > 0 and x["st"] == "on":
                x["st"] = "wait"
                if ss.sel == x["pid"]:
                    ss.sel = None
        core.queue_save(ss.uid, newq)
    elif kind in ("done", "cancel") and pid:
        leave_queue(pid, kind)


def waiting_list():
    """진료 대기자 (최대 5명): 고른 환자가 1번 · ⠿ 끌거나 길게 눌러 순서 변경.
    상태 원: 1번 = 진료 중 → 수술 중 → 대기 중, 2~5번 = 수술 중 ↔ 대기 중."""
    st.markdown('<div class="bpr-sec">진료 대기자</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bpr-legend"><span><span class="bpr-dot" style="background:#22c55e"></span>진료 중</span>'
                f'<span><span class="bpr-dot" style="background:#f5b301"></span>수술 중</span>'
                f'<span><span class="bpr-dot" style="border:1.5px solid {MUTED}"></span>대기 중</span>'
                f'<span style="margin-left:auto">⠿ 끌거나 길게 눌러 순서 변경</span></div>', unsafe_allow_html=True)
    vis = ALL.set_index("등록번호")
    q = [x for x in core.queue_load(ss.uid) if x["pid"] in vis.index][:5]
    if ss.sel is None:                     # 새로고침(화면 모드 바꿈 등) 뒤: 진료 중 환자를 다시 엶
        ss.sel = next((x["pid"] for x in q if x["st"] == "on"), None)
    if not q:
        st.caption("아래 환자 목록에서 환자를 고르면 여기에 올라옵니다.")
        return
    rows = []
    for x in q:
        r = vis.loc[x["pid"]]
        bg, fg = CHIP.get(r["아형"], ("#eee", INK))
        rows.append({"pid": x["pid"], "name": r["환자명"], "age": r["나이"] or "?", "sub": r["아형"], "bg": bg, "fg": fg,
                     "st": x["st"], "sel": x["pid"] == ss.sel})
    colors = {"card": CARD, "line": LINE, "soft": SOFT, "hover": HOVER, "sel": SEL, "pill": PILL, "pilltx": PILLTX,
              "ink": INK, "muted": MUTED, "crimson": CRIMSON, "coral": CORAL}
    QUEUE_UI(data={"rows": rows, "colors": colors}, key="bpr_queue", on_act_change=_queue_act)


def row_html(r, on):
    new = '<span class="bpr-new">NEW</span>' if r.구분 == "등록" else ""
    return (f'<div class="bpr-row{" on" if on else ""}"><div><div class="nm">{r.환자명}{new} '
            f'<span class="bpr-pill">만 {r.나이 or "?"}세</span></div><div class="id">{r.등록번호}</div></div>'
            f'<div class="r"><div>{chip(r.아형)}</div><div>{rc(r.HR, "HR")}{rc(r.HER2, "HER2")}</div></div></div>')


def prep_upload(pid, f0, f1):
    """등록용 영상 zip → 저장된 입력(Web/cache/registered/). 반환 (입력 dict, 종양 부피 cm³, 경고, 오류)."""
    import shutil
    D, note = load_dicom(f0.getvalue())
    if D is None:
        return None, None, [], [note]
    bad = ispy2_check(D["dce_files"], D["seg"], pid)
    if bad:
        return None, None, [], bad
    arr, sp = D["arr"], D["sp"]
    mask = dio.read_seg(D["seg"], D["zs"], D["nrm"], arr.shape[1:], sp[0])
    err, warn = dio.validate(arr, sp, mask, D["e"], D["l"])
    if err:
        return None, None, warn, err
    dce, mp = dio.write_inputs(arr, mask, core.REG / pid / "pre")
    w3 = None
    if f1 is not None:
        dce3, seg3, _ = split_study(f1.getvalue(), "bpr_w3chk_")
        bad = ["3주 차: " + x for x in ispy2_check(dce3, seg3, pid)] if dce3 else ["3주 차 zip에서 MRI 영상을 찾지 못했습니다."]
        if bad:
            return None, None, warn, bad
        d = w3_dirs(f1.getvalue()); dest = core.REG / "T1" / pid
        shutil.copytree(d[0], dest / "dce", dirs_exist_ok=True); shutil.copytree(d[1], dest / "mask", dirs_exist_ok=True)
        w3 = [str(dest / "dce"), str(dest / "mask")]
    inp = {"dce": {str(t): str(v) for t, v in dce.items()}, "mask": str(mp), "t_pre": 0, "t_early": int(D["e"]),
           "t_late": int(D["l"]), "spacing": [float(x) for x in sp], "flip": bool(D["flip"]), "w3": w3}
    return inp, float(mask.sum() * np.prod(sp) / 1000), warn, []


@st.dialog("새 환자 등록", width="large")
def register_dialog():
    c = st.columns([1.3, 1.3, 0.8])
    pid = c[0].text_input("등록번호 *", placeholder="예: 12345678").strip()
    name = c[1].text_input("환자명 *", placeholder="예: 홍길동").strip()
    age = c[2].number_input("나이 (만) *", min_value=18, max_value=100, value=None, step=1, placeholder="세")
    with st.expander("환자 정보", expanded=True, icon=":material/person:"):
        a, b = st.columns(2)
        hr = a.segmented_control("호르몬 수용체(HR)", ["양성", "음성", "모름"], default="모름", required=True, key="r_hr")
        her2 = b.segmented_control("HER2", ["양성", "음성", "모름"], default="모름", required=True, key="r_her2")
        sub = core.subtype_of(hr, her2)
        st.markdown(f'<span class="bpr-muted">아형 (자동)</span> &nbsp; {chip(sub)}', unsafe_allow_html=True)
        a, b = st.columns(2)
        race = a.selectbox("인종", ["아시아인", "백인", "흑인", "기타"], key="r_race")
        meno = b.segmented_control("폐경 여부", ["폐경 후", "폐경 전", "해당 사항 없음"], default="해당 사항 없음",
                                   required=True, key="r_meno")
    with st.expander("기타", expanded=False, icon=":material/notes:"):
        etc = st.text_area("기타 정보", placeholder="병기, 조직학적 등급, Ki-67, 치료 계획 등", key="r_etc",
                           label_visibility="collapsed")
    st.markdown("**영상**")
    f0 = st.file_uploader("치료 전 MRI (TCIA I-SPY2 형식 검사 폴더 zip · 조영 MRI + Analysis Mask)", type="zip", key="r_f0")
    f1 = st.file_uploader("치료 3주 차 MRI (선택)", type="zip", key="r_f1")
    st.caption("영상은 나중에 왼쪽 '분석하기'에서 올려도 됩니다.")
    if st.button("등록하기", type="primary", width="stretch", icon=":material/how_to_reg:"):
        miss = [k for k, v in (("등록번호", pid), ("환자명", name), ("나이", age)) if not v]
        if miss:
            st.error(" · ".join(miss) + "을(를) 입력해 주세요."); return
        if pid in set(ALL_RAW["등록번호"]):
            st.error("이미 등록된 등록번호입니다."); return
        inp, vol = None, float("nan")
        if f0 is not None:
            with st.spinner("영상 확인·저장 중 ..."):
                inp, vol, warn, err = prep_upload(pid, f0, f1)
            for x in err:
                st.error(x)
            if err:
                return
        yn = lambda v: v if v in ("양성", "음성") else ""  # noqa: E731
        core.register({"등록번호": pid, "환자명": name, "나이": str(int(age)), "HR": yn(hr), "HER2": yn(her2),
                       "인종": race, "폐경": meno, "기타": etc, "부피": vol, "inputs": inp, "담당": ss.uid})
        patients.clear()
        ss.sel = pid
        st.rerun()


def patient_list():
    st.markdown('<div class="bpr-sec">환자 목록</div>', unsafe_allow_html=True)
    top = st.container(horizontal=True, vertical_alignment="center", gap="small")
    q = top.text_input("검색", placeholder="등록번호 또는 환자명", key="q", label_visibility="collapsed",
                       icon=":material/search:")
    if top.button("", icon=":material/person_add:", key="add_pt", help="새 환자 등록", type="primary"):
        register_dialog()
    with st.expander("필터", icon=":material/filter_list:"):
        sub = st.pills("아형", core.SUBTYPES, selection_mode="multi", default=core.SUBTYPES, key="sub")
        only3 = st.toggle("3주 차 MRI 있는 환자만", value=False, key="only3")
        train = st.toggle("학습에 쓴 환자 포함", value=False, key="train",
                          help="기본은 모델 학습에 쓰지 않은 내부 검증 환자 198명(+ 새로 등록한 환자)입니다. 켜면 982명 전체가 보입니다.")
    P = ALL if train else ALL[ALL["구분"] != "학습"]
    P = P[P["아형"].isin(sub or []) | ((P["구분"] == "등록") & (P["아형"] == "판정 불가"))]
    if only3:
        P = P[P["3주차"]]
    if q:
        s = q.strip()
        P = P[P["등록번호"].str.contains(s, regex=False, case=False) | P["환자명"].str.contains(s, regex=False, case=False)]
    P = P.reset_index(drop=True)
    MAX = 250
    st.caption(f"{len(P)}명 · 누르면 환자가 열립니다" + (f" · 앞 {MAX}명만 표시 (검색으로 좁혀 주세요)" if len(P) > MAX else ""))
    with st.container(height=330 if ss.sel else 540, key="plist", gap=None):
        for r in P.head(MAX).itertuples():
            with st.container(key=f"prow_{r.등록번호}", gap=None):
                st.markdown(row_html(r, r.등록번호 == ss.sel), unsafe_allow_html=True)
                st.button("열기", key=f"pt_{r.등록번호}", on_click=pick, args=(r.등록번호,), width="stretch")


def analyze_box(p):
    """분석하기: 모드 · 영상 출처 · 분석 시작. 반환 = 화면 모드."""
    st.markdown('<div class="bpr-sec">분석하기</div>', unsafe_allow_html=True)
    mode = MODES.get(ss.get("mri_sel") or "치료 전 MRI", "pre")
    if p is None:
        st.caption("환자를 고르면 분석할 수 있습니다.")
        return mode
    pid = p["등록번호"]
    own = "등록된 영상" if p["구분"] == "등록" else "I-SPY2 자료"
    src = st.segmented_control("영상 출처", [own, "검사 zip 업로드"], default=own, required=True,
                               key=f"src_{pid}", width="stretch")
    inputs = None
    if src == own:
        demo = demo_inputs(pid)
        if demo is None:
            st.warning("이 환자는 저장된 영상이 없습니다. 검사 zip을 올려 주세요.")
        elif mode == "t3" and demo["w3"] is None:
            st.warning("이 환자는 치료 3주 차 MRI가 없습니다. 가운데에서 '치료 전 MRI'를 골라 주세요.")
        else:
            inputs = demo
    else:
        inputs = dicom_panel(pid, mode)
    done = get_result(pid, mode) is not None
    st.caption("✓ 분석 완료 · 저장된 결과를 보여 줍니다" if done else "아직 분석하지 않았습니다")
    row = st.container(horizontal=True, vertical_alignment="center", gap="small")
    if row.button("분석 초기화" if done else "분석 시작", type="secondary" if done else "primary", width="stretch",
                  disabled=inputs is None, icon=":material/restart_alt:" if done else ":material/play_arrow:"):
        ss.run_req, ss.run_inputs = (pid, mode), inputs
        st.rerun()
    if done:
        row.markdown('<span class="bpr-q2 up">?<span class="tip"><b>분석 초기화</b>는 저장된 분석 결과를 지우고 '
                     '처음부터 다시 계산합니다.<br><br>같은 영상·같은 모델이면 결과는 항상 같으므로 <b>평소에는 누를 필요가 '
                     '없습니다.</b><br><br>필요한 경우<br>· 같은 환자의 검사 영상(zip)을 새로 올렸을 때<br>'
                     '· 예측 모델을 새로 바꾼 뒤 다시 계산할 때<br>· 분석 도중 끊겨 결과가 이상할 때</span></span>',
                     unsafe_allow_html=True, width="content")
    return mode


_BPR = str(WEB / "assets" / ("bpr-logo-wide-dark.png" if DARK else "bpr-logo-wide.png"))
st.logo(_BPR, size="large", icon_image=_BPR)
with st.sidebar:
    st.image(str(WEB / "assets" / ("cansler-logo-dark.png" if DARK else "cansler-logo.png")), width=120)
    card_box = st.container()
    waiting_list()
    patient_list()
    st.divider()
    cur = ALL[ALL["등록번호"] == ss.sel].iloc[0] if ss.sel in set(ALL["등록번호"]) else None
    mode = analyze_box(cur)
    if cur is not None:
        with card_box:
            patient_card(cur)


# ══ 가운데·오른쪽 ═══════════════════════════════════════════════════
def run_analysis(p, inputs):
    pid = p["등록번호"]
    bar = st.progress(0, text="분석 준비 중 ...")

    def step(f, t):
        bar.progress(min(int(f * 100), 100), text=f"분석 중 · {t}  ({f * 100:.0f}%)")
    try:
        res = core.run(pid, mode, inputs, core.to_clinical(p), save=not inputs.get("upload"), progress=step)
        ss.results[(pid, mode)] = res
        saved_probs.clear()
        return True
    except inf.InferError as e:
        bar.empty(); st.error(f"분석할 수 없는 영상입니다: {e}")
        return False


def header(p):
    st.markdown(f"""<div class="bpr-head"><span class="nm">{p['환자명']}</span>
      <span class="k">{p['등록번호']}</span><span class="k">만 {p['나이'] or '?'}세</span>{chip(p['아형'])}
      <span>{rc(p['HR'], 'HR')} {rc(p['HER2'], 'HER2')}</span>
      </div>""", unsafe_allow_html=True)


def welcome():
    import base64
    logo = base64.b64encode((WEB / "assets" / ("bpr-logo-wide-dark.png" if DARK else "bpr-logo-wide.png")).read_bytes()).decode()
    st.markdown(f'''<div class="bpr-hero"><img src="data:image/png;base64,{logo}" style="height:120px">
      <div class="sl">치료를 시작하기 전에, <em>반응</em>을 먼저 봅니다</div>
      <div class="sub">See the response before the treatment</div></div>''', unsafe_allow_html=True)
    with st.container(border=True, height=200, vertical_alignment="center", horizontal_alignment="center"):
        st.markdown('<div style="text-align:center"><b>왼쪽 환자 목록에서 환자를 고르세요</b><br>'
                    '<span class="bpr-muted">등록번호나 환자명으로 찾거나 필터로 아형을 고를 수 있습니다. '
                    '환자를 고르면 MRI 2D·3D 영상과 예측 결과가 이곳에 나타납니다.</span></div>',
                    unsafe_allow_html=True)


MRI_TIP = ("<b>치료 전 MRI</b>: 항암치료를 시작하기 전에 찍은 MRI로 예측합니다. 치료 시작 전에 바로 쓸 수 있습니다. "
           "(내부 검증 AUC 0.757)<br><br><b>3주 차 MRI</b>: 항암치료 3주 뒤 찍은 MRI까지 더해, 종양이 얼마나 줄고 "
           "조영 양상이 어떻게 바뀌었는지 반영해 예측합니다. 더 정확하지만 3주 차 검사가 있어야 합니다. "
           "(내부 검증 AUC 0.816)<br><br>고른 MRI에 맞춰 가운데 영상과 오른쪽 예측 결과가 함께 바뀝니다.")
REAL_TIP = ("<b>실제 결과</b>는 이 환자가 선행항암치료를 마친 뒤 받은 <b>수술의 병리 결과</b>입니다 "
            "(I-SPY2 임상시험 공개 기록).<br><br><b>pCR</b> = 수술 조직의 유방·겨드랑이 림프절에 침윤암이 남아 있지 않음<br>"
            "<b>잔존 종양</b> = 침윤암이 남아 있음<br><br>AI 예측이 실제와 얼마나 맞는지 비교해 볼 수 있도록 보여 줍니다. "
            "새로 등록한 환자는 수술 전이라 표시되지 않습니다.")


def mri_picker(p):
    """가운데 패널 위: 치료 전 / 3주 차 MRI (영상 + 예측 모델) + 설명 물음표."""
    row = st.container(horizontal=True, vertical_alignment="center", gap="small")
    row.segmented_control("MRI", list(MODES), default="치료 전 MRI", required=True, key="mri_sel",
                          label_visibility="collapsed")
    row.markdown(f'<span class="bpr-q2 left">?<span class="tip">{MRI_TIP}</span></span>'
                 + ("" if p["3주차"] else '<span class="bpr-muted" style="margin-left:10px">이 환자는 3주 차 MRI 없음</span>'),
                 unsafe_allow_html=True, width="content")


def main():
    if cur is None:
        welcome(); return
    p, pid = cur, cur["등록번호"]
    header(p)
    mri_picker(p)
    res = get_result(pid, mode)
    req = ss.run_req == (pid, mode)
    if req or res is None:
        inputs = ss.run_inputs if req else None
        ss.run_req, ss.run_inputs = None, None
        if inputs is None:                       # 저장된 결과가 없으면 I-SPY2 자료로 바로 분석
            demo = demo_inputs(pid)
            inputs = demo if demo is not None and (mode == "pre" or demo["w3"] is not None) else None
        if inputs is None:
            st.info("이 환자는 " + ("치료 3주 차 MRI가 없습니다. 위에서 '치료 전 MRI'를 골라 주세요."
                                 if mode == "t3" else "저장된 영상이 없습니다. 왼쪽 분석하기에서 검사 zip을 올려 주세요."))
            return
        if run_analysis(p, inputs):
            st.rerun()
        return
    c, r = st.columns([2.05, 1.1], gap="large")
    with c:
        viewer(p, res)
    with r:
        result_panel(p, res)


# ── 영상 + 정량 수치 (조작 시 이 부분만 다시 그림) ─────────────────────
@st.fragment
def viewer(p, res):
    pid, inp = p["등록번호"], res["inputs"]
    keys = _keys(res, mode, pid)
    with st.container(border=True):
        st.markdown(sec_title("mri", "MRI 영상", ("치료 3주 차" if mode == "t3" else "치료 전") + " 조영 MRI · 2D 단면 / 3D 재구성"),
                    unsafe_allow_html=True)
        t = st.container(horizontal=True, vertical_alignment="center")
        dim = t.segmented_control("보기", ["2D", "3D"], default="2D", required=True, key="dim",
                                  label_visibility="collapsed")
        when = "3주 차" if mode == "t3" else "치료 전"
        if when == "3주 차":
            which, k2 = "w3", _w3keys(inp)
            W, layers = vol_w3(*k2), ["종양 영역", "원본만"]
            names = core.phase_names(len(W["phases"]), 0, W["t_early"])
        else:
            which, k2 = "pre", keys
            W, layers = vol_pre(*keys), ["종양 영역", "종양 내부 패턴", "조영제 유출", "원본만"]
            ts = sorted(inp["dce"])
            names = core.phase_names(len(W["phases"]), W["t_pre"], W["t_early"],
                                     ts.index(inp["t_late"]) if inp["t_late"] in ts else None)
        ph = view2d(t, W, layers, inp["flip"], when, names, which, k2) if dim == "2D" else view3d(t, when, which, k2)
    numbers(res, which, k2, names, ph, when)


def view2d(t, W, layers, flip, when, names, which, keys):
    view = t.segmented_control("단면", ["축상", "시상", "관상", "MIP"], default="축상", required=True, key="plane",
                               label_visibility="collapsed",
                               help="축상 = 촬영 단면 · 시상/관상 = 재구성 단면 · MIP = 조영 전후 차이 영상의 최대값 투영")
    layer = t.segmented_control("표시", layers, default=layers[0], required=True, key=f"layer_{when}",
                                label_visibility="collapsed")
    diff = st.toggle("차이 영상 (조영 후 − 조영 전)", value=False, key=f"diff_{when}", disabled=view == "MIP",
                     help="조영 전 영상을 빼서 조영제가 들어온 곳만 밝게 봅니다")
    P, t0, init = W["phases"], W["t_pre"], W["t_early"]
    sub = diff or view == "MIP"
    lo, hi = window(which, keys, sub)
    T = {"mask": W["mask"], "vox": W["vox"], "hab": W.get("hab"), "ser": W.get("ser")}
    if view == "MIP":
        imgs, asp = mip_frames(which, keys, flip)
        hi = hi * 1.3; lo = 0.0
        m2, _ = core.plane(T, "mask", "MIP", 0, flip)
        k, n = 0, 1
    else:
        n = core.n_slices(W, view)
        k = st.slider("단면 위치", 1, n, core.center(W, view) + 1, key=f"k_{view}_{when}_{ss.sel}") - 1
        imgs = []
        base = core.plane({"img": P[t0], "vox": W["vox"]}, "img", view, k, flip)[0].astype(np.float32)
        for i in range(len(P)):
            im, asp = core.plane({"img": P[i], "vox": W["vox"]}, "img", view, k, flip)
            im = im.astype(np.float32)
            imgs.append(np.clip(im - base, 0, None) if sub else im)
        m2, _ = core.plane(T, "mask", view, k, flip)
    frames = [np.clip((im - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8) for im in imgs]
    lay = ("윤곽" if layer != "원본만" else "원본만") if view == "MIP" else layer
    ov = {"종양 내부 패턴": "hab", "조영제 유출": "ser"}.get(lay)
    o2 = core.plane(T, ov, view, k, flip)[0] if ov and T.get(ov) is not None else None
    st.plotly_chart(fig2d(frames, names, init, m2, lay, asp, o2), key=f"im_{when}_{view}_{k}_{lay}_{sub}_{ss.sel}",
                    config={"displayModeBar": False, "scrollZoom": True, "doubleClick": "reset"})
    if layer == "종양 내부 패턴" and view != "MIP":
        st.markdown(" &nbsp; ".join(f'<span style="color:{c}">■</span> {x}'
                                    for c, x in zip([PINK, CORAL, CRIMSON], hab_labels())), unsafe_allow_html=True)
    elif layer == "조영제 유출" and view != "MIP":
        st.caption("파랑 = 조영제가 천천히 빠짐 · 빨강 = 빨리 빠짐 (SER, 초기·후기 영상 기준)")
    else:
        st.caption(f"{when} MRI · " + ("조영 후 − 조영 전 차이 영상의 최대값 투영(MIP) · 분홍 선 = 종양 영역 투영"
                                       if view == "MIP" else f"{view} 단면 {k + 1}/{n}" + (" · 차이 영상" if diff else ""))
                   + " · 아래 막대를 끌면 조영 시점이 바뀝니다 · 휠로 확대, 두 번 누르면 원래 크기")
    return None


def view3d(t, when, which, keys):
    opts = ["종양 영역", "종양 내부 패턴", "조영제 유출"] if when == "치료 전" else ["종양 영역"]
    col = t.segmented_control("종양 색", opts, default=opts[0], required=True, key=f"c3_{when}",
                              label_visibility="collapsed")
    show = t.toggle("주변 조직", value=True, key=f"s3_{when}",
                    help="조영 전후 차이 영상을 3D로 투영 (혈관·조영되는 유선 조직이 밝게 보임)")
    color_by = {"종양 내부 패턴": "hab", "조영제 유출": "ser"}.get(col, "none")
    S = surf(which, keys, color_by, Path(core.__file__).stat().st_mtime)
    st.plotly_chart(fig3d(S, color_by, show), config={"displaylogo": False}, key=f"p3_{when}_{color_by}_{show}")
    if color_by == "hab":
        st.markdown(" &nbsp; ".join(f'<span style="color:{c}">■</span> {x}'
                                    for c, x in zip([PINK, CORAL, CRIMSON], hab_labels())), unsafe_allow_html=True)
    st.caption(f"{when} MRI · 종양 표면 3D 재구성 + 주변 조직(조영 전후 차이) · 끌어서 돌리고, 휠로 확대합니다")
    return None


TIP = {
    "curve": "<b>시간-신호 곡선</b>은 조영제를 주사한 뒤 시점마다 <b>종양의 평균 신호가 조영 전보다 몇 % 늘었는지</b>를 "
             "이은 선입니다.<br><br>· <b>유출형</b>: 빨리 올라갔다가 내려감 → 악성 종양에서 흔함<br>"
             "· <b>평탄형</b>: 초기 이후 거의 그대로<br>· <b>지속 상승형</b>: 끝까지 계속 올라감 → 양성 병변에서 흔함<br><br>"
             "영상 아래 조영 시점 막대를 움직이면 각 시점의 영상을 볼 수 있습니다.",
    "ftv_pch": "치료 전과 비교해 3주 차에 <b>조영 증강 종양 부피(FTV)</b>가 얼마나 줄었는지(−) 또는 늘었는지(+)입니다. "
               "치료 초기에 크게 줄수록 pCR 가능성이 높은 경향이 있습니다.",
    "washout": "종양 안에서 <b>SER이 1.1보다 큰 화소</b>의 비율입니다.<br><br>SER(신호 증강비) = (초기 − 조영 전) ÷ "
               "(후기 − 조영 전). 1.1보다 크면 조영제가 빨리 빠지는 <b>유출형</b> 조직으로, 악성도가 높은 부분에서 흔합니다.",
    "enh": "조영 초기 영상에서 신호가 조영 전보다 <b>70% 넘게 늘어난</b> 종양 화소의 비율입니다 "
           "(I-SPY2 종양 부피 FTV와 같은 기준). 높을수록 종양 전체가 고르게 강하게 조영됩니다.",
    "peak": "시간-신호 곡선에서 <b>가장 높은 평균 증강률</b>과 그때의 조영 시점입니다.",
    "ser": "종양 화소들의 <b>평균 SER</b>입니다. 1보다 크면 초기에 강하게 들어왔다가 빠지는 양상, "
           "1보다 작으면 시간이 갈수록 계속 차오르는 양상입니다.",
    "ftv": "조영이 일정 수준 이상 되는 <b>종양 부피(FTV)</b>를 치료 전과 3주 차로 비교한 값입니다.",
    "hab": "종양 화소를 <b>조영제가 들어오는 속도와 빠지는 양상</b>으로 3가지 유형(habitat)으로 나눈 비율입니다. "
           "같은 종양 안에서도 부분마다 성질이 다른 정도(이질성)를 보여 줍니다.",
}


def tip(key, cls="sm"):
    return f'<span class="bpr-q2 {cls}">?<span class="tip">{TIP[key]}</span></span>'


def numbers(res, which, keys, names, ph, when):
    """영상 아래 정량 수치: 크기 · 조영 곡선 · 내부 구성."""
    import plotly.graph_objects as go
    S = stats(which, keys)
    f = res["features"]
    st.markdown('<div style="height:10px"></div>' + sec_title("chart", "정량 수치", f"{when} MRI · 종양 크기와 조영 양상"),
                unsafe_allow_html=True)
    mc = st.columns(4 if "early_pct" in S else 2, gap="small")   # 한 줄 고정 (칸이 다음 줄로 넘어가지 않게)
    mc[0].metric("종양 부피", f"{S['vol_cm3']:.1f} cm³", border=True, help="종양 영역(Analysis Mask) 부피")
    mc[1].metric("최대 직경", f"{S['feret_mm'] / 10:.1f} cm", border=True, help="가장 큰 병변의 최대 길이")
    if "early_pct" in S:
        mc[2].metric("초기 증강률", f"{S['early_pct']:.0f}%", border=True,
                     help="조영 초기 영상에서 종양 평균 신호가 조영 전보다 늘어난 비율")
        mc[3].metric("곡선 유형", S["curve_type"], border=True,
                 help=f"초기 이후 마지막 시점까지 변화 {S['late_change']:+.0f}% "
                      "(−10% 아래 = 유출형, +10% 위 = 지속 상승형, 사이 = 평탄형)")
    g1, g2 = st.columns([1.2, 1], gap="medium")
    with g1, st.container(border=True):
        st.markdown(f'<div class="bpr-title" style="margin-bottom:2px"><span class="ic">{_ICON["curve"]}</span>'
                    f'<div class="tx" style="display:flex;align-items:center;gap:7px">시간-신호 곡선{tip("curve", "left")}'
                    f'</div></div><div class="bpr-muted" style="font-size:12px;margin-left:40px">종양 평균 증강률 · '
                    f'조영 전 대비</div>', unsafe_allow_html=True)
        if "curve" in S:
            x = list(range(len(S["curve"])))
            fig = go.Figure(go.Scatter(x=x, y=S["curve"], mode="lines+markers", line=dict(color=CRIMSON, width=3),
                                       marker=dict(size=9, color=CRIMSON, line=dict(color=CARD, width=2)), text=names,
                                       fill="tozeroy", fillcolor="rgba(180,14,38,0.08)",
                                       hovertemplate="%{text}<br>%{y:.0f}%<extra></extra>"))
            fig.update_layout(height=300, margin=dict(l=8, r=28, t=14, b=8), showlegend=False, plot_bgcolor=CARD,
                              paper_bgcolor=CARD, font=dict(color=INK, size=12),
                              xaxis=dict(tickvals=x, ticktext=names, tickfont=dict(size=11), gridcolor=LINE,
                                         range=[-0.35, len(x) - 0.65], automargin=True),
                              yaxis=dict(ticksuffix="%", gridcolor=LINE, zeroline=False, tickfont=dict(size=11.5)))
            st.plotly_chart(fig, config={"displayModeBar": False, "responsive": True}, key=f"curve_{which}", width="stretch")
    with g2:
        rows = []
        if mode == "t3" and when == "3주 차" and res.get("ftv_pch") is not None:
            rows.append(("부피 변화 (치료 전 대비)", f"{res['ftv_pch']:+.0f}%", "ftv_pch"))
        if "washout_frac" in S:
            rows.append(("유출 영역 비율", f"{S['washout_frac']:.0f}%", "washout"))
        if "enh_frac" in S:
            rows.append(("강하게 조영되는 종양 비율", f"{S['enh_frac']:.0f}%", "enh"))
        if "peak_pct" in S:
            rows.append(("최고 증강률 (시점)", f"{S['peak_pct']:.0f}% ({names[S['peak_t']]})", "peak"))
        if "ser_mean" in S:
            rows.append(("평균 SER", f"{S['ser_mean']:.2f}", "ser"))
        if mode == "t3" and res.get("ftv_cm3"):
            a, b = res["ftv_cm3"]
            rows.append(("조영 증강 부피", f"{float(a):.1f} → {float(b):.1f} cm³", "ftv"))
        html = "".join(f'<div class="bpr-kv3"><span class="k">{k}{tip(t)}</span><b>{v}</b></div>' for k, v, t in rows)
        fr = [f.get(f"hab3_{j}_frac", np.nan) for j in range(3)]
        if not np.isnan(fr).any():
            seg = "".join(f'<div style="width:{v * 100:.1f}%;background:{c}"></div>'
                          for v, c in zip(fr, [PINK, CORAL, CRIMSON]))
            lab = "".join(f'<div class="bpr-kv" style="font-size:13.5px;color:{INK}"><span><span style="color:{c}">■</span> '
                          f'{x}</span><b>{v * 100:.0f}%</b></div>' for v, c, x in zip(fr, [PINK, CORAL, CRIMSON], hab_labels()))
            html += (f'<div class="bpr-kv3" style="border-bottom:none;padding-bottom:2px"><span class="k">치료 전 종양 내부 구성'
                     f'{tip("hab")}</span></div>'
                     f'<div style="display:flex;height:13px;border-radius:999px;overflow:hidden;margin:6px 0 4px">{seg}</div>{lab}')
        st.markdown(f'<div class="bpr-card" style="margin:0;padding:16px 18px">'
                    f'{sec_title("table", "종양 분석 결과", "영상에서 계산한 주요 수치")}{html}</div>', unsafe_allow_html=True)


# ── 오른쪽: 결과 · 판단 근거 · 소견 · CSV ─────────────────────────────
_ICON = {
    "result": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" r="9"/>'
              '<circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="1" fill="currentColor"/></svg>',
    "why": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round">'
           '<path d="M5 19V11M12 19V5M19 19v-6"/></svg>',
    "note": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linejoin="round">'
            '<path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13 7l4 4"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">'
             '<path d="M4 20h16"/><path d="M7 16v-5M12 16V7M17 16v-8"/></svg>',
    "curve": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
             'stroke-linejoin="round"><path d="M3 19c3 0 4-11 7-11s4 7 6 7 3-4 5-4"/></svg>',
    "table": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linejoin="round">'
             '<rect x="4" y="4" width="16" height="16" rx="2.5"/><path d="M4 10h16M10 10v10"/></svg>',
    "mri": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3"><circle cx="12" cy="12" r="8.5"/>'
           '<path d="M8 12a4 4 0 0 1 8 0" stroke-linecap="round"/><circle cx="12" cy="14" r="1.6" fill="currentColor"/></svg>',
    "save": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
            'stroke-linejoin="round"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
}


def sec_title(icon, text, sub=""):
    return (f'<div class="bpr-title"><span class="ic">{_ICON[icon]}</span><div class="tx">{text}'
            + (f'<span class="sub">{sub}</span>' if sub else "") + '</div></div>')


PCR_TIP = ("<b>병리학적 완전관해(pCR)</b>란 수술 전 항암치료(선행항암치료)를 마친 뒤 수술로 떼어 낸 조직에서 "
           "<b>유방과 겨드랑이 림프절에 침윤암이 남아 있지 않은 상태</b>(ypT0/Tis ypN0)를 말합니다.<br><br>"
           "이 값은 AI가 <b>MRI 영상(종양 크기·조영 양상·내부 구성 등)과 임상 정보(HR·HER2·나이 등)</b>로 계산한 "
           "pCR 도달 확률입니다. pCR에 도달한 환자는 재발 위험이 낮은 경향이 있습니다.<br><br>"
           "원 둘레의 <b>하얀 눈금</b>은 이 환자와 같은 아형 환자들이 실제로 pCR에 도달한 비율(평균) 위치입니다. "
           "빨간 부분이 하얀 눈금을 넘으면 같은 아형 평균보다 가능성이 높다는 뜻입니다.")


def result_panel(p, res):
    prob = res["prob"] * 100; sub = p["아형"]; avg = SUBTYPE_RATE.get(sub)
    needle = "" if avg is None else f'<div class="avg" style="transform:rotate({avg * 3.6:.1f}deg)"></div>'
    gauge = (f'<div class="bpr-gauge" style="background:conic-gradient({CRIMSON} 0 {prob:.1f}%, {SOFT} {prob:.1f}% 100%)">'
             f'{needle}<div class="in"><div class="v">{prob:.0f}<span>%</span></div><div class="l">pCR 도달 확률</div></div>'
             f'</div>')
    big = (f'<div class="bpr-bigq">항암치료 후 종양 완전 소실 가능성'
           f'<span class="bpr-q2">?<span class="tip">{PCR_TIP}</span></span></div>')
    cmp = ""
    if avg is not None:
        word = "평균보다 높음" if prob > avg + 5 else ("평균보다 낮음" if prob < avg - 5 else "평균과 비슷함")
        avg_tip = (f"I-SPY2 임상시험 환자 982명 가운데 <b>{sub}</b> 아형 환자들이 실제로 pCR에 도달한 비율이 "
                   f"<b>{avg}%</b>입니다 (원 둘레의 하얀 눈금).<br><br>이 환자의 예측값 <b>{prob:.0f}%</b>를 이 평균과 "
                   f"비교해<br>· 5%p 넘게 높으면 <b>평균보다 높음</b><br>· 5%p 넘게 낮으면 <b>평균보다 낮음</b><br>"
                   f"· 차이가 5%p 이내면 <b>평균과 비슷함</b><br>으로 표시합니다.")
        cmp = (f'<div class="bpr-cmp"><span style="display:flex;align-items:center;gap:6px">같은 아형({sub}) 평균 '
               f'<b>{avg}%</b><span class="bpr-q2 sm">?<span class="tip">{avg_tip}</span></span></span>'
               f'<b class="w">{word}</b></div>')
    extra = ""
    if mode == "t3":
        extra = (f'<div class="bpr-kv2" style="margin-top:8px"><span>치료 전 예측 → 3주 차 반영</span>'
                 f'<b>{res["prob_pre"] * 100:.0f}% → {prob:.0f}%</b></div>')
    gap = "" if extra else ' style="margin-top:8px"'
    real = "" if not p["실제"] else (
        f'<div class="bpr-kv2"{gap}><span style="display:flex;align-items:center;gap:6px">실제 결과 (I-SPY2 수술 기록)'
        f'<span class="bpr-q2 sm">?<span class="tip">{REAL_TIP}</span></span></span>'
        f'<b style="color:{CRIMSON if p["실제"] == "pCR" else INK}">{p["실제"]}</b></div>')
    st.markdown(f'<div class="bpr-card res">{sec_title("result", "분석 결과", AUC[mode][0])}{gauge}{big}{cmp}{extra}{real}'
                f'<div class="bpr-muted" style="font-size:11.5px;margin-top:6px">내부 검증 정확도(AUC) {AUC[mode][1]}</div>'
                f'</div>', unsafe_allow_html=True)
    evidence(p, res)
    note = notes(p)
    csv_box(p, res, prob, avg, note)


def evidence(p, res):
    e = inf.explain(res["features"], "ens_main_lr" if mode == "pre" else "ens_t3_lr", 16 if mode == "pre" else 12)
    g = e.groupby([group_name(f, p) for f in e.index]).sum()
    g = g[g.abs() > 1e-3].sort_values(key=abs, ascending=False).head(6)
    mx = float(g.abs().max()) if len(g) else 1.0
    rows = ""
    for k, v in g.items():
        tag = '<span class="bpr-tag-up">높임</span>' if v > 0 else '<span class="bpr-tag-dn">낮춤</span>'
        rows += (f'<div style="display:grid;grid-template-columns:1.5fr 1fr 48px;gap:10px;align-items:center;'
                 f'padding:5px 0;font-size:14.5px;color:{INK}"><span>{k}</span>'
                 f'<div style="height:9px;background:{SOFT};border-radius:999px;overflow:hidden">'
                 f'<div style="height:100%;width:{abs(v) / mx * 100:.0f}%;background:{CRIMSON if v > 0 else "#8a8390"}"></div>'
                 f'</div><span style="text-align:right">{tag}</span></div>')
    st.markdown(f'<div class="bpr-card">{sec_title("why", "판단 근거", "예측에 영향을 준 주요 요인")}{rows}</div>',
                unsafe_allow_html=True)


def notes(p):
    pid = p["등록번호"]
    saved = core.load_note(pid, ss.uid)
    with st.container(border=True):
        st.markdown(sec_title("note", "의사 소견"), unsafe_allow_html=True)
        txt = st.text_area("의사 소견", value=saved["text"], key=f"note_{pid}", height=130, label_visibility="collapsed",
                           placeholder="판독 소견, 치료 계획 메모 등을 적어 주세요")
        c = st.container(horizontal=True, vertical_alignment="center")
        if c.button("소견 저장", icon=":material/save:", key=f"ns_{pid}"):
            saved = core.save_note(pid, txt, ss.uid)
            st.toast("소견을 저장했습니다", icon=":material/check:")
        c.caption(f"저장: {saved['saved']}" if saved["saved"] else "아직 저장하지 않았습니다")
    return txt


def csv_box(p, res, prob, avg, note):
    pid = p["등록번호"]
    S = stats("pre", _keys(res, mode, pid))
    row = {"등록번호": pid, "환자명": p["환자명"], "나이": p["나이"], "아형": p["아형"],
           "수용체": f"HR{sign(p['HR'])} HER2{sign(p['HER2'])}", "분석 모드": AUC[mode][0],
           "pCR 가능성(%)": round(prob, 1), "아형 평균 pCR(%)": avg, "실제 결과": p["실제"],
           "치료 전 종양 부피(cm³)": round(S["vol_cm3"], 2), "치료 전 최대 직경(cm)": round(S["feret_mm"] / 10, 2),
           "초기 조영 증강률(%)": round(S.get("early_pct", np.nan), 1), "조영 곡선 유형": S.get("curve_type", ""),
           "유출 영역 비율(%)": round(S.get("washout_frac", np.nan), 1)}
    if mode == "t3":
        row.update({"치료 전 예측(%)": round(res["prob_pre"] * 100, 1), "3주 차 부피 변화(%)": res.get("ftv_pch")})
    row["의사 소견"] = note
    with st.container(border=True):
        st.markdown(sec_title("save", "결과 저장"), unsafe_allow_html=True)
        st.download_button("결과 CSV 내려받기", pd.DataFrame([row]).to_csv(index=False).encode("utf-8-sig"),
                           icon=":material/download:", file_name=f"BPR_{pid}_{mode}.csv", mime="text/csv",
                           width="stretch", type="primary")
        st.caption("예측 결과 · 정량 수치 · 의사 소견이 한 줄로 저장됩니다 (엑셀에서 바로 열림)")


main()
bridge()
