"""
auth.py ─ 의료진 계정 (로그인 · 비밀번호 · 간편 비밀번호 · 세션) — Streamlit 의존 없음
==================================================================
저장: Web/cache/auth/users.json (비밀번호·간편 비밀번호(6자리, 인쇄 때도 씀)는 PBKDF2-SHA256 + 무작위 salt 해시만 저장, 원문 없음)
      Web/cache/auth/sessions.json (로그인 세션 토큰 → 사용자, 만료 시각)
계정 만들기는 기관 등록 코드(.streamlit/secrets.toml 의 [auth] signup_code)를 알아야 함.
로그인 5회 연속 실패 시 5분 잠금. 세션은 마지막 사용 후 30분 지나면 만료.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import re
import secrets
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent / "cache" / "auth"
USERS, SESS = DIR / "users.json", DIR / "sessions.json"
IDLE_SEC = 30 * 60
LOCK_N, LOCK_SEC = 5, 5 * 60
_ITER = 200_000


def _load(f):
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _save(f, d):
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(f)


def _hash(secret, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), bytes.fromhex(salt), _ITER).hex()
    return f"{salt}${h}"


def _check(secret, stored):
    try:
        salt, _ = stored.split("$", 1)
    except (AttributeError, ValueError):
        return False
    return hmac.compare_digest(_hash(secret, salt), stored)


def pw_problem(pw):
    if len(pw) < 8 or not re.search(r"[A-Za-z]", pw) or not re.search(r"\d", pw):
        return "비밀번호는 영문과 숫자를 섞어 8자 이상이어야 합니다."
    return ""


def pin_problem(pin):
    return "" if re.fullmatch(r"\d{6}", pin or "") else "간편 비밀번호는 숫자 6자리여야 합니다."


def create_user(uid, name, hospital, dept, pw, pin, birth="", phone="", email=""):
    """새 계정. pin = 간편 비밀번호(숫자 6자리, 간편 로그인·인쇄에 씀). 반환: 오류 문장 ("" = 성공)."""
    uid = uid.strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{4,30}", uid):
        return "아이디는 영문 소문자·숫자 4~30자로 정해 주세요."
    U = _load(USERS)
    if uid in U:
        return "이미 쓰고 있는 아이디입니다."
    err = pw_problem(pw) or pin_problem(pin)
    if err:
        return err
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()):
        return "이메일 형식이 맞지 않습니다."
    if phone and not re.fullmatch(r"[0-9\-+ ]{9,16}", phone.strip()):
        return "전화번호는 숫자와 - 로 입력해 주세요 (예: 010-1234-5678)."
    U[uid] = {"name": name.strip(), "hospital": hospital.strip(), "dept": dept.strip(), "birth": str(birth or ""),
              "phone": phone.strip(), "email": email.strip(), "pw": _hash(pw), "pin": _hash(pin), "theme": "light",
              "created": time.strftime("%Y-%m-%d %H:%M"), "fails": 0, "lock_until": 0}
    _save(USERS, U)
    return ""


def login(uid, pw, field="pw"):
    """반환 (토큰 또는 None, 오류 문장). field="pin"이면 간편 비밀번호(6자리)로 로그인."""
    uid = uid.strip().lower()
    U = _load(USERS)
    u = U.get(uid)
    now = time.time()
    what = "간편 비밀번호" if field == "pin" else "비밀번호"
    if u is None:
        return None, f"아이디 또는 {what}가 맞지 않습니다."
    if u.get("lock_until", 0) > now:
        return None, f"로그인을 여러 번 실패해 잠겼습니다. {int((u['lock_until'] - now) // 60) + 1}분 뒤 다시 시도해 주세요."
    if not _check(pw, u[field]):              # 비밀번호·간편 비밀번호 실패 횟수는 함께 셈
        u["fails"] = u.get("fails", 0) + 1
        if u["fails"] >= LOCK_N:
            u["fails"], u["lock_until"] = 0, now + LOCK_SEC
        _save(USERS, U)
        return None, f"아이디 또는 {what}가 맞지 않습니다."
    u["fails"], u["lock_until"] = 0, 0
    _save(USERS, U)
    tok = secrets.token_urlsafe(32)
    S = {k: v for k, v in _load(SESS).items() if v["exp"] > now}
    S[tok] = {"user": uid, "exp": now + IDLE_SEC}
    _save(SESS, S)
    return tok, ""


def session_user(tok, touch=True):
    """유효한 세션이면 사용자 아이디 (마지막 사용 시각을 늘림), 아니면 None."""
    if not tok:
        return None
    S = _load(SESS)
    s = S.get(tok)
    now = time.time()
    if s is None or s["exp"] < now or s["user"] not in _load(USERS):
        return None
    if touch and s["exp"] - now < IDLE_SEC - 60:     # 1분에 한 번만 파일을 고침
        s["exp"] = now + IDLE_SEC
        _save(SESS, S)
    return s["user"]


def logout(tok):
    S = _load(SESS)
    if S.pop(tok, None) is not None:
        _save(SESS, S)


def user(uid):
    u = dict(_load(USERS).get(uid, {}))
    u.pop("pw", None); u.pop("pin", None)
    return u


def check_pin(uid, pin):
    u = _load(USERS).get(uid)
    return bool(u) and _check(pin, u["pin"])


def change_password(uid, old, new):
    U = _load(USERS)
    if uid not in U or not _check(old, U[uid]["pw"]):
        return "지금 비밀번호가 맞지 않습니다."
    err = pw_problem(new)
    if err:
        return err
    U[uid]["pw"] = _hash(new)
    _save(USERS, U)
    return ""


def change_pin(uid, pw, pin):
    U = _load(USERS)
    if uid not in U or not _check(pw, U[uid]["pw"]):
        return "비밀번호가 맞지 않습니다."
    err = pin_problem(pin)
    if err:
        return err
    U[uid]["pin"] = _hash(pin)
    _save(USERS, U)
    return ""


def set_theme(uid, theme):
    U = _load(USERS)
    if uid in U:
        U[uid]["theme"] = theme
        _save(USERS, U)


# ── 기기 기억 (다음 로그인부터 간편 비밀번호) ───────────────────────────
DEV = DIR / "devices.json"      # sha256(기기 토큰) → 사용자 (토큰 원문은 브라우저에만 있음)
DEV_DAYS = 30


def _dh(tok):
    return hashlib.sha256(tok.encode("utf-8")).hexdigest()


def remember_device(uid):
    """이 브라우저에서 다음부터 간편 비밀번호로 로그인하게 함. 반환: 브라우저에 둘 기기 토큰."""
    tok = secrets.token_urlsafe(32)
    D = {k: v for k, v in _load(DEV).items() if v["exp"] > time.time()}
    D[_dh(tok)] = {"user": uid, "exp": time.time() + DEV_DAYS * 86400}
    _save(DEV, D)
    return tok


def device_user(tok):
    if not tok:
        return None
    d = _load(DEV).get(_dh(tok))
    return d["user"] if d and d["exp"] > time.time() and d["user"] in _load(USERS) else None


def forget_device(tok):
    D = _load(DEV)
    if tok and D.pop(_dh(tok), None) is not None:
        _save(DEV, D)


def update_profile(uid, **kw):
    """개인정보 변경 (이름·생년월일·병원·진료과·전화번호·이메일). 반환: 오류 문장."""
    U = _load(USERS)
    if uid not in U:
        return "계정을 찾을 수 없습니다."
    email, phone = kw.get("email", "").strip(), kw.get("phone", "").strip()
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return "이메일 형식이 맞지 않습니다."
    if phone and not re.fullmatch(r"[0-9\-+ ]{9,16}", phone):
        return "전화번호는 숫자와 - 로 입력해 주세요 (예: 010-1234-5678)."
    if not kw.get("name", "").strip():
        return "이름을 입력해 주세요."
    for k in ("name", "birth", "hospital", "dept", "phone", "email"):
        U[uid][k] = str(kw.get(k, "") or "").strip()
    _save(USERS, U)
    return ""


def check_password(uid, pw):
    u = _load(USERS).get(uid.strip().lower())
    return bool(u) and _check(pw, u["pw"])
