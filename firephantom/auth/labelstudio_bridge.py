"""Open a Label Studio browser session for a FirePhantom user.

Label Studio Community has no OAuth client. The server logs into the shadow
account and returns the signed session cookie for the browser to store.
"""

from __future__ import annotations

from urllib.parse import urlparse
import os

import httpx

from firephantom.annotation.labelstudio import endpoint_from_env


class LabelStudioAuthError(RuntimeError):
    """Label Studio could not accept the shadow account."""


def origin() -> str:
    parsed = urlparse(endpoint_from_env())
    if not parsed.scheme or not parsed.netloc:
        raise LabelStudioAuthError("LABEL_STUDIO_URL 不是有效地址")
    return f"{parsed.scheme}://{parsed.netloc}"


def browser_url(hostname: str) -> str:
    parsed = urlparse(origin())
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if (parsed.scheme == "https" and port == 443) or (parsed.scheme == "http" and port == 80):
        return f"{parsed.scheme}://{hostname}/"
    return f"{parsed.scheme}://{hostname}:{port}/"


def open_session(email: str, shadow_password: str) -> str:
    """Return a Label Studio ``sessionid`` for this email, creating the user when needed."""
    base = origin()
    try:
        with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
            existing = _login(client, base, email, shadow_password)
            if existing:
                return existing
            _signup(client, base, email, shadow_password)
            created = _session_id(client)
            if created:
                return created
            signed_in = _login(client, base, email, shadow_password)
    except LabelStudioAuthError:
        raise
    except httpx.HTTPError as exc:
        raise LabelStudioAuthError(f"Label Studio 不可用：{exc}") from exc
    if not signed_in:
        raise LabelStudioAuthError("Label Studio 没有接受这个账号的登录")
    return signed_in


def logout(session_id: str | None) -> None:
    if not session_id:
        return
    base = origin()
    try:
        with httpx.Client(timeout=10, follow_redirects=False, trust_env=False) as client:
            client.get(f"{base}/logout/", cookies={"sessionid": session_id})
    except httpx.HTTPError as exc:
        raise LabelStudioAuthError(f"Label Studio 不可用：{exc}") from exc


def _csrf(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    token = client.cookies.get("csrftoken")
    if response.status_code >= 500 or not token:
        raise LabelStudioAuthError(f"Label Studio 登录页不可用（HTTP {response.status_code}）")
    return token


def _session_id(client: httpx.Client) -> str | None:
    return client.cookies.get("sessionid") or None


def _login(client: httpx.Client, base: str, email: str, password: str) -> str | None:
    page = f"{base}/user/login/"
    token = _csrf(client, page)
    response = client.post(
        page,
        data={
            "email": email,
            "password": password,
            "csrfmiddlewaretoken": token,
            "persist_session": "on",
        },
        headers={"Referer": page},
    )
    session_id = _session_id(client)
    location = response.headers.get("location", "")
    if response.status_code in {301, 302, 303} and session_id and "login" not in location:
        return session_id
    return None


def _signup(client: httpx.Client, base: str, email: str, password: str) -> None:
    token = os.environ.get("LABEL_STUDIO_TOKEN", "").strip() or os.environ.get("LABEL_STUDIO_API_KEY", "").strip()
    if not token:
        raise LabelStudioAuthError("没有配置 LABEL_STUDIO_TOKEN，无法创建标注账号")
    invite = client.get(f"{base}/api/invite", headers={"Authorization": f"Token {token}"})
    if invite.status_code != 200:
        raise LabelStudioAuthError(f"无法读取 Label Studio 邀请（HTTP {invite.status_code}）")
    try:
        invite_token = invite.json()["token"]
    except (ValueError, KeyError, TypeError) as exc:
        raise LabelStudioAuthError("Label Studio 邀请响应缺少 token") from exc
    page = f"{base}/user/signup/?token={invite_token}"
    csrf = _csrf(client, page)
    response = client.post(
        f"{base}/user/signup/",
        params={"token": invite_token},
        data={
            "email": email,
            "password": password,
            "allow_newsletters": "",
            "how_find_us": "",
            "elaborate": "",
            "csrfmiddlewaretoken": csrf,
        },
        headers={"Referer": page},
    )
    if response.status_code >= 500:
        raise LabelStudioAuthError(f"Label Studio 注册失败（HTTP {response.status_code}）")
