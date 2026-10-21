from fastapi.testclient import TestClient

from firephantom.api import create_app
from firephantom.auth.store import SERVICE_EMAIL


def test_bootstrap_login_and_admin_accounts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        anonymous = client.get("/api/v1/auth/me")
        assert anonymous.status_code == 200
        assert anonymous.json()["needs_bootstrap"] is True

        rejected = client.post("/api/v1/auth/bootstrap", json={"email": "not-an-email", "password": "longenough"})
        assert rejected.status_code == 400

        reserved = client.post(
            "/api/v1/auth/bootstrap",
            json={"email": SERVICE_EMAIL, "password": "longenough"},
        )
        assert reserved.status_code == 400

        created = client.post(
            "/api/v1/auth/bootstrap",
            json={"email": "Admin@Example.com", "password": "correct-horse"},
        )
        assert created.status_code == 200
        assert created.json()["is_admin"] is True
        assert created.json()["email"] == "admin@example.com"
        assert client.cookies.get("fp_session")

        again = client.post(
            "/api/v1/auth/bootstrap",
            json={"email": "other@example.com", "password": "correct-horse"},
        )
        assert again.status_code == 409

        added = client.post("/api/v1/auth/users", json={"email": "reviewer@example.com", "password": "review-pass"})
        assert added.status_code == 200
        assert added.json()["is_admin"] is False
        duplicate = client.post("/api/v1/auth/users", json={"email": "reviewer@example.com", "password": "review-pass"})
        assert duplicate.status_code == 409

        client.post("/api/v1/auth/logout")
        assert client.get("/api/v1/auth/me").json()["authenticated"] is False
        assert client.get("/auth/label-studio").status_code == 401
        forbidden = client.post("/api/v1/auth/users", json={"email": "third@example.com", "password": "review-pass"})
        assert forbidden.status_code == 401

        wrong = client.post("/api/v1/auth/login", json={"email": "reviewer@example.com", "password": "nope-nope"})
        assert wrong.status_code == 401
        signed_in = client.post(
            "/api/v1/auth/login",
            json={"email": "reviewer@example.com", "password": "review-pass"},
        )
        assert signed_in.status_code == 200
        assert signed_in.json()["is_admin"] is False
        still_forbidden = client.post(
            "/api/v1/auth/users",
            json={"email": "third@example.com", "password": "review-pass"},
        )
        assert still_forbidden.status_code == 403


def test_label_studio_bridge_sets_session_cookie(tmp_path, monkeypatch):
    calls = []

    class Response:
        def __init__(self, status_code, payload=None, location=""):
            self.status_code = status_code
            self._payload = payload or {}
            self.headers = {"location": location}

        def json(self):
            return self._payload

    class Client:
        def __init__(self, **_kwargs):
            self.cookies = {}
            self.login_attempts = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, url, **kwargs):
            calls.append(("GET", url))
            if url.endswith("/api/invite"):
                return Response(200, {"token": "invite-token"})
            self.cookies["csrftoken"] = "csrf-token"
            return Response(200)

        def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs.get("data", {}).get("email")))
            if url.endswith("/user/login/"):
                self.login_attempts += 1
                if self.login_attempts > 1 and kwargs["data"]["password"] == "shadow-secret":
                    self.cookies["sessionid"] = "signed-session"
                    return Response(302, location="/projects/")
            return Response(200)

    monkeypatch.setenv("LABEL_STUDIO_URL", "http://127.0.0.1:8080/api/tasks")
    monkeypatch.setenv("LABEL_STUDIO_TOKEN", "service-token")
    monkeypatch.setattr("firephantom.auth.labelstudio_bridge.httpx.Client", Client)
    monkeypatch.setattr(
        "firephantom.auth.store.secrets.token_urlsafe",
        lambda _n: "shadow-secret",
    )

    with TestClient(create_app(tmp_path)) as client:
        client.post("/api/v1/auth/bootstrap", json={"email": "person@example.com", "password": "correct-horse"})
        response = client.get("/auth/label-studio", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "http://testserver:8080/"
    assert response.cookies.get("sessionid") == "signed-session"
    assert ("GET", "http://127.0.0.1:8080/api/invite") in calls


def test_schema_setup_runs_once_even_in_parallel():
    import threading

    import firephantom.db as db

    db._schema_ready.clear()
    calls = []
    original = db._apply_schema

    def wrapped(url=None):
        calls.append(url)
        return original(url)

    db._apply_schema = wrapped
    try:
        errors = []

        def run():
            try:
                db.ensure_schema()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert len(calls) == 1
        db.ensure_schema()
        assert len(calls) == 1
    finally:
        db._apply_schema = original
