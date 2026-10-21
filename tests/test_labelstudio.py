import json

from firephantom.annotation import labelstudio


def test_label_studio_pull_uses_project_route_and_results(monkeypatch):
    calls = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"count": 1, "results": [{"id": 17, "is_labeled": False}]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, endpoint, *, params, headers):
            calls.update(endpoint=endpoint, params=params, headers=headers)
            return Response()

    monkeypatch.setenv("LABEL_STUDIO_URL", "http://127.0.0.1:8080/api/tasks")
    monkeypatch.setenv("LABEL_STUDIO_PROJECT_ID", "7")
    monkeypatch.setattr(labelstudio.httpx, "Client", Client)

    result = labelstudio.pull_tasks(limit=12)

    assert calls["endpoint"] == "http://127.0.0.1:8080/api/projects/7/tasks/"
    assert calls["params"] == {"page": 1, "page_size": 100}
    assert result["success"] is True
    assert result["tasks"] == [{"id": 17, "is_labeled": False}]


def test_label_studio_pull_follows_pages_and_reads_the_review(monkeypatch):
    pages = []

    class Response:
        status_code = 200

        def __init__(self, page):
            self.page = page

        def raise_for_status(self):
            pass

        def json(self):
            if self.page == 1:
                return {"total": 3, "tasks": [{"id": 1}, {"id": 2}]}
            return {"total": 3, "tasks": [{"id": 3}]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get(self, endpoint, *, params, headers):
            pages.append(params["page"])
            return Response(params["page"])

    monkeypatch.setattr(labelstudio.httpx, "Client", Client)
    result = labelstudio.pull_tasks(page_size=2)
    assert pages == [1, 2]
    assert [task["id"] for task in result["tasks"]] == [1, 2, 3]

    annotation = {"result": [
        {"type": "rectanglelabels", "original_width": 100, "original_height": 50,
         "value": {"x": 10, "y": 20, "width": 30, "height": 40, "rectanglelabels": ["fire"]}},
        {"type": "keypointlabels", "original_width": 100, "original_height": 50,
         "value": {"x": 20, "y": 50, "keypointlabels": ["flame_center"]}},
        {"type": "choices", "value": {"choices": ["室内", "浓烟"]}},
    ]}
    (obj,) = labelstudio.objects_from_annotation(annotation, 100, 50, "image.png")
    assert obj["category"] == "flame"
    assert obj["bbox"] == [10.0, 10.0, 30.0, 20.0]
    assert obj["keypoints"] == [{"name": "flame_center", "x": 20.0, "y": 25.0}]
    assert obj["attributes"] == ["室内", "浓烟"]
    newest = labelstudio.latest_annotation({"annotations": [
        {"id": 1, "updated_at": "2026-01-02"}, {"id": 2, "updated_at": "2026-01-01"},
    ]})
    assert newest["id"] == 1


class _Response:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self):
        return self._body


def test_push_task_stores_predictions_on_their_own_endpoint(monkeypatch):
    calls = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, *, json, headers):
            calls.append(("post", url, json))
            if url.rstrip("/").endswith("/api/tasks"):
                return _Response(201, {"id": 9})
            return _Response(201, {"id": 4})

        def delete(self, url, *, headers):
            calls.append(("delete", url))
            return _Response(204, {})

    monkeypatch.setenv("LABEL_STUDIO_URL", "http://127.0.0.1:8080/api/tasks")
    monkeypatch.setenv("LABEL_STUDIO_PROJECT_ID", "1")
    monkeypatch.setattr(labelstudio.httpx, "Client", Client)
    task = {
        "data": {"image": "/api/v1/samples/x/files/image.png", "sample_id": "x"},
        "predictions": [{
            "model_version": "firephantom-1.0.0",
            "score": 1.0,
            "result": [{"type": "choices", "value": {"choices": ["晴"]}}],
        }],
    }
    result = labelstudio.push_task(task)
    assert result["success"] is True and result["task_id"] == 9
    created, predicted = [item for item in calls if item[0] == "post"]
    assert created[1].rstrip("/").endswith("/api/tasks")
    assert "predictions" not in created[2]
    assert created[2]["project"] == 1
    assert predicted[1].rstrip("/").endswith("/api/predictions")
    assert predicted[2]["task"] == 9
    assert predicted[2]["project"] == 1
    assert predicted[2]["result"][0]["type"] == "choices"


def test_push_task_deletes_the_task_when_the_prediction_is_refused(monkeypatch):
    calls = []

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, *, json, headers):
            calls.append(("post", url))
            if url.rstrip("/").endswith("/api/tasks"):
                return _Response(201, {"id": 9})
            return _Response(400, {"detail": "no"})

        def delete(self, url, *, headers):
            calls.append(("delete", url))
            return _Response(204, {})

    monkeypatch.setattr(labelstudio.httpx, "Client", Client)
    monkeypatch.setenv("LABEL_STUDIO_URL", "http://127.0.0.1:8080/api/tasks")
    monkeypatch.setenv("LABEL_STUDIO_PROJECT_ID", "1")
    task = {"data": {"image": "http://lab/a.png"}, "predictions": [{"result": []}]}
    result = labelstudio.push_task(task)
    assert result["success"] is False
    assert result["task_id"] is None
    assert calls[-1] == ("delete", "http://127.0.0.1:8080/api/tasks/9/")


def test_backfill_writes_predictions_only_for_empty_tasks(tmp_path, monkeypatch):
    sample = "abc"
    folder = tmp_path / "samples" / sample
    folder.mkdir(parents=True)
    (folder / "label_studio.json").write_text(json.dumps({
        "predictions": [{"model_version": "firephantom-1.0.0", "score": 1, "result": []}],
    }), encoding="utf-8")
    monkeypatch.setattr(labelstudio, "pull_tasks", lambda **_kwargs: {
        "success": True,
        "tasks": [
            {"id": 3, "predictions": [], "data": {"sample_id": sample}},
            {"id": 4, "predictions": [{"id": 1}], "data": {"sample_id": sample}},
            {"id": 5, "predictions": [], "data": {"sample_id": "missing"}},
        ],
    })
    posted = []

    def fake_post(_client, _endpoint, task_id, _project, _predictions):
        posted.append(task_id)
        return None

    monkeypatch.setattr(labelstudio, "_post_predictions", fake_post)
    monkeypatch.setattr(labelstudio.httpx, "Client", _Response and type("Client", (), {
        "__init__": lambda self, **_kwargs: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *_args: False,
    }))
    result = labelstudio.backfill_missing_predictions(tmp_path)
    assert posted == [3]
    assert result["filled"] == 1
    assert result["skipped"] == 2
    assert result["success"] is True


def test_label_studio_origin_can_load_sample_files(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from firephantom.api import create_app

    monkeypatch.setenv("LABEL_STUDIO_URL", "http://127.0.0.1:8080/api/tasks")
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    client = TestClient(create_app(tmp_path))
    url = "/api/v1/samples/missing/files/image.png"

    for origin in ("http://127.0.0.1:8080", "http://localhost:8080"):
        response = client.get(url, headers={"Origin": origin})
        assert response.headers.get("access-control-allow-origin") == origin
    preflight = client.options(url, headers={"Origin": "http://127.0.0.1:8080", "Access-Control-Request-Method": "GET"})
    assert preflight.status_code == 200
    assert "access-control-allow-origin" not in client.get(url, headers={"Origin": "http://evil.example"}).headers
