import json

import numpy as np
import pytest

import main

VIDEO = {"file": ("squat.mp4", b"not a real video, the models are mocked", "video/mp4")}


def register(client, email="a@test.com", password="password123"):
    r = client.post("/register", json={"email": email, "password": password, "name": "Test"})
    assert r.status_code == 200, r.text
    body = r.json()
    return body["user_id"], {"Authorization": f"Bearer {body['access_token']}"}


@pytest.fixture
def mock_models(monkeypatch, make_squat):
    """swaps YOLO out for synthetic keypoints so the endpoint runs without a video"""
    xy, conf = make_squat()
    n = len(xy)
    monkeypatch.setattr(main, "run_pose", lambda path: (xy, conf))
    #bar never detected, so every bar position comes back as NaN
    monkeypatch.setattr(main, "run_detection",
                        lambda path, weights: (np.full((n, 2), np.nan, np.float32), np.zeros(n, np.float32)))


def test_register_rejects_duplicate_email(client):
    register(client)
    r = client.post("/register", json={"email": "a@test.com", "password": "other", "name": "Test"})
    assert r.status_code == 400


def test_login(client):
    register(client)
    ok = client.post("/login", json={"email": "a@test.com", "password": "password123"})
    wrong_password = client.post("/login", json={"email": "a@test.com", "password": "nope"})
    unknown_email = client.post("/login", json={"email": "b@test.com", "password": "password123"})

    assert ok.status_code == 200
    assert ok.json()["access_token"]
    assert wrong_password.status_code == 401
    assert unknown_email.status_code == 401


def test_rejects_missing_or_invalid_token(client):
    user_id, _ = register(client)
    missing = client.get(f"/users/{user_id}")
    invalid = client.get(f"/users/{user_id}", headers={"Authorization": "Bearer garbage"})

    assert missing.status_code in (401, 403)   #HTTPBearer's code for no header varies by FastAPI version
    assert invalid.status_code == 401


def test_cannot_read_another_users_data(client, mock_models):
    a_id, a_headers = register(client, "a@test.com")
    b_id, b_headers = register(client, "b@test.com")
    assert client.post("/analyze-video", headers=b_headers, files=VIDEO).status_code == 200
    b_session = client.get(f"/users/{b_id}/sessions", headers=b_headers).json()[0]["id"]

    assert client.get(f"/users/{b_id}", headers=a_headers).status_code == 403
    assert client.get(f"/users/{b_id}/sessions", headers=a_headers).status_code == 403
    assert client.get(f"/sessions/{b_session}", headers=a_headers).status_code == 403
    assert client.get("/sessions/9999", headers=a_headers).status_code == 404


@pytest.mark.parametrize("params, files", [
    ({}, {"file": ("notes.txt", b"hello", "text/plain")}),
    ({"fps": 500}, VIDEO),
])
def test_analyze_video_rejects_bad_input(client, params, files):
    _, headers = register(client)
    r = client.post("/analyze-video", headers=headers, params=params, files=files)
    assert r.status_code == 400


def test_analyze_video_saves_session(client, mock_models):
    user_id, headers = register(client)

    r = client.post("/analyze-video", headers=headers, files=VIDEO)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_reps"] == 3
    #frames with no bar must serialise as null, not crash the response with NaN
    assert all(point == [None, None] for point in body["bar_path"])

    sessions = client.get(f"/users/{user_id}/sessions", headers=headers).json()
    assert len(sessions) == 1
    assert sessions[0]["total_reps"] == 3
    assert [rep["depth_quality"] for rep in sessions[0]["reps"]] == ["below"] * 3


def test_convert_numpy_output_is_json_safe():
    out = main.convert_numpy({
        "array": np.array([1.5, np.nan]),
        "int": np.int64(3),
        "bool": np.bool_(True),
        "nested": [np.float32(np.inf)],
    })
    assert out == {"array": [1.5, None], "int": 3, "bool": True, "nested": [None]}
    json.dumps(out, allow_nan=False)
