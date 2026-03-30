from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_rembg = types.ModuleType("rembg")
fake_rembg.new_session = lambda model_name: {"model": model_name}
fake_rembg.remove = lambda in_bytes, session=None, **kwargs: in_bytes
sys.modules.setdefault("rembg", fake_rembg)


import base64

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.exceptions import ApiError
import app.main as main_module
import app.api.routes as routes_module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "warmup_models", lambda: None)
    with TestClient(main_module.app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_service_info_uses_success_envelope(client):
    response = client.get("/service-info")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "data" in body
    assert body["data"]["processing_defaults"]["max_side"] == 512


def test_remove_bg_success_response_format(client, monkeypatch):
    png_bytes = b"fake-png-bytes"
    expected_b64 = base64.b64encode(png_bytes).decode("utf-8")

    def fake_process_remove_bg(*, raw: bytes):
        assert raw == b"abc"
        return {
            "image": {
                "filename": "removed_bg.png",
                "mime_type": "image/png",
                "base64": expected_b64,
                "width": 100,
                "height": 200,
            },
            "model": "isnet-general-use",
            "fallback_used": False,
            "edge_quality_low_candidate": False,
            "metrics": {
                "edge_band_ratio": 0.1,
                "edge_band_mid_ratio": 0.05,
                "edge_band_low_ratio": 0.01,
            },
            "processing": {
                "max_side": 512,
                "quality": "fast",
                "reject_low_confidence": True,
                "reject_edge_quality": True,
            },
        }

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.png", b"abc", "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "ok": True,
        "data": {
            "image": {
                "filename": "removed_bg.png",
                "mime_type": "image/png",
                "base64": expected_b64,
                "width": 100,
                "height": 200,
            },
            "model": "isnet-general-use",
            "fallback_used": False,
            "edge_quality_low_candidate": False,
            "metrics": {
                "edge_band_ratio": 0.1,
                "edge_band_mid_ratio": 0.05,
                "edge_band_low_ratio": 0.01,
            },
            "processing": {
                "max_side": 512,
                "quality": "fast",
                "reject_low_confidence": True,
                "reject_edge_quality": True,
            },
        },
    }


def test_remove_bg_business_rejection_error_format(client, monkeypatch):
    def fake_process_remove_bg(*, raw: bytes):
        raise ApiError(
            status_code=422,
            code="LOW_CONFIDENCE_MASK",
            message="背景過於複雜或主體邊界不清楚，建議改用純色背景重新拍攝。",
            details={
                "reason": "complex_background_low_confidence",
                "metrics": {
                    "foreground_ratio": 0.2,
                    "alpha_mean": 10.0,
                    "mid_alpha_ratio": 0.1,
                    "high_alpha_ratio": 0.1,
                    "bbox_width_ratio": 0.8,
                    "bbox_height_ratio": 0.9,
                    "edge_band_ratio": 0.2,
                    "edge_band_mid_ratio": 0.1,
                    "edge_band_low_ratio": 0.05,
                    "edge_quality_low_candidate": True,
                },
            },
        )

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.png", b"abc", "image/png")},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "LOW_CONFIDENCE_MASK"
    assert body["error"]["details"]["reason"] == "complex_background_low_confidence"


def test_remove_bg_request_validation_error_format(client):
    response = client.post("/remove-bg")

    assert response.status_code == 422
    body = response.json()
    assert body == {
        "ok": False,
        "error": {
            "code": "REQUEST_VALIDATION_ERROR",
            "message": "請求參數驗證失敗。",
            "details": {
                "errors": [
                    {
                        "loc": ["body", "file"],
                        "msg": "Field required",
                        "type": "missing",
                    }
                ]
            },
        },
    }


def test_remove_bg_http_error_format(client, monkeypatch):
    def fake_process_remove_bg(*, raw: bytes):
        raise HTTPException(status_code=400, detail="Invalid image")

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.png", b"abc", "image/png")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "BAD_REQUEST",
            "message": "Invalid image",
            "details": None,
        },
    }


def test_remove_bg_unexpected_error_format(client, monkeypatch):
    def fake_process_remove_bg(*, raw: bytes):
        raise RuntimeError("boom")

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.png", b"abc", "image/png")},
    )

    assert response.status_code == 500
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "服務發生未預期錯誤。",
            "details": None,
        },
    }


def test_openapi_remove_bg_no_long_query_parameters(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/remove-bg"]["post"]

    assert operation.get("parameters") in (None, [])
    request_body = operation["requestBody"]
    assert "multipart/form-data" in request_body["content"]