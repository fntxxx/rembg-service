from pathlib import Path
import os
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fake_rembg = types.ModuleType("rembg")
fake_rembg.new_session = lambda model_name: {"model": model_name}
fake_rembg.remove = lambda in_bytes, session=None, **kwargs: in_bytes
sys.modules.setdefault("rembg", fake_rembg)

fake_pillow_heif = types.ModuleType("pillow_heif")
fake_pillow_heif.register_heif_opener = lambda *args, **kwargs: None
sys.modules.setdefault("pillow_heif", fake_pillow_heif)


import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import DEFAULT_MODEL, DEFAULT_REJECT_EDGE_QUALITY, DEFAULT_REJECT_LOW_CONFIDENCE, SERVICE_NAME
from app.core.exceptions import ApiError
from app.core.file_validation import (
    ALLOWED_IMAGE_CONTENT_TYPES,
    ALLOWED_IMAGE_EXTENSIONS,
    validate_image_content_type,
    validate_image_filename,
)
import app.main as main_module
import app.api.routes as routes_module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module, "warmup_models", lambda: None)
    monkeypatch.setenv("INTERNAL_API_TOKEN", "test-token")
    with TestClient(main_module.app, raise_server_exceptions=False) as test_client:
        yield test_client


def auth_headers(token: str = "test-token"):
    return {"Authorization": f"Bearer {token}"}


def test_service_info_uses_success_envelope(client):
    response = client.get("/service-info", headers=auth_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "data" in body
    assert body["data"]["processing_defaults"]["max_side"] == 512
    assert body["data"]["service"] == SERVICE_NAME
    assert body["data"]["model"] == DEFAULT_MODEL
    assert body["data"]["processing_defaults"]["reject_low_confidence"] is DEFAULT_REJECT_LOW_CONFIDENCE
    assert body["data"]["processing_defaults"]["reject_edge_quality"] is DEFAULT_REJECT_EDGE_QUALITY


def test_health_and_healthz_share_same_contract(client):
    health_response = client.get("/health")
    healthz_response = client.get("/healthz")

    assert health_response.status_code == 200
    assert health_response.json() == healthz_response.json()
    assert health_response.json() == {
        "ok": True,
        "data": {
            "service": SERVICE_NAME,
            "model": DEFAULT_MODEL,
            "model_warmed": False,
        },
    }


def test_remove_bg_success_response_returns_png_binary(client, monkeypatch):
    png_bytes = b"fake-png-bytes"

    def fake_process_remove_bg(*, raw: bytes):
        assert raw == b"abc"
        return {
            "image_bytes": png_bytes,
            "headers": {
                "X-RemoveBg-Model": "isnet-general-use",
                "X-RemoveBg-Fallback-Used": "false",
                "X-Edge-Quality-Candidate": "false",
                "X-Edge-Band-Ratio": "0.100000",
                "X-Edge-Band-Mid-Ratio": "0.050000",
                "X-Edge-Band-Low-Ratio": "0.010000",
            },
        }

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.png", b"abc", "image/png")},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert response.content == png_bytes
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-removebg-model"] == "isnet-general-use"
    assert response.headers["x-removebg-fallback-used"] == "false"
    assert response.headers["x-edge-quality-candidate"] == "false"
    assert response.headers["x-edge-band-ratio"] == "0.100000"
    assert response.headers["x-edge-band-mid-ratio"] == "0.050000"
    assert response.headers["x-edge-band-low-ratio"] == "0.010000"


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
        headers=auth_headers(),
    )

    assert response.status_code == 422
    body = response.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "LOW_CONFIDENCE_MASK"
    assert body["error"]["details"]["reason"] == "complex_background_low_confidence"


def test_remove_bg_request_validation_error_format(client):
    response = client.post("/remove-bg", headers=auth_headers())

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
        headers=auth_headers(),
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
        headers=auth_headers(),
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



def test_service_info_requires_token_when_header_is_missing(client):
    response = client.get("/service-info")

    assert response.status_code == 401
    assert response.json() == {
        "ok": False,
        "error": {
            "code": "UNAUTHORIZED",
            "message": "缺少或無效的 API Token。",
            "details": {"reason": "missing_authorization_header"},
        },
    }


def test_service_info_requires_bearer_format(client):
    response = client.get("/service-info", headers={"Authorization": "Token test-token"})

    assert response.status_code == 401
    assert response.json()["error"]["details"]["reason"] == "invalid_authorization_scheme"


def test_service_info_rejects_incorrect_token(client):
    response = client.get("/service-info", headers=auth_headers("wrong-token"))

    assert response.status_code == 401
    assert response.json()["error"]["details"]["reason"] == "invalid_api_token"


def test_warmup_requires_correct_token(client):
    response = client.get("/warmup", headers=auth_headers())

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_openapi_remains_public_without_token(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/remove-bg" in response.json()["paths"]


def test_openapi_remove_bg_no_long_query_parameters(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/remove-bg"]["post"]

    assert operation.get("parameters") in (None, [])
    request_body = operation["requestBody"]
    assert "multipart/form-data" in request_body["content"]


def test_openapi_remove_bg_success_response_declares_png_binary(client):
    schema = client.get("/openapi.json").json()
    success_response = schema["paths"]["/remove-bg"]["post"]["responses"]["200"]

    assert "image/png" in success_response["content"]
    assert success_response["content"]["image/png"]["schema"] == {
        "type": "string",
        "format": "binary",
    }

@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("demo.jpg", "image/jpeg"),
        ("demo.jpeg", "image/jpeg"),
        ("demo.png", "image/png"),
        ("demo.webp", "image/webp"),
        ("demo.avif", "image/avif"),
        ("demo.heic", "image/heic"),
        ("demo.heif", "image/heif"),
    ],
)
def test_remove_bg_allowed_extension_and_content_type_can_pass(client, monkeypatch, filename, content_type):
    monkeypatch.setattr(
        routes_module,
        "process_remove_bg",
        lambda *, raw: {
            "image_bytes": b"ok",
            "headers": {
                "X-RemoveBg-Model": "isnet-general-use",
                "X-RemoveBg-Fallback-Used": "false",
                "X-Edge-Quality-Candidate": "false",
                "X-Edge-Band-Ratio": "0.100000",
                "X-Edge-Band-Mid-Ratio": "0.050000",
                "X-Edge-Band-Low-Ratio": "0.010000",
            },
        },
    )

    response = client.post(
        "/remove-bg",
        files={"file": (filename, b"abc", content_type)},
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert response.content == b"ok"


def test_remove_bg_rejects_svg(client, monkeypatch):
    called = {"value": False}

    def fake_process_remove_bg(*, raw: bytes):
        called["value"] = True
        return {"image_bytes": raw, "headers": {}}

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.svg", b"<svg/>", "image/svg+xml")},
        headers=auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid image"
    assert called["value"] is False


def test_remove_bg_rejects_disallowed_extension(client, monkeypatch):
    called = {"value": False}

    def fake_process_remove_bg(*, raw: bytes):
        called["value"] = True
        return {"image_bytes": raw, "headers": {}}

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.gif", b"gif89a", "image/gif")},
        headers=auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid image"
    assert called["value"] is False


def test_remove_bg_rejects_disallowed_content_type(client, monkeypatch):
    called = {"value": False}

    def fake_process_remove_bg(*, raw: bytes):
        called["value"] = True
        return {"image_bytes": raw, "headers": {}}

    monkeypatch.setattr(routes_module, "process_remove_bg", fake_process_remove_bg)

    response = client.post(
        "/remove-bg",
        files={"file": ("demo.png", b"abc", "application/octet-stream")},
        headers=auth_headers(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Invalid image"
    assert called["value"] is False


def test_file_validation_allows_missing_content_type_when_extension_is_allowed():
    for extension in ALLOWED_IMAGE_EXTENSIONS:
        validate_image_filename(f"demo{extension}")

    validate_image_content_type(None)
    validate_image_content_type("")
    validate_image_content_type("   ")


@pytest.mark.parametrize("content_type", sorted(ALLOWED_IMAGE_CONTENT_TYPES))
def test_file_validation_allows_whitelisted_content_types(content_type):
    validate_image_content_type(content_type)



def test_openapi_remove_bg_description_mentions_whitelist_rules(client):
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/remove-bg"]["post"]
    description = operation["description"]

    assert ".jpg" in description
    assert ".heif" in description
    assert "image/avif" in description
    assert "image/svg+xml" in description


def test_openapi_remove_bg_file_field_description_mentions_whitelist_rules(client):
    schema = client.get("/openapi.json").json()
    request_schema = schema["paths"]["/remove-bg"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
    component_name = request_schema["$ref"].split("/")[-1]
    file_property = schema["components"]["schemas"][component_name]["properties"]["file"]
    description = file_property["description"]

    assert ".avif" in description
    assert ".heic" in description
    assert "image/heif" in description
    assert "image/svg+xml" in description
