from fastapi import Depends, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_internal_api_token
from app.core.exceptions import ApiError


_bearer_scheme = HTTPBearer(auto_error=False)


def require_internal_api_token(
    raw_authorization: str | None = Header(default=None, alias="Authorization", include_in_schema=False),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    configured_token = get_internal_api_token()
    if not configured_token:
        raise ApiError(
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
            message="服務發生未預期錯誤。",
            details={"reason": "internal_api_token_not_configured"},
        )

    if raw_authorization is None:
        raise ApiError(
            status_code=401,
            code="UNAUTHORIZED",
            message="缺少或無效的 API Token。",
            details={"reason": "missing_authorization_header"},
        )

    if credentials is None:
        raise ApiError(
            status_code=401,
            code="UNAUTHORIZED",
            message="缺少或無效的 API Token。",
            details={"reason": "invalid_authorization_scheme"},
        )

    provided_token = credentials.credentials.strip()
    if not provided_token or provided_token != configured_token:
        raise ApiError(
            status_code=401,
            code="UNAUTHORIZED",
            message="缺少或無效的 API Token。",
            details={"reason": "invalid_api_token"},
        )
