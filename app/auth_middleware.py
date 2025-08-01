# app/auth_middleware.py
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from app.db import SessionLocal, APIKey

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip check for health and key generation
        if request.url.path in ["/health", "/generate-api-key"]:
            return await call_next(request)

        api_key = request.headers.get("x-api-key")
        if not api_key:
            return self._unauthorized(request)

        db = SessionLocal()
        exists = db.query(APIKey).filter(APIKey.key == api_key).first()
        db.close()

        if not exists:
            return self._unauthorized(request)

        return await call_next(request)

    def _unauthorized(self, request):
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "error": {
                    "code": 401,
                    "message": "Missing or invalid API Key",
                    "path": str(request.url.path)
                }
            }
        )
