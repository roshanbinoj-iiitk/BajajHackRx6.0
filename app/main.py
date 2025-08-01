from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.auth_middleware import AuthMiddleware
from app.api import router
from app.create_api_key import api_key_router

app = FastAPI(
    title="HackRx API",
    description="API for HackRx application",
    version="0.0.469"
)

# app.add_middleware(AuthMiddleware)

app.include_router(api_key_router)
app.include_router(router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    print(exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": request.url.path
            }
        },
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

