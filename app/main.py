import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.api.v1.king_county import router as king_county_router
from app.api.v1.dataset_rag import router as dataset_rag_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="King County Agentic Backend",
        description="FastAPI backend exposing an agentic LLM system for King County and a dataset RAG API.",
        version="0.2.0",
    )

    # CORS – allow any origin so the frontend can be used from anywhere (public access).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    async def health_check() -> dict:
        return {"status": "ok"}

    # Versioned API routes
    app.include_router(king_county_router, prefix="/v1/king-county", tags=["king_county"])
    app.include_router(dataset_rag_router, prefix="/v1/dataset-rag", tags=["dataset_rag"])

    # Serve frontend publicly at /app/ (anyone can open the app in a browser).
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _frontend_dir = os.path.join(_project_root, "frontend")
    if os.path.isdir(_frontend_dir):
        app.mount("/app", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return RedirectResponse(url="/app/", status_code=302)

    return app


app = create_app()

