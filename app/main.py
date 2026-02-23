from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.king_county import router as king_county_router
from app.api.v1.dataset_rag import router as dataset_rag_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="King County Agentic Backend",
        description="FastAPI backend exposing an agentic LLM system for King County and a dataset RAG API.",
        version="0.2.0",
    )

    # CORS – allow localhost during development; tighten for production.
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

    return app


app = create_app()

