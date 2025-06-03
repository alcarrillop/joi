import os

from fastapi import FastAPI

app = FastAPI(
    title="Joi - English Learning Assistant",
    description="WhatsApp AI agent for English learning with memory and progress tracking",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "message": "Joi - English Learning Assistant API",
        "status": "running",
        "docs": "/docs",
        "debug": "/debug/health",
    }


@app.get("/health")
async def simple_health_check():
    """Simple health check that doesn't require database or external services."""
    return {
        "status": "healthy",
        "message": "Service is running",
        "environment": os.getenv("RAILWAY_ENVIRONMENT", "unknown"),
    }


# Only load heavy dependencies after the basic app is set up
@app.on_event("startup")
async def load_routers():
    """Load routers on startup to avoid import-time dependency issues."""
    try:
        from agent.interfaces.debug.debug_endpoints import debug_router
        from agent.interfaces.whatsapp.whatsapp_response import whatsapp_router

        app.include_router(whatsapp_router)
        app.include_router(debug_router)
    except Exception as e:
        print(f"Warning: Could not load some routers: {e}")
        # App will still start with basic health check
