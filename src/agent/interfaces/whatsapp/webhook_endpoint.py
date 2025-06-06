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


@app.get("/debug/db-test")
async def test_database_connection():
    """Test database connectivity for debugging."""
    import asyncpg

    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"status": "error", "message": "DATABASE_URL not set"}

        # Log the URL format (without credentials)
        url_info = database_url.split("@")[1] if "@" in database_url else "URL format issue"

        # Check for potential encoding issues
        has_special_chars = any(char in database_url for char in ["@", "#", "$", "%", "&", "+", "/", "?"])
        protocol = database_url.split("://")[0] if "://" in database_url else "unknown"

        # Test basic connection with asyncpg
        conn = await asyncpg.connect(database_url, timeout=30)

        # Test a simple query
        test_query_result = await conn.fetchval("SELECT 1")
        # Test table creation capability
        await conn.execute("SELECT NOW()")

        await conn.close()

        return {
            "status": "success",
            "message": "Database connection successful with asyncpg",
            "host_info": url_info,
            "test_query": test_query_result,
            "protocol": protocol,
            "has_special_chars": has_special_chars,
        }

    except Exception as e:
        import traceback

        return {
            "status": "error",
            "message": f"Database connection failed: {str(e)}",
            "error_type": type(e).__name__,
            "url_format": url_info if "url_info" in locals() else "Could not parse URL",
            "full_error": traceback.format_exc()[-500:],  # Last 500 chars of traceback
        }


# Only load heavy dependencies after the basic app is set up
@app.on_event("startup")
async def load_routers():
    """Load routers on startup to avoid import-time dependency issues."""
    try:
        # from agent.interfaces.debug.debug_endpoints import debug_router  # Temporarily disabled for migration
        from agent.interfaces.whatsapp.whatsapp_response import whatsapp_router

        app.include_router(whatsapp_router)
        # app.include_router(debug_router)  # Temporarily disabled for migration
    except Exception as e:
        print(f"Warning: Could not load some routers: {e}")
        # App will still start with basic health check
