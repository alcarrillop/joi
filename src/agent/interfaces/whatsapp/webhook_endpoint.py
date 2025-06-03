from fastapi import FastAPI

from agent.interfaces.whatsapp.whatsapp_response import whatsapp_router
from agent.interfaces.debug.debug_endpoints import debug_router

app = FastAPI(
    title="Joi - English Learning Assistant",
    description="WhatsApp AI agent for English learning with memory and progress tracking",
    version="1.0.0"
)

app.include_router(whatsapp_router)
app.include_router(debug_router)

@app.get("/")
async def root():
    return {
        "message": "Joi - English Learning Assistant API",
        "status": "running",
        "docs": "/docs",
        "debug": "/debug/health"
    }
