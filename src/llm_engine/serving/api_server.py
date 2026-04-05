
"""
FastAPI server module for the LLM inference engine serving layer.

Exposes a REST API with a /generate endpoint for text generation
and a /health endpoint for liveness checks. Uses a factory function
pattern to inject the Router dependency.
"""

import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .router import Router

class GenerateRequest(BaseModel):
    
    """
    Description:
        Pydantic model for the /generate endpoint request body.
    Attributes:
        prompt (str): The input text to generate from.
        max_tokens (int): Maximum number of tokens to generate.
    """
    
    prompt: str
    max_tokens: int
        
def create_app(router: Router, 
               max_concurrent_requests: int = 256,
               request_timeout: int = 30,
    ) -> FastAPI:
    
    """
    Description:
        Factory function that creates a FastAPI application with the given Router.
    Args:
        router (Router): The router instance that handles generation requests.
    Returns:
        FastAPI: The configured FastAPI application.
    """

    app = FastAPI()
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    @app.on_event("startup")
    async def startup_event():
        router.start()
    
    @app.post("/generate")
    async def generate(request: GenerateRequest):
        if semaphore.locked():
            raise HTTPException(status_code=503, detail="Server at capacity")
        async with semaphore:
            try:
                response = await asyncio.wait_for(router.generate(request.prompt, request.max_tokens),
                                                  timeout = request_timeout
                                )
                return response
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Request timed out")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
    