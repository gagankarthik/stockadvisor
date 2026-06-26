"""AWS Lambda entry point.

Mangum adapts the ASGI FastAPI app to the Lambda/API Gateway event model. Set
the Lambda handler to ``api.handler.handler``. The same `app` runs locally
under uvicorn (``uvicorn api.main:app``), so there is one code path everywhere.
"""

from __future__ import annotations

from mangum import Mangum

from .main import app

# `api_gateway_base_path="/"` keeps routes identical whether the API is served
# at the stage root or behind a custom domain.
handler = Mangum(app, lifespan="off")
