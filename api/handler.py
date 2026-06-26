"""AWS Lambda entry point.

One function serves two roles, dispatched by event shape:

* HTTP requests (Lambda Function URL) -> the FastAPI app via Mangum.
* EventBridge scheduled events        -> the training/snapshot refresh job.

Mangum adapts the ASGI FastAPI app to the Lambda event model. The same `app`
runs locally under uvicorn (``uvicorn api.main:app``), so there is one code
path everywhere. Set the Lambda handler to ``api.handler.handler``.
"""

from __future__ import annotations

from mangum import Mangum

from .main import app

# `lifespan="off"` because there is no ASGI startup/shutdown work to run.
_http_handler = Mangum(app, lifespan="off")


def _is_scheduled_event(event) -> bool:
    """True for an EventBridge/CloudWatch scheduled invocation (the refresh job).

    HTTP invocations from the Function URL carry a `requestContext`/`http` shape;
    scheduled events come from `aws.events` with detail-type "Scheduled Event".
    """
    if not isinstance(event, dict):
        return False
    if event.get("source") == "aws.events":
        return True
    if event.get("detail-type") == "Scheduled Event":
        return True
    # Explicit manual invoke escape hatch: {"job": "refresh"}.
    return event.get("job") == "refresh"


def handler(event, context):  # noqa: ANN001
    if _is_scheduled_event(event):
        # Imported lazily so cold starts for HTTP requests don't pay for it.
        from train import lambda_handler as run_refresh

        return run_refresh(event, context)
    return _http_handler(event, context)
