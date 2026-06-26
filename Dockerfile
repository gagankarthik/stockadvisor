# MarketDesk — single AWS Lambda container image (FastAPI via Mangum + the
# scheduled refresh job). The handler in api/handler.py dispatches by event:
# HTTP requests -> FastAPI, EventBridge schedule -> train.lambda_handler.
#
# Build & deploy:
#   aws ecr get-login-password | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
#   docker build -t marketdesk-api .
#   docker tag marketdesk-api:latest <acct>.dkr.ecr.<region>.amazonaws.com/marketdesk-api:latest
#   docker push <acct>.dkr.ecr.<region>.amazonaws.com/marketdesk-api:latest
# Then point a Lambda (function URL or API Gateway) at the image.

FROM public.ecr.aws/lambda/python:3.12

# Install dependencies into the Lambda task root.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t "${LAMBDA_TASK_ROOT}"

# Application code: the API, the shared library, and the training entry point
# (imported by the handler for scheduled refreshes).
COPY marketdesk ${LAMBDA_TASK_ROOT}/marketdesk
COPY api ${LAMBDA_TASK_ROOT}/api
COPY train.py ${LAMBDA_TASK_ROOT}/train.py

# Lambda handler: module.function
CMD [ "api.handler.handler" ]
