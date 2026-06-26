# MarketDesk API — AWS Lambda container image (FastAPI via Mangum).
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

# Application code.
COPY marketdesk ${LAMBDA_TASK_ROOT}/marketdesk
COPY api ${LAMBDA_TASK_ROOT}/api

# Lambda handler: module.function
CMD [ "api.handler.handler" ]
