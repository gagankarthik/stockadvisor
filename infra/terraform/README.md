# MarketDesk — Terraform deploy

Provisions the whole serverless stack on AWS:

- **ECR** repos for the two container images, built & pushed by `terraform apply`
- **S3** artifact bucket (encrypted, private) — the model + snapshot live here
- **API Lambda** (FastAPI via Mangum) behind a public **Function URL**
- **Training Lambda** (more memory/timeout) on a daily **EventBridge** schedule
- Shared least-privilege **IAM** role (logs + this bucket only)

No SageMaker; both heavy ML steps run as Lambda **container images** (the only
Lambda packaging that fits pandas/scikit-learn/optuna).

```
            terraform apply
                  │  builds + pushes images, creates everything
   ┌──────────────┼───────────────────────────┐
   ▼              ▼                             ▼
 ECR (api,    S3 artifacts            EventBridge (cron)
  train)      ▲        ▲                     │ invokes
              │ reads  │ writes              ▼
        API Lambda   Train Lambda  ◄─────────┘
        (Func URL)
```

## Prerequisites

- Terraform ≥ 1.6, AWS CLI configured (`aws configure`), and **Docker running**.
- On Windows run from **Git Bash** or **WSL** (the image build uses `bash`).
- IAM permissions to manage ECR, Lambda, S3, IAM, EventBridge, CloudWatch Logs.

## Deploy

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # add your API keys

terraform init
terraform apply -var image_tag=$(git rev-parse --short HEAD)
```

`apply` builds and pushes both images, then stands up the stack. On completion:

```bash
terraform output api_url        # -> set as NEXT_PUBLIC_API_BASE_URL in the UI
```

Seed the first snapshot (otherwise the API returns 503 until the schedule fires):

```bash
aws lambda invoke --function-name marketdesk-train /dev/stdout
```

## Updating

Code changed? Re-apply with a **new tag** so the Lambdas pick it up:

```bash
terraform apply -var image_tag=$(git rev-parse --short HEAD)
```

(Reusing the same tag won't trigger a Lambda update — the image URI is unchanged.)

## CI (GitHub Actions)

A manual workflow — `.github/workflows/deploy-terraform.yml` — runs this stack
with your repo's AWS secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION`) and the provider keys. Trigger it from the **Actions** tab
("Run workflow"); pass `destroy: true` to tear the stack down.

> **Important:** for repeatable CI runs, enable the **S3 remote backend** in
> `versions.tf` first. Without it, Terraform state lives only on the ephemeral
> runner, so a second run starts blank and collides with the already-created
> resources. Local runs keep state in this folder and are fine as-is.

Pick **one** deploy path. This Terraform workflow is `workflow_dispatch`-only so
it never double-deploys alongside the SAM workflow (`deploy.yml`, which runs on
push to `main`). If you standardize on Terraform, disable or delete `deploy.yml`.

## Notes

- **Production hardening**: enable the S3 remote backend in `versions.tf`, set
  `cors_origin` to your UI domain, and consider API Gateway + a WAF / authorizer
  instead of a public Function URL if the data must be access-controlled.
- **CI**: the same two commands run on a Linux runner (Docker preinstalled).
  Use GitHub OIDC to assume a deploy role instead of static keys.
- Secrets are passed as `-var` and stored in Terraform state — use the encrypted
  S3 backend, or swap to AWS Secrets Manager + `aws_lambda_function` env refs.
