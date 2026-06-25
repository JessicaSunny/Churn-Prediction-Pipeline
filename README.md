# Customer Churn Prediction — End-to-End ML Pipeline

A churn prediction model served as a REST API, containerized, with automated
testing (CI) and a path to AWS deployment (CD). Built as a learning project
to demonstrate the full ML deployment pipeline: Git → CI → Docker → AWS → monitoring.

## Architecture

```
Customer data → train_model.py → model/model.pkl
                                        │
                                        ▼
                              app/main.py (FastAPI)
                                        │
                              ┌─────────┴─────────┐
                          Dockerfile          GitHub Actions
                          (container)          (CI: runs tests
                                │               on every push)
                                ▼
                        AWS ECR (image registry)
                                │
                                ▼
                   AWS Lambda + API Gateway (live endpoint)
                                │
                                ▼
                      CloudWatch (logs + monitoring)
```

## What's in this repo

| File | Purpose |
|---|---|
| `train_model.py` | Trains the model, saves `model/model.pkl` |
| `app/main.py` | FastAPI app serving `/predict` and `/health` |
| `tests/test_api.py` | Automated tests — run locally with `pytest`, and automatically in CI |
| `Dockerfile` | Packages the app + model into a container |
| `.github/workflows/ci.yml` | Runs tests automatically on every push to GitHub |
| `generate_synthetic_data.py` | Creates placeholder data — **replace with the real dataset** (see below) |

## 1. Use the real dataset

Right now `data/telco_churn.csv` is synthetic data I generated so the pipeline
could be built and tested end-to-end. Replace it with the real one:

1. Download from [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Save it as `data/telco_churn.csv` (same filename, same columns — nothing else changes)
3. Re-run `python train_model.py`

## 2. Run it locally (no Docker yet)

```bash
pip install -r requirements-dev.txt
python train_model.py              # trains the model
uvicorn app.main:app --reload      # starts the API on localhost:8000
```

In another terminal:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Run the test suite:
```bash
pytest tests/ -v
```

## 3. Run it in Docker

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```
Same `curl` commands as above should now hit the containerized version. If
this works, the container will behave identically on AWS — that's the entire
point of containerizing it.

## 4. Push to GitHub → watch CI run

```bash
git init
git add .
git commit -m "Initial churn prediction pipeline"
git remote add origin <your-repo-url>
git push -u origin main
```

Go to the **Actions** tab on GitHub. You'll see the `CI` workflow run
automatically — it installs dependencies, runs `pytest`, and (if that passes)
builds the Docker image. This is your CI pipeline, live.

Try breaking a test on purpose (e.g. change `assert response.status_code == 200`
to `204`), push it, and watch CI fail. Then fix it and watch it go green again.
This is the fastest way to *feel* what CI is actually doing.

## 5. Deploy to AWS (the CD part)

These steps you'll run yourself with the AWS CLI configured locally —
deliberately not automated in CI yet, since that needs your AWS credentials
stored as GitHub Secrets first.

**a) Push the image to ECR:**
```bash
aws ecr create-repository --repository-name churn-prediction-api
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com

docker tag churn-prediction-api:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/churn-prediction-api:latest
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/churn-prediction-api:latest
```

**b) Create the Lambda function from that image:**
```bash
aws lambda create-function \
  --function-name churn-prediction-api \
  --package-type Image \
  --code ImageUri=<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/churn-prediction-api:latest \
  --role <your-lambda-execution-role-arn> \
  --timeout 30 \
  --memory-size 512
```
Note: Lambda needs a small adapter to translate API Gateway events into ASGI
calls FastAPI understands. The cleanest way is `pip install mangum`, then in
`app/main.py` add:
```python
from mangum import Mangum
handler = Mangum(app)
```
and reference `app.main.handler` (not `app.main.app`) as the Lambda handler.

**c) Front it with API Gateway** so you get a public URL — easiest done via
the AWS Console: API Gateway → Create API → HTTP API → point it at your
Lambda function → deploy. You'll get a URL like
`https://abc123.execute-api.<region>.amazonaws.com/predict`.

**d) Automate steps a–b in CI** once this works manually: add an `aws-deploy`
job to `ci.yml` (needs: build), store your AWS credentials as GitHub Secrets
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), and use
`aws-actions/amazon-ecr-login` and `aws-actions/configure-aws-credentials`.
This is what turns "CI" into full "CI/CD" — deployment becomes automatic too.

## 6. Monitoring

Once deployed, Lambda logs land in **CloudWatch Logs** automatically — no
setup needed. Worth adding next:
- A custom CloudWatch metric counting predictions per risk tier (`low`/`medium`/`high`)
- A CloudWatch Alarm if error rate or latency spikes
- Periodically re-running `train_model.py` against fresh data and comparing
  ROC-AUC to the previous version — this is the start of detecting **model drift**

## For your CV / interview talking points

- "Built and deployed an end-to-end ML pipeline for customer churn prediction"
- "Containerized model served via AWS Lambda/API Gateway"
- "CI/CD via GitHub Actions — automated testing gates every deployment"
- "Designed for retention targeting — churn risk tiers (low/medium/high) map directly to campaign prioritization"
