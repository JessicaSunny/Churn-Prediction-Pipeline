FROM python:3.12-slim

WORKDIR /code

# Install dependencies first (separate layer = Docker caches this when only
# your code changes, not your dependencies -- much faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual app code and pre-trained model
COPY app/ ./app/
COPY model/ ./model/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
