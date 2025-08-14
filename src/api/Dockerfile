# ---------- Build Stage ----------
FROM python:3.9-slim AS builder

WORKDIR /build

# Install build dependencies (only for compiling Python packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- Final Stage ----------
FROM python:3.9-slim

WORKDIR /code

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create model folder and copy artifacts
RUN mkdir -p /code/models/trained
COPY models/trained/ /code/models/trained/


# Copy application code
COPY . /code

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
