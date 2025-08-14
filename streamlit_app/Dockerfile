# ===========================
# Stage 1: Build dependencies
# ===========================
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build tools (only in build stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc=4:10.2.1-1 \
    make=4.3-4.1 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

# Install dependencies into a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# ===========================
# Stage 2: Final lightweight image
# ===========================
FROM python:3.9-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /venv /venv

# Make sure Python uses venv by default
ENV PATH="/venv/bin:$PATH"

# Copy only necessary app files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
