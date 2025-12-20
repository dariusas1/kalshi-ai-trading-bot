# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY beast_mode_bot.py .
COPY trading_dashboard.py .
COPY main.py .
COPY init_database.py .
COPY sync_kalshi.py .
COPY sync_positions.py .
COPY start_railway.sh .

# Make startup script executable
RUN chmod +x start_railway.sh

# Create logs and data directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose Streamlit port
EXPOSE 8501

# Default command (overridden by railway.toml but good for reference)
CMD ["./start_railway.sh"]
