FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies + MS ODBC Driver 18 (no apt-key)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg \
    unixodbc unixodbc-dev \
    gcc g++ \
  && mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
     | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg \
  && chmod 644 /etc/apt/keyrings/microsoft.gpg \
  && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
     > /etc/apt/sources.list.d/mssql-release.list \
  && apt-get update \
  && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8020
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8020"]