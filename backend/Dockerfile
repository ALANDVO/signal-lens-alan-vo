FROM python:3.12-slim

WORKDIR /app

# Create unprivileged nonroot user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure app directory permissions
RUN mkdir -p /data && chown -R appuser:appgroup /app /data

USER appuser

EXPOSE 8000

ENV APP_ENV=production
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
