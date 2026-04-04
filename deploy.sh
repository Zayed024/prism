#!/bin/bash
set -e

# ── CONFIG ──────────────────────────────────────────────────
PROJECT_ID="responsive-amp-438114-j0"
REGION="us-central1"
SERVICE_NAME="prism"
VPC_CONNECTOR="pantry-vpc-connector"  # Reuse existing VPC connector

# Database config (AlloyDB private IP)
ALLOYDB_IP="10.63.208.2"
DB_USER="postgres"
DB_PASSWORD="YOUR_PASSWORD"  # <-- Replace with actual password
DB_NAME="postgres"

# Gemini config
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"  # <-- Replace with actual key
GEMINI_MODEL="gemini-2.5-flash"

DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${ALLOYDB_IP}:5432/${DB_NAME}"

# ── DEPLOY ──────────────────────────────────────────────────
echo "==> Deploying Prism to Cloud Run..."

gcloud run deploy "$SERVICE_NAME" \
    --source=. \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --vpc-connector="$VPC_CONNECTOR" \
    --set-env-vars="DATABASE_URL=${DATABASE_URL},GEMINI_API_KEY=${GEMINI_API_KEY},GEMINI_MODEL=${GEMINI_MODEL}" \
    --memory=1Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=3 \
    --timeout=120

echo ""
echo "==> Deployment complete!"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project="$PROJECT_ID" --region="$REGION" --format="value(status.url)")
echo "==> Live at: $SERVICE_URL"
