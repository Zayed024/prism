#!/bin/bash
set -e

# ── CONFIG ──────────────────────────────────────────────────
PROJECT_ID="responsive-amp-438114-j0"
REGION="us-east4"  # Cloud Run region (must match AlloyDB region)
SERVICE_NAME="prism"
VPC_CONNECTOR="pantry-vpc-connector"  # Reuse existing VPC connector

# Database config (AlloyDB private IP)
ALLOYDB_IP="10.63.208.2"
DB_USER="postgres"
DB_PASSWORD="YOUR_PASSWORD"  # <-- Replace with actual password
DB_NAME="postgres"

# Gemini config — use Vertex AI on Cloud Run for higher rate limits
GEMINI_MODEL="gemini-3-flash-preview"
# Vertex AI location — use 'global' for Gemini 3 Flash Preview (it's not in regional endpoints)
VERTEX_LOCATION="global"

DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${ALLOYDB_IP}:5432/${DB_NAME}"

# Enable Vertex AI API
echo "==> Enabling Vertex AI API..."
gcloud services enable aiplatform.googleapis.com --project="$PROJECT_ID"

# ── DEPLOY ──────────────────────────────────────────────────
echo "==> Deploying Prism to Cloud Run..."

gcloud run deploy "$SERVICE_NAME" \
    --source=. \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --platform=managed \
    --allow-unauthenticated \
    --vpc-connector="$VPC_CONNECTOR" \
    --set-env-vars="DATABASE_URL=${DATABASE_URL},GEMINI_MODEL=${GEMINI_MODEL},GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${VERTEX_LOCATION}" \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=3 \
    --timeout=300 \
    --concurrency=10

echo ""
echo "==> Deployment complete!"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --project="$PROJECT_ID" --region="$REGION" --format="value(status.url)")
echo "==> Live at: $SERVICE_URL"
