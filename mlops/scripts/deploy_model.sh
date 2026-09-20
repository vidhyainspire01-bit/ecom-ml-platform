#!/usr/bin/env bash
# Usage: deploy_model.sh <target_stage> <registered_model_name> <model_version> <platform_model_name>
set -euo pipefail

TARGET_STAGE="$1"
MODEL_NAME="$2"
MODEL_VERSION="$3"
PLATFORM_MODEL_NAME="$4"

ENDPOINT_SLUG=$(echo "$PLATFORM_MODEL_NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
ENDPOINT_NAME="${ENDPOINT_SLUG}-${TARGET_STAGE}"
DEPLOYMENT_NAME="blue"

echo "Deploying ${MODEL_NAME}:${MODEL_VERSION} to endpoint '${ENDPOINT_NAME}', deployment '${DEPLOYMENT_NAME}'"

if ! az ml online-endpoint show --name "$ENDPOINT_NAME" >/dev/null 2>&1; then
  echo "Endpoint ${ENDPOINT_NAME} doesn't exist yet, creating it..."
  sed "s/PLACEHOLDER_ENDPOINT_NAME/${ENDPOINT_NAME}/" mlops/deployment/endpoint.yml > /tmp/endpoint.yml
  az ml online-endpoint create -f /tmp/endpoint.yml
else
  echo "Endpoint ${ENDPOINT_NAME} already exists, reusing it."
fi

SCORING_ABS_PATH="$(pwd)/mlops/deployment/scoring"

sed \
  -e "s/PLACEHOLDER_DEPLOYMENT_NAME/${DEPLOYMENT_NAME}/" \
  -e "s/PLACEHOLDER_ENDPOINT_NAME/${ENDPOINT_NAME}/" \
  -e "s/PLACEHOLDER_MODEL_NAME/${MODEL_NAME}/" \
  -e "s/PLACEHOLDER_MODEL_VERSION/${MODEL_VERSION}/" \
  -e "s/PLACEHOLDER_PLATFORM_MODEL_NAME/${PLATFORM_MODEL_NAME}/" \
  -e "s|PLACEHOLDER_ABSOLUTE_CODE_PATH|${SCORING_ABS_PATH}|" \
  mlops/deployment/deployment.yml > mlops/deployment/_rendered_deployment.yml


if az ml online-deployment show --name "$DEPLOYMENT_NAME" --endpoint-name "$ENDPOINT_NAME" >/dev/null 2>&1; then
  echo "Deployment ${DEPLOYMENT_NAME} already exists -- deleting and recreating (brief downtime, no blue/green yet)"
  az ml online-deployment delete --name "$DEPLOYMENT_NAME" --endpoint-name "$ENDPOINT_NAME" --yes
fi

az ml online-deployment create -f mlops/deployment/_rendered_deployment.yml --all-traffic

echo "Deployed. Endpoint scoring URI:"
SCORING_URI=$(az ml online-endpoint show --name "$ENDPOINT_NAME" --query scoring_uri -o tsv)
echo "$SCORING_URI"

FEATURES=$(python -c "import yaml; print(yaml.safe_load(open('mlops/models/${PLATFORM_MODEL_NAME}/promotion_config.yml'))['features'])")
SAMPLE_ROW=$(python3 -c "
import csv
features = '${FEATURES}'.split(',')
with open('validation_data/validation.csv') as f:
    row = next(csv.DictReader(f))
    print(','.join(row[c] for c in features))
")
API_KEY=$(az ml online-endpoint get-credentials --name "$ENDPOINT_NAME" --query primaryKey -o tsv)

python mlops/scripts/smoke_test_endpoint.py \
  --scoring_uri "$SCORING_URI" \
  --api_key "$API_KEY" \
  --features "$FEATURES" \
  --sample_values "$SAMPLE_ROW"