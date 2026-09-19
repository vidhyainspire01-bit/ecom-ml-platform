#!/usr/bin/env bash
# Usage: submit_pipeline.sh <model_name> <git_sha> <data_version>
# Run from the REPO ROOT (paths below are repo-root-relative).
# Reads mlops/models/<model_name>/config.yml and submits the shared
# platform pipeline with those values overridden via --set.
set -euo pipefail

MODEL_NAME="$1"
GIT_SHA="$2"
DATA_VERSION="$3"
CONFIG_PATH="mlops/models/${MODEL_NAME}/config.yml"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "No config found at $CONFIG_PATH" >&2
  exit 1
fi

read_field() {
  python3 -c "import yaml,sys; print(yaml.safe_load(open('$CONFIG_PATH'))['$1'])"
}

RAW_DATA_PATH=$(read_field raw_data_path)
FEATURES=$(read_field features)
TARGET_COLUMN=$(read_field target_column)
MODEL_TYPE=$(read_field model_type)
HYPERPARAMETERS=$(read_field hyperparameters)
METRIC_NAME=$(read_field metric_name)
MIN_THRESHOLD=$(read_field min_threshold)

az ml job create \
  -f mlops/pipelines/train_pipeline.yml \
  --set inputs.raw_data.path="${RAW_DATA_PATH}" \
  --set inputs.features="${FEATURES}" \
  --set inputs.target_column="${TARGET_COLUMN}" \
  --set inputs.model_type="${MODEL_TYPE}" \
  --set inputs.hyperparameters="${HYPERPARAMETERS}" \
  --set inputs.metric_name="${METRIC_NAME}" \
  --set inputs.min_threshold="${MIN_THRESHOLD}" \
  --set inputs.model_name="${MODEL_NAME}" \
  --set inputs.git_sha="${GIT_SHA}" \
  --set inputs.data_version="${DATA_VERSION}" \
  --stream
