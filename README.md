# ecom-ml-platform

Three folders, three owners. The folder boundary IS the responsibility
boundary — this is the actual point of the structure.

## `ds_notebook/`
What Data Science runs, on their own compute instance, in their own
notebook. Exploratory in style, never touched by CI/CD. Imports the
SAME feature logic the platform pipeline uses (from
`platform/feature_components/`) so DS validates against exactly what
production will compute — no train/serve skew from two implementations
of the same features.

DS's own responsibility here: churn window definition, model choice,
hyperparameters. They register their own model, tagged `stage: dev`.

## `platform/`
Shared, reusable, written once. Never edited per model.
- `data_platform/` — bronze -> silver curation. Generic across every model.
- `feature_components/<model>/` — model-specific gold-layer feature
  logic. Lives here (not in `ds_notebook/` or `mlops/`) because BOTH
  the DS notebook and the platform's retrain pipeline import it — one
  implementation, two consumers.
- `components/` — preprocess/train/evaluate/register. Used only by
  Flow 2 (see below), the platform's own retraining capability.
- `environments/` — one shared runtime spec for every component.

## `mlops/`
My responsibility. Two DISTINCT flows live here — do not conflate them:

**Flow 1 — DS handoff promotion** (`generate_config_from_run.py` +
`validate_and_promote.py`, triggered by `promote-churn-model.yml`).
DS hands off a run_id. `generate_config_from_run.py` looks up which
model+version that run registered and writes `config.yml`. Pushing
that file triggers CI, which loads THAT EXACT registered model,
re-scores it against a platform-controlled validation set (never
trusts DS's own reported number), and if it passes, re-tags the
SAME artifact to the next stage. No training happens in this flow.

**Flow 2 — platform-owned retraining** (`pipelines/`, `scripts/submit_pipeline.sh`,
triggered by `train-churn-model.yml`). Trains a model from scratch
using the shared `platform/components/`. Independent of any DS
handoff — this is what a scheduled automated retrain would call, or
what runs if the platform is trusted to retrain without a human
validating each run.

Both flows read/write `models/<name>/config.yml`, but Flow 1 never
triggers Flow 2's workflow and vice versa — their path filters in
`.github/workflows/` are deliberately non-overlapping.

## Onboarding a new model

1. Add model-specific feature logic to `platform/feature_components/<new_model>/`.
2. DS explores in `ds_notebook/`, importing that feature module, registers to `dev`.
3. Run `generate_config_from_run.py` against their run_id.
4. `mlops/models/<new_model>/config.yml` now exists — Flow 1's promotion
   workflow needs a matching trigger file and target validation data asset.
   `platform/components/` and `data_platform/` need zero changes.




Create the ADLS Gen2 storage account


az storage account create --name ecomadlsgen2vt01 --resource-group ecom-mlops-rg --location eastus --sku Standard_LRS --kind StorageV2 --hierarchical-namespace true


az storage fs create --account-name ecomadlsgen2vt01 --name raw --auth-mode login

az storage fs create --account-name ecomadlsgen2vt01 --name silver --auth-mode login

az storage fs create --account-name ecomadlsgen2vt01 --name feature-store --auth-mode login


upload raw dataset to adls:

az storage fs file upload --account-name ecomadlsgen2vt01 --file-system raw --source data/raw/Online_Retail.csv --path Online_Retail.csv --auth-mode login



az ml job create -f mlops/pipelines/feature_pipeline.yml --stream


az role assignment create --assignee e1ef69de-f2ba-48f7-b255-9436810a164d --role "Storage Blob Data Contributor" --scope "/subscriptions/345f72c2-10bd-4dbf-95ab-4cd43df6adb7/resourceGroups/ecom-mlops-rg/providers/Microsoft.Storage/storageAccounts/ecomadlsgen2vt01"


get user id :
az ad signed-in-user show --query id -o tsv



role assihnemt for stotage account:


az role assignment list --assignee 25de3425-94f7-4579-be10-a14b6a95b401 --scope "/subscriptions/345f72c2-10bd-4dbf-95ab-4cd43df6adb7/resourceGroups/ecom-mlops-rg/providers/Microsoft.Storage/storageAccounts/ecomadlsgen2vt01" --output table


az role assignment create --assignee 25de3425-94f7-4579-be10-a14b6a95b401 --role "Storage Blob Data Contributor" --scope "/subscriptions/345f72c2-10bd-4dbf-95ab-4cd43df6adb7/resourceGroups/ecom-mlops-rg/providers/Microsoft.Storage/storageAccounts/ecomadlsgen2vt01"
