"""
SHARED, model-agnostic smoke test -- confirms a newly deployed
endpoint actually answers a real scoring request correctly BEFORE
it's trusted with production traffic. Takes the feature columns and
a sample row from the model's own promotion_config.yml, so this file
never needs per-model changes.
"""
import argparse
import json
import sys
import requests


def run_smoke_test(scoring_uri: str, api_key: str, features: list, sample_values: list) -> bool:
    payload = {"data": [sample_values], "columns": features}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(scoring_uri, headers=headers, data=json.dumps(payload), timeout=30)
    except requests.RequestException as e:
        print(f"SMOKE TEST FAILED: request error: {e}")
        return False

    if resp.status_code != 200:
        print(f"SMOKE TEST FAILED: HTTP {resp.status_code}: {resp.text}")
        return False

    try:
        result = json.loads(resp.json()) if isinstance(resp.json(), str) else resp.json()
    except (json.JSONDecodeError, ValueError) as e:
        print(f"SMOKE TEST FAILED: response not valid JSON: {e}")
        return False

    if "predictions" not in result:
        print(f"SMOKE TEST FAILED: no 'predictions' key in response: {result}")
        return False

    print(f"SMOKE TEST PASSED: {result}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scoring_uri", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--features", required=True, help="Comma-separated column names")
    parser.add_argument("--sample_values", required=True, help="Comma-separated numeric values, matching --features order")
    args = parser.parse_args()

    features = args.features.split(",")
    sample_values = [float(v) for v in args.sample_values.split(",")]

    passed = run_smoke_test(args.scoring_uri, args.api_key, features, sample_values)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()