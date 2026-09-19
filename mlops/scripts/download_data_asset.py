"""
Downloads a registered Azure ML data asset's files locally.
Replaces `az ml data download`, which isn't available in every CLI
version -- this works consistently via the Python SDK instead.
Uses AzureCliCredential, which picks up whatever identity
`azure/login@v2` (or a local `az login`) already established --
no separate auth needed.
"""
import argparse
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription_id", required=True)
    parser.add_argument("--resource_group", required=True)
    parser.add_argument("--workspace_name", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--download_path", required=True)
    args = parser.parse_args()

    ml_client = MLClient(
        AzureCliCredential(),
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    ml_client.data.download(name=args.name, version=args.version, download_path=args.download_path)
    print(f"Downloaded {args.name}:{args.version} -> {args.download_path}")


if __name__ == "__main__":
    main()