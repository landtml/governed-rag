#!/usr/bin/env bash
# Delete everything the Azure deployment created: the whole resource group.
# The subscription-level budget is kept; remove it with
#   az consumption budget delete --budget-name governed-rag-monthly
set -euo pipefail
RG="${RG:-rg-governed-rag}"
read -r -p "Delete resource group $RG and everything in it? [y/N] " answer
[[ "$answer" == "y" ]] || { echo "aborted"; exit 1; }
az group delete -n "$RG" --yes
rm -f "$(dirname "$0")/../.env.azure"
echo "deleted $RG"
