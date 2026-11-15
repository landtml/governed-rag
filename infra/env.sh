# Source this file:  source infra/env.sh
# Exports the settings from .env.azure (written by infra/deploy.sh).
#
# The application fetches its own Microsoft Entra token for PostgreSQL on every
# new connection, so it needs nothing more. PGPASSWORD is exported only for
# command-line tools such as psql; that token is valid for about an hour, so
# source this file again to renew it.
set -a
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/../.env.azure"
set +a
PGPASSWORD=$(az account get-access-token --resource-type oss-rdbms --query accessToken -o tsv)
export PGPASSWORD
