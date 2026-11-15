// Phase 1: the data and model layer of governed-rag on Azure.
//
//   * Azure Database for PostgreSQL Flexible Server, Burstable B1ms, with the
//     `vector` extension allowlisted and Microsoft Entra ID as the only way in
//     (no password exists).
//   * An Azure AI Foundry resource (kind AIServices) with local (key) auth
//     disabled, holding three pinned deployments: a generator, a judge and an
//     embedder. Callers authenticate with Entra ID tokens.
//
// Deploy with infra/deploy.sh, which fills in the caller's identity and IP.
// infra/app.bicep adds the container app on top of this layer.
targetScope = 'resourceGroup'

param location string = resourceGroup().location

@description('Region for Postgres, when it should differ from the model region.')
param pgLocation string = location

@description('Short lowercase suffix that makes globally unique names (server host name, AI subdomain).')
param suffix string = uniqueString(resourceGroup().id)

@description('Object ID of the Entra user who administers Postgres and calls the models.')
param adminObjectId string

@description('User principal name of that user, as Postgres sees the Entra admin role.')
param adminPrincipalName string

@description('Public IPv4 address allowed through the Postgres firewall (the developer machine).')
param clientIp string

@description('Tokens-per-minute capacity, in thousands, for each deployment.')
param generatorCapacity int = 300
param judgeCapacity int = 200
param embedderCapacity int = 100

@description('Availability zone for Postgres, or empty to let Azure choose.')
param pgZone string = ''

@description('False skips Postgres, so the model layer can be deployed on its own.')
param deployPg bool = true

@description('False skips the model layer, so Postgres can be deployed on its own.')
param deployAi bool = true

var tags = { project: 'governed-rag' }

// ---------------------------------------------------------------- Postgres

resource pg 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = if (deployPg) {
  name: 'pg-governed-rag-${uniqueString(resourceGroup().id, pgLocation)}'
  location: pgLocation
  tags: tags
  sku: { name: 'Standard_B1ms', tier: 'Burstable' }  // the free-account SKU
  properties: {
    version: '17'
    availabilityZone: empty(pgZone) ? null : pgZone
    storage: { storageSizeGB: 32, autoGrow: 'Disabled' }  // 32 GB is the free-account storage
    backup: { backupRetentionDays: 7, geoRedundantBackup: 'Disabled' }
    highAvailability: { mode: 'Disabled' }
    network: { publicNetworkAccess: 'Enabled' }
    authConfig: {
      activeDirectoryAuth: 'Enabled'
      passwordAuth: 'Disabled'
      tenantId: tenant().tenantId
    }
  }
}

// The server accepts one control-plane change at a time, so the children are chained.
resource pgAdmin 'Microsoft.DBforPostgreSQL/flexibleServers/administrators@2024-08-01' = if (deployPg) {
  parent: pg
  name: adminObjectId
  properties: {
    principalType: 'User'
    principalName: adminPrincipalName
    tenantId: tenant().tenantId
  }
}

resource pgExtensions 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2024-08-01' = if (deployPg) {
  parent: pg
  name: 'azure.extensions'
  properties: { value: 'VECTOR', source: 'user-override' }
  dependsOn: [ pgAdmin ]
}

resource pgDb 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2024-08-01' = if (deployPg) {
  parent: pg
  name: 'rag'
  properties: { charset: 'UTF8', collation: 'en_US.utf8' }
  dependsOn: [ pgExtensions ]
}

resource pgFirewall 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = if (deployPg) {
  parent: pg
  name: 'developer'
  properties: { startIpAddress: clientIp, endIpAddress: clientIp }
  dependsOn: [ pgDb ]
}

// ---------------------------------------------------------- Azure AI Foundry

resource ai 'Microsoft.CognitiveServices/accounts@2025-06-01' = if (deployAi) {
  name: 'ai-governed-rag-${suffix}'
  location: location
  tags: tags
  kind: 'AIServices'
  sku: { name: 'S0' }
  identity: { type: 'SystemAssigned' }  // Foundry projects need it
  properties: {
    customSubDomainName: 'ai-governed-rag-${suffix}'  // required for Entra ID tokens
    disableLocalAuth: true                            // no API keys: Entra ID only
    publicNetworkAccess: 'Enabled'
    networkAcls: { defaultAction: 'Allow' }
    allowProjectManagement: true                      // Foundry project management
    defaultProject: 'rg-governed-rag-proj'
  }
}

// Deployments on one account are created one at a time, so each waits for the previous one.
resource generator 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = if (deployAi) {
  parent: ai
  name: 'generator'
  sku: { name: 'GlobalStandard', capacity: generatorCapacity }  // see docs/azure/DECISIONS.md, D11
  properties: {
    model: { format: 'OpenAI', name: 'gpt-5-mini', version: '2025-08-07' }
    versionUpgradeOption: 'NoAutoUpgrade'  // a model change must go through the eval gate
  }
}

resource judge 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = if (deployAi) {
  parent: ai
  name: 'judge'
  sku: { name: 'GlobalStandard', capacity: judgeCapacity }
  properties: {
    model: { format: 'OpenAI', name: 'gpt-5-mini', version: '2025-08-07' }
    versionUpgradeOption: 'NoAutoUpgrade'
  }
  dependsOn: [ generator ]
}

resource embedder 'Microsoft.CognitiveServices/accounts/deployments@2025-06-01' = if (deployAi) {
  parent: ai
  name: 'embedder'
  sku: { name: 'Standard', capacity: embedderCapacity }
  properties: {
    model: { format: 'OpenAI', name: 'text-embedding-3-large', version: '1' }
    versionUpgradeOption: 'NoAutoUpgrade'
  }
  dependsOn: [ judge ]
}

// "Cognitive Services OpenAI User": call deployments, nothing else.
var openAiUserRole = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

resource adminCanCallModels 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployAi) {
  scope: ai
  name: guid(ai.id, adminObjectId, openAiUserRole)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', openAiUserRole)
    principalId: adminObjectId
    principalType: 'User'
  }
}

output postgresHost string = deployPg ? pg!.properties.fullyQualifiedDomainName : ''
output postgresServer string = deployPg ? pg.name : ''
output aiEndpoint string = deployAi ? ai!.properties.endpoint : ''
output aiAccount string = deployAi ? ai.name : ''
