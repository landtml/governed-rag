// The application layer of governed-rag on Azure, on top of infra/main.bicep:
//
//   * Azure Container Registry for the service image (built with `az acr build`).
//   * A Log Analytics workspace and an Azure Container Apps environment that ships
//     container logs to it.
//   * A user-assigned managed identity for the app: it pulls the image, calls the
//     Azure OpenAI deployments and logs in to PostgreSQL with Microsoft Entra ID.
//     No key, password or connection secret exists anywhere in the deployment.
//   * The container app: `rag serve` behind HTTPS ingress, scaling to zero when idle.
//
// Deploy with infra/deploy-app.sh, which runs this template twice: once with
// deployApp=false to create the registry, then again with the built image.
targetScope = 'resourceGroup'

param location string = resourceGroup().location

@description('Short lowercase suffix that makes globally unique names.')
param suffix string = uniqueString(resourceGroup().id)

@description('Name of the PostgreSQL flexible server created by infra/main.bicep.')
param pgServerName string

@description('Name of the Azure AI Foundry account created by infra/main.bicep.')
param aiAccountName string

@description('False creates everything except the container app, so the image can be built first.')
param deployApp bool = true

@description('Image reference in the registry, for example governed-rag:1.')
param image string = ''

@secure()
@description('The bearer-token file for `rag serve`: JSON {"<token>": ["group", ...]}. Stored as a Container Apps secret.')
param tokensJson string = ''

@description('Deployment names inside the AI account.')
param generatorDeployment string = 'generator'
param embedderDeployment string = 'embedder'

@description('Upper bound on replicas. The lower bound is 0, so an idle app costs no compute.')
param maxReplicas int = 2

@description('Let Azure-hosted clients reach PostgreSQL. Container Apps on the Consumption plan have no fixed outbound address; login still requires an Entra token.')
param allowAzureServicesToPostgres bool = true

var tags = { project: 'governed-rag' }
var appName = 'ca-governed-rag'

// ------------------------------------------------------------ existing layer

resource pg 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' existing = {
  name: pgServerName
}

resource ai 'Microsoft.CognitiveServices/accounts@2025-06-01' existing = {
  name: aiAccountName
}

// ------------------------------------------------------------------ identity

resource appIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-governed-rag'
  location: location
  tags: tags
}

// "Cognitive Services OpenAI User": call deployments, nothing else.
var openAiUserRole = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
// "AcrPull": read images, nothing else.
var acrPullRole = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

resource appCanCallModels 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: ai
  name: guid(ai.id, appIdentity.id, openAiUserRole)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', openAiUserRole)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// The identity becomes an Entra administrator of the server, so the app can apply
// its schema at start-up. A narrower role is possible with pgaadauth_create_principal.
module appPgLogin 'pg-login.bicep' = {
  name: 'governed-rag-pg-login'
  params: {
    pgServerName: pg.name
    principalId: appIdentity.properties.principalId
    principalName: appIdentity.name
  }
}

resource pgAzureServices 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2024-08-01' = if (allowAzureServicesToPostgres) {
  parent: pg
  name: 'AllowAllAzureServicesAndResourcesWithinAzureIps'
  properties: { startIpAddress: '0.0.0.0', endIpAddress: '0.0.0.0' }
  dependsOn: [ appPgLogin ]  // the server accepts one control-plane change at a time
}

// ------------------------------------------------------------------ registry

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'crgovernedrag${suffix}'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }  // pulls use the managed identity
}

resource appCanPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  name: guid(registry.id, appIdentity.id, acrPullRole)
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRole)
    principalId: appIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ------------------------------------------------------- logs and environment

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-governed-rag-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-governed-rag'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

// --------------------------------------------------------------------- app

resource app 'Microsoft.App/containerApps@2024-03-01' = if (deployApp) {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${appIdentity.id}': {} }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8080
        transport: 'auto'
        allowInsecure: false
      }
      registries: [ { server: registry.properties.loginServer, identity: appIdentity.id } ]
      secrets: [ { name: 'tokens-json', value: tokensJson } ]
    }
    template: {
      containers: [
        {
          name: 'rag'
          image: '${registry.properties.loginServer}/${image}'
          resources: { cpu: json('2.0'), memory: '4Gi' }  // the reranker runs on CPU in the container
          env: [
            { name: 'DATABASE_URL', value: 'postgresql://${appIdentity.name}@${pg.properties.fullyQualifiedDomainName}:5432/rag?sslmode=require' }
            { name: 'AZURE_CLIENT_ID', value: appIdentity.properties.clientId }
            { name: 'AZURE_OPENAI_ENDPOINT', value: ai.properties.endpoint }
            { name: 'AZURE_OPENAI_GENERATOR_DEPLOYMENT', value: generatorDeployment }
            { name: 'AZURE_OPENAI_EMBEDDER_DEPLOYMENT', value: embedderDeployment }
          ]
          volumeMounts: [ { volumeName: 'secrets', mountPath: '/secrets' } ]
          probes: [
            {
              type: 'Startup'
              httpGet: { path: '/healthz', port: 8080 }
              periodSeconds: 10
              failureThreshold: 30  // loading the reranker takes a while on a cold start
            }
            {
              type: 'Liveness'
              httpGet: { path: '/healthz', port: 8080 }
              periodSeconds: 30
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'secrets'
          storageType: 'Secret'
          secrets: [ { secretRef: 'tokens-json', path: 'tokens.json' } ]
        }
      ]
      scale: { minReplicas: 0, maxReplicas: maxReplicas }
    }
  }
  dependsOn: [ appCanPull, appCanCallModels, appPgLogin ]
}

output registryName string = registry.name
output registryServer string = registry.properties.loginServer
output appUrl string = deployApp ? 'https://${app!.properties.configuration.ingress.fqdn}' : ''
output identityName string = appIdentity.name
