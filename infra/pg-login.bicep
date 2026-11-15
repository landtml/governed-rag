// Adds a managed identity as a Microsoft Entra administrator of a PostgreSQL
// flexible server. A module, because the administrator's name is the identity's
// object ID, which exists only once the identity has been created.
targetScope = 'resourceGroup'

param pgServerName string
param principalId string
param principalName string

resource pg 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' existing = {
  name: pgServerName
}

resource login 'Microsoft.DBforPostgreSQL/flexibleServers/administrators@2024-08-01' = {
  parent: pg
  name: principalId
  properties: {
    principalType: 'ServicePrincipal'
    principalName: principalName
    tenantId: tenant().tenantId
  }
}
