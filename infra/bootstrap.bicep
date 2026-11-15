// Subscription-level setup, deployed once before anything that costs money:
// the resource group every other resource lives in, and a monthly budget
// that emails when actual spend crosses 50%, 80% and 100%, or when the
// forecast crosses 100%.
//
//   az deployment sub create -l <region> -f infra/bootstrap.bicep \
//     -p budgetAmount=50 contactEmail=<you@example.com>
targetScope = 'subscription'

@description('Region for the resource group; every resource in it defaults to the same region. Defaults to the region given with -l.')
param location string = deployment().location

@description('Name of the single resource group that holds the whole deployment.')
param resourceGroupName string = 'rg-governed-rag'

@description('Monthly budget in the billing currency of the subscription.')
param budgetAmount int = 50

@description('Address that receives the budget alerts. Passed on the command line, never committed.')
param contactEmail string

@description('First day of the current month, as the budget start date.')
param budgetStart string = '${utcNow('yyyy-MM')}-01'

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: resourceGroupName
  location: location
  tags: { project: 'governed-rag' }
}

@description('False skips the budget, for subscription offers that do not support Cost Management budgets.')
param deployBudget bool = true

resource budget 'Microsoft.Consumption/budgets@2024-08-01' = if (deployBudget) {
  name: 'governed-rag-monthly'
  properties: {
    category: 'Cost'
    amount: budgetAmount
    timeGrain: 'Monthly'
    timePeriod: { startDate: budgetStart }
    notifications: {
      actual50: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 50
        thresholdType: 'Actual'
        contactEmails: [ contactEmail ]
      }
      actual80: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 80
        thresholdType: 'Actual'
        contactEmails: [ contactEmail ]
      }
      actual100: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Actual'
        contactEmails: [ contactEmail ]
      }
      forecast100: {
        enabled: true
        operator: 'GreaterThanOrEqualTo'
        threshold: 100
        thresholdType: 'Forecasted'
        contactEmails: [ contactEmail ]
      }
    }
  }
}

output resourceGroup string = rg.name
