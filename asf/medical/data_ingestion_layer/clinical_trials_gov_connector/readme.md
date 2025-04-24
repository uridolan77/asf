# ClinicalTrials.gov API Connector

A comprehensive TypeScript/JavaScript library for interacting with the ClinicalTrials.gov API. This connector provides easy access to clinical trial data, powerful search capabilities, and export functionality.

## Features

- **Complete API Coverage**: Access all ClinicalTrials.gov API endpoints
- **Powerful Search Builder**: Construct complex search queries with an intuitive builder pattern
- **Type Definitions**: Full TypeScript type definitions for all API responses
- **Data Export**: Export studies in CSV and RIS formats
- **Error Handling**: Robust error handling and reporting
- **Browser & Node.js**: Works in both browser and Node.js environments

## Installation

```bash
npm install clinicaltrials-connector
```

## Quick Start

```typescript
import { ClinicalTrialsClient, SearchQueryBuilder } from 'clinicaltrials-connector';

// Create a client
const client = new ClinicalTrialsClient();

// Get API version
const version = await client.getVersion();
console.log('API Version:', version.version);

// Search for studies using the query builder
const query = new SearchQueryBuilder()
  .condition('COVID-19')
  .phase('PHASE3')
  .status('RECRUITING')
  .hasResults(true);

const results = await client.searchStudies(query, { pageSize: 10 });
console.log(`Found ${results.totalCount} studies`);

// Get details for a specific study
const study = await client.getStudy('NCT04280705');
console.log('Study Title:', study.protocolSection.identificationModule.briefTitle);
```

## Search Query Builder

The search query builder provides a fluent interface for constructing complex search queries:

```typescript
const builder = new SearchQueryBuilder()
  // Search for a specific condition
  .condition('diabetes')
  
  // Filter by study phase
  .phase(['PHASE2', 'PHASE3'])
  
  // Filter by recruitment status
  .status('RECRUITING')
  
  // Filter by location
  .locationSearch('Boston', 'Massachusetts', 'United States')
  
  // Filter by date ranges
  .startDateRange('01/01/2020', 'MAX')
  
  // Filter by age group
  .ageGroup('ADULT')
  
  // Search in specific fields
  .field('BriefTitle', 'Type 2 Diabetes')
  
  // Combine with raw expressions
  .and('NOT AREA[InterventionType]DEVICE');

// Use with the client
const results = await client.searchStudies(builder);
```

## Exporting Data

### CSV Export

```typescript
import { CSVExporter } from 'clinicaltrials-connector';

// Export studies to CSV
const csv = CSVExporter.exportStudies(studies);

// Export with custom columns
const customCsv = CSVExporter.exportStudies(studies, [
  'NCT Number',
  'Study Title',
  'Sponsor',
  'Study Status',
  'Start Date'
]);

// Download directly from the API
const nctIds = ['NCT04280705', 'NCT04535167'];
const downloadedCsv = await client.downloadCSV(nctIds);
```

### RIS Export (Citation Format)

```typescript
import { RISExporter } from 'clinicaltrials-connector';

// Export studies to RIS format
const ris = RISExporter.exportStudies(studies);

// Download directly from the API
const nctIds = ['NCT04280705', 'NCT04535167'];
const downloadedRis = await client.downloadRIS(nctIds);
```

## Working with Study Data

The library provides comprehensive type definitions for working with study data:

```typescript
// Get a study
const study = await client.getStudy('NCT04280705');

// Access basic information
const id = study.protocolSection.identificationModule;
console.log(`NCT ID: ${id.nctId}`);
console.log(`Title: ${id.briefTitle}`);

// Access status information
const status = study.protocolSection.statusModule;
console.log(`Status: ${status?.overallStatus}`);

// Access sponsor information
const sponsor = study.protocolSection.sponsorCollaboratorsModule;
console.log(`Lead Sponsor: ${sponsor?.leadSponsor?.name}`);

// Access eligibility information
const eligibility = study.protocolSection.eligibilityModule;
console.log(`Gender: ${eligibility?.sex}`);
console.log(`Age Range: ${eligibility?.minimumAge} to ${eligibility?.maximumAge}`);

// Check for results
if (study.hasResults && study.resultsSection) {
  console.log('Study has results');
  
  // Access outcome measures
  const outcomes = study.resultsSection.outcomeMeasuresModule;
  if (outcomes?.outcomeMeasures) {
    outcomes.outcomeMeasures.forEach(outcome => {
      console.log(`- ${outcome.title}`);
    });
  }
}
```

## API Reference

### `ClinicalTrialsClient`

- `constructor(options?: { baseUrl?: string, headers?: Record<string, string> })`
- `getVersion(): Promise<VersionInfo>`
- `searchStudies(query: string | SearchQuery, params?: SearchParams): Promise<SearchResponse>`
- `getStudy(nctId: string, params?: StudyParams): Promise<Study>`
- `getMetadata(): Promise<MetadataResponse>`
- `getSearchAreas(): Promise<SearchAreasResponse>`
- `getEnums(): Promise<EnumsResponse>`
- `getStudySizes(): Promise<StudySizesResponse>`
- `getFieldValues(params: FieldValuesParams): Promise<FieldValuesResponse>`
- `getFieldSizes(params: FieldSizesParams): Promise<FieldSizesResponse>`
- `downloadCSV(ids: string[], columns?: string[]): Promise<string>`
- `downloadRIS(ids: string[]): Promise<string>`

### `SearchQueryBuilder`

- `term(term: string): SearchQueryBuilder`
- `condition(condition: string): SearchQueryBuilder`
- `intervention(intervention: string): SearchQueryBuilder`
- `title(title: string): SearchQueryBuilder`
- `outcome(outcome: string): SearchQueryBuilder`
- `sponsor(sponsor: string): SearchQueryBuilder`
- `location(location: string): SearchQueryBuilder`
- `id(id: string): SearchQueryBuilder`
- `studyType(type: string): SearchQueryBuilder`
- `phase(phase: string | string[]): SearchQueryBuilder`
- `status(status: string | string[]): SearchQueryBuilder`
- `gender(gender: string): SearchQueryBuilder`
- `ageRange(min: string, max: string): SearchQueryBuilder`
- `ageGroup(ageGroup: string | string[]): SearchQueryBuilder`
- `funderType(funderType: string | string[]): SearchQueryBuilder`
- `startDateRange(from: string | 'MIN', to: string | 'MAX'): SearchQueryBuilder`
- `completionDateRange(from: string | 'MIN', to: string | 'MAX'): SearchQueryBuilder`
- `firstPostedDateRange(from: string | 'MIN', to: string | 'MAX'): SearchQueryBuilder`
- `hasResults(hasResults?: boolean): SearchQueryBuilder`
- `field(field: string, value: string, expansion?: ExpansionType): SearchQueryBuilder`
- `raw(expression: string): SearchQueryBuilder`
- `and(builder: SearchQueryBuilder | string): SearchQueryBuilder`
- `or(builder: SearchQueryBuilder | string): SearchQueryBuilder`
- `not(builder: SearchQueryBuilder | string): SearchQueryBuilder`
- `locationSearch(city?: string, state?: string, country?: string, status?: string): SearchQueryBuilder`
- `build(): string`

### Data Exporters

- `CSVExporter.exportStudies(studies: Study[], columns?: string[]): string`
- `CSVExporter.parseCSV(csv: string): Record<string, string>[]`
- `RISExporter.exportStudies(studies: Study[]): string`
- `RISExporter.parseRIS(ris: string): Record<string, string[]>[]`

## License

MIT