/**
 * ClinicalTrials.gov API Connector Usage Examples
 * 
 * This file demonstrates how to use the ClinicalTrials.gov API connector
 * for various common tasks.
 */

import { ClinicalTrialsClient } from './client';
import { SearchQueryBuilder } from './search';
import { CSVExporter, RISExporter } from './data-exporters';

/**
 * Basic API client usage
 */
async function basicClientUsage() {
  // Create a client
  const client = new ClinicalTrialsClient();
  
  // Get API version information
  const versionInfo = await client.getVersion();
  console.log('API Version:', versionInfo.version);
  console.log('Data Timestamp:', versionInfo.dataTimestamp);
  
  // Get study metadata (available fields)
  const metadata = await client.getMetadata();
  console.log('Available fields:', metadata);
  
  // Get a single study by NCT ID
  const study = await client.getStudy('NCT04280705'); // Example: COVID-19 NIAID remdesivir study
  console.log('Study Title:', study.protocolSection.identificationModule.briefTitle);
  
  // Simple search by keyword
  const results = await client.searchStudies('COVID-19 AND AREA[Phase]PHASE3', {
    pageSize: 10
  });
  console.log(`Found ${results.totalCount} studies. First 10 results:`);
  results.studies.forEach(study => {
    console.log(`- ${study.protocolSection.identificationModule.nctId}: ${study.protocolSection.identificationModule.briefTitle}`);
  });
}

/**
 * Using the search query builder
 */
async function searchQueryBuilderUsage() {
  const client = new ClinicalTrialsClient();
  
  // Build a search query to find Phase 3 studies about COVID-19 
  // that are currently recruiting and have posted results
  const builder = new SearchQueryBuilder()
    .condition('COVID-19')
    .phase('PHASE3')
    .status('RECRUITING')
    .hasResults(true);
  
  // Execute the search
  const results = await client.searchStudies(builder, {
    pageSize: 10,
    fields: [
      'NCTId', 
      'BriefTitle', 
      'OverallStatus', 
      'LeadSponsorName',
      'StartDate',
      'PrimaryCompletionDate'
    ]
  });
  
  console.log(`COVID-19 Phase 3 Recruiting Studies with Results: ${results.totalCount}`);
  results.studies.forEach(study => {
    const id = study.protocolSection.identificationModule;
    const status = study.protocolSection.statusModule;
    const sponsor = study.protocolSection.sponsorCollaboratorsModule?.leadSponsor;
    
    console.log(`
      NCT ID: ${id.nctId}
      Title: ${id.briefTitle}
      Status: ${status?.overallStatus}
      Sponsor: ${sponsor?.name}
      Start Date: ${status?.startDateStruct?.date}
      Primary Completion: ${status?.primaryCompletionDateStruct?.date}
    `);
  });
}

/**
 * Advanced search examples
 */
async function advancedSearchExamples() {
  const client = new ClinicalTrialsClient();
  
  // 1. Find pediatric studies for a specific condition
  const pediatricSearch = new SearchQueryBuilder()
    .condition('leukemia')
    .ageGroup('CHILD')
    .phase(['PHASE1', 'PHASE2'])
    .status(['RECRUITING', 'NOT_YET_RECRUITING']);
  
  const pediatricResults = await client.searchStudies(pediatricSearch, { pageSize: 5 });
  console.log(`Pediatric leukemia studies: ${pediatricResults.totalCount}`);
  
  // 2. Find studies in a specific location
  const locationSearch = new SearchQueryBuilder()
    .locationSearch('Boston', 'Massachusetts', 'United States', 'RECRUITING')
    .condition('diabetes')
    .phase('PHASE3');
  
  const locationResults = await client.searchStudies(locationSearch, { pageSize: 5 });
  console.log(`Boston diabetes studies: ${locationResults.totalCount}`);
  
  // 3. Find studies by a specific sponsor
  const sponsorSearch = new SearchQueryBuilder()
    .sponsor('National Cancer Institute')
    .status('RECRUITING')
    .startDateRange('01/01/2023', 'MAX');
  
  const sponsorResults = await client.searchStudies(sponsorSearch, { pageSize: 5 });
  console.log(`Recent NCI studies: ${sponsorResults.totalCount}`);
  
  // 4. Using raw complex query syntax
  const rawQuery = 'SEARCH[Location](AREA[LocationCountry]United States AND AREA[LocationState]California) AND AREA[OverallStatus]RECRUITING AND AREA[Condition]cancer AND AREA[Phase]PHASE3';
  
  const rawResults = await client.searchStudies(rawQuery, { pageSize: 5 });
  console.log(`California cancer studies: ${rawResults.totalCount}`);
}

/**
 * Working with CSV exports
 */
async function csvExportExample() {
  const client = new ClinicalTrialsClient();
  
  // Find some studies
  const searchResults = await client.searchStudies('AREA[Condition]Alzheimer AND AREA[Phase]PHASE3', {
    pageSize: 10
  });
  
  // Export to CSV
  const csv = CSVExporter.exportStudies(searchResults.studies);
  console.log('CSV Export:');
  console.log(csv.substring(0, 500) + '...');
  
  // Export with custom columns
  const customCsv = CSVExporter.exportStudies(searchResults.studies, [
    'NCT Number',
    'Study Title',
    'Sponsor',
    'Phases',
    'Study Status',
    'Start Date',
    'Primary Completion Date'
  ]);
  console.log('Custom CSV Export:');
  console.log(customCsv);
  
  // Directly download studies as CSV (server-side extraction)
  const nctIds = searchResults.studies.map(s => s.protocolSection.identificationModule.nctId);
  const downloadCsv = await client.downloadCSV(nctIds);
  console.log('Downloaded CSV:');
  console.log(downloadCsv.substring(0, 500) + '...');
}

/**
 * Working with RIS exports
 */
async function risExportExample() {
  const client = new ClinicalTrialsClient();
  
  // Find some studies
  const searchResults = await client.searchStudies('AREA[Condition]"breast cancer" AND AREA[HasResults]true', {
    pageSize: 5
  });
  
  // Export to RIS
  const ris = RISExporter.exportStudies(searchResults.studies);
  console.log('RIS Export:');
  console.log(ris.substring(0, 500) + '...');
  
  // Directly download studies as RIS (server-side extraction)
  const nctIds = searchResults.studies.map(s => s.protocolSection.identificationModule.nctId);
  const downloadRis = await client.downloadRIS(nctIds);
  console.log('Downloaded RIS:');
  console.log(downloadRis.substring(0, 500) + '...');
}

/**
 * Working with study details
 */
async function studyDetailsExample() {
  const client = new ClinicalTrialsClient();
  
  // Get a complete study record
  const study = await client.getStudy('NCT04280705');
  
  // Access identification information
  const id = study.protocolSection.identificationModule;
  console.log(`NCT ID: ${id.nctId}`);
  console.log(`Brief Title: ${id.briefTitle}`);
  console.log(`Official Title: ${id.officialTitle}`);
  
  // Access status information
  const status = study.protocolSection.statusModule;
  console.log(`Status: ${status?.overallStatus}`);
  console.log(`Start Date: ${status?.startDateStruct?.date}`);
  console.log(`Completion Date: ${status?.completionDateStruct?.date}`);
  
  // Access sponsor information
  const sponsor = study.protocolSection.sponsorCollaboratorsModule;
  console.log(`Lead Sponsor: ${sponsor?.leadSponsor?.name} (${sponsor?.leadSponsor?.class})`);
  
  if (sponsor?.collaborators) {
    console.log('Collaborators:');
    sponsor.collaborators.forEach(c => {
      console.log(`- ${c.name} (${c.class})`);
    });
  }
  
  // Access design information
  const design = study.protocolSection.designModule;
  console.log(`Study Type: ${design?.studyType}`);
  console.log(`Phases: ${design?.phases?.join(', ')}`);
  console.log(`Enrollment: ${design?.enrollmentInfo?.count}`);
  
  // Access arms and interventions
  const arms = study.protocolSection.armsInterventionsModule;
  
  if (arms?.armGroups) {
    console.log('Arms:');
    arms.armGroups.forEach(arm => {
      console.log(`- ${arm.label} (${arm.type}): ${arm.description}`);
    });
  }
  
  if (arms?.interventions) {
    console.log('Interventions:');
    arms.interventions.forEach(intervention => {
      console.log(`- ${intervention.name} (${intervention.type}): ${intervention.description}`);
    });
  }
  
  // Access eligibility information
  const eligibility = study.protocolSection.eligibilityModule;
  console.log(`Gender: ${eligibility?.sex}`);
  console.log(`Age: ${eligibility?.minimumAge} to ${eligibility?.maximumAge}`);
  console.log(`Healthy Volunteers: ${eligibility?.healthyVolunteers ? 'Yes' : 'No'}`);
  
  // Access outcomes information
  const outcomes = study.protocolSection.outcomesModule;
  
  if (outcomes?.primaryOutcomes) {
    console.log('Primary Outcomes:');
    outcomes.primaryOutcomes.forEach(outcome => {
      console.log(`- ${outcome.measure} (${outcome.timeFrame})`);
    });
  }
  
  // Check if the study has results
  if (study.hasResults && study.resultsSection) {
    console.log('Study has results');
    
    // Access baseline characteristics
    const baseline = study.resultsSection.baselineCharacteristicsModule;
    console.log(`Population: ${baseline?.populationDescription}`);
    
    // Access outcome measures
    const resultOutcomes = study.resultsSection.outcomeMeasuresModule;
    
    if (resultOutcomes?.outcomeMeasures) {
      console.log('Outcome Measure Results:');
      resultOutcomes.outcomeMeasures.forEach(outcome => {
        console.log(`- ${outcome.title} (${outcome.type})`);
      });
    }
    
    // Access adverse events
    const adverseEvents = study.resultsSection.adverseEventsModule;
    
    if (adverseEvents?.seriousEvents) {
      console.log(`Serious Adverse Events: ${adverseEvents.seriousEvents.length}`);
    }
  }
  
  // Access locations
  const locations = study.protocolSection.contactsLocationsModule;
  
  if (locations?.locations) {
    console.log(`Study Locations: ${locations.locations.length}`);
    
    // Show first 3 locations
    locations.locations.slice(0, 3).forEach(location => {
      console.log(`- ${location.facility}, ${location.city}, ${location.state}, ${location.country}`);
    });
  }
}

// Run examples
async function main() {
  try {
    console.log('=== Basic Client Usage ===');
    await basicClientUsage();
    
    console.log('\n=== Search Query Builder Usage ===');
    await searchQueryBuilderUsage();
    
    console.log('\n=== Advanced Search Examples ===');
    await advancedSearchExamples();
    
    console.log('\n=== CSV Export Example ===');
    await csvExportExample();
    
    console.log('\n=== RIS Export Example ===');
    await risExportExample();
    
    console.log('\n=== Study Details Example ===');
    await studyDetailsExample();
    
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Uncomment to run the examples
// main();