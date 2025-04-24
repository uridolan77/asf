/**
 * ClinicalTrials.gov Data Export Utilities
 * 
 * Utilities for parsing and exporting data in CSV and RIS formats
 */

import { Study } from './study-model';

/**
 * CSV Export Utilities
 */
export class CSVExporter {
  /**
   * Default columns to include in CSV export
   */
  static readonly DEFAULT_COLUMNS = [
    'NCT Number',
    'Study Title',
    'Study URL',
    'Study Status',
    'Conditions',
    'Interventions',
    'Sponsor',
    'Sex',
    'Age',
    'Phases',
    'Enrollment',
    'Study Type',
    'Study Design',
    'Start Date',
    'Primary Completion Date',
    'First Posted',
    'Results First Posted',
    'Last Update Posted'
  ];

  /**
   * Convert an array of studies to CSV format
   * @param studies Array of studies to convert
   * @param columns Specific columns to include (optional)
   * @returns CSV-formatted string
   */
  static exportStudies(
    studies: Study[], 
    columns?: string[]
  ): string {
    const columnsToUse = columns || this.DEFAULT_COLUMNS;
    
    // Create CSV header
    let csv = columnsToUse.join(',') + '\n';
    
    // Add rows for each study
    for (const study of studies) {
      const row = columnsToUse.map(column => this.getColumnValue(study, column));
      csv += row.join(',') + '\n';
    }
    
    return csv;
  }

  /**
   * Get the value for a specific column from a study
   * @param study The study to extract data from
   * @param column The column name
   * @returns Formatted cell value for the CSV
   */
  private static getColumnValue(study: Study, column: string): string {
    // Extract the appropriate data based on the column
    let value = '';
    
    switch (column) {
      case 'NCT Number':
        value = study.protocolSection?.identificationModule?.nctId || '';
        break;
      case 'Study Title':
        value = study.protocolSection?.identificationModule?.briefTitle || '';
        break;
      case 'Study URL':
        const nctId = study.protocolSection?.identificationModule?.nctId;
        value = nctId ? `https://clinicaltrials.gov/study/${nctId}` : '';
        break;
      case 'Acronym':
        value = study.protocolSection?.identificationModule?.acronym || '';
        break;
      case 'Study Status':
        value = study.protocolSection?.statusModule?.overallStatus || '';
        break;
      case 'Brief Summary':
        value = study.protocolSection?.descriptionModule?.briefSummary || '';
        break;
      case 'Study Results':
        value = study.hasResults ? 'Has Results' : 'No Results';
        break;
      case 'Conditions':
        value = study.protocolSection?.conditionsModule?.conditions?.join('; ') || '';
        break;
      case 'Interventions':
        if (study.protocolSection?.armsInterventionsModule?.interventions) {
          value = study.protocolSection.armsInterventionsModule.interventions
            .map(i => `${i.type}: ${i.name}`)
            .join('; ');
        }
        break;
      case 'Primary Outcome Measures':
        if (study.protocolSection?.outcomesModule?.primaryOutcomes) {
          value = study.protocolSection.outcomesModule.primaryOutcomes
            .map(o => o.measure)
            .join('; ');
        }
        break;
      case 'Secondary Outcome Measures':
        if (study.protocolSection?.outcomesModule?.secondaryOutcomes) {
          value = study.protocolSection.outcomesModule.secondaryOutcomes
            .map(o => o.measure)
            .join('; ');
        }
        break;
      case 'Other Outcome Measures':
        if (study.protocolSection?.outcomesModule?.otherOutcomes) {
          value = study.protocolSection.outcomesModule.otherOutcomes
            .map(o => o.measure)
            .join('; ');
        }
        break;
      case 'Sponsor':
        value = study.protocolSection?.sponsorCollaboratorsModule?.leadSponsor?.name || '';
        break;
      case 'Collaborators':
        if (study.protocolSection?.sponsorCollaboratorsModule?.collaborators) {
          value = study.protocolSection.sponsorCollaboratorsModule.collaborators
            .map(c => c.name)
            .join('; ');
        }
        break;
      case 'Sex':
        value = study.protocolSection?.eligibilityModule?.sex || '';
        break;
      case 'Age':
        const minAge = study.protocolSection?.eligibilityModule?.minimumAge;
        const maxAge = study.protocolSection?.eligibilityModule?.maximumAge;
        if (minAge && maxAge) {
          value = `${minAge} to ${maxAge}`;
        } else if (minAge) {
          value = `≥ ${minAge}`;
        } else if (maxAge) {
          value = `≤ ${maxAge}`;
        }
        break;
      case 'Phases':
        value = study.protocolSection?.designModule?.phases?.join('; ') || '';
        break;
      case 'Enrollment':
        value = study.protocolSection?.designModule?.enrollmentInfo?.count?.toString() || '';
        break;
      case 'Funder Type':
        value = study.protocolSection?.sponsorCollaboratorsModule?.leadSponsor?.class || '';
        break;
      case 'Study Type':
        value = study.protocolSection?.designModule?.studyType || '';
        break;
      case 'Study Design':
        const design = study.protocolSection?.designModule?.designInfo;
        if (design) {
          const parts = [];
          if (design.allocation) parts.push(design.allocation);
          if (design.interventionModel) parts.push(design.interventionModel);
          if (design.primaryPurpose) parts.push(design.primaryPurpose);
          if (design.maskingInfo?.masking) parts.push(design.maskingInfo.masking);
          value = parts.join(', ');
        }
        break;
      case 'Other IDs':
        const orgId = study.protocolSection?.identificationModule?.orgStudyIdInfo?.id;
        const secondaryIds = study.protocolSection?.identificationModule?.secondaryIdInfos?.map(id => id.id);
        const allIds = [orgId, ...(secondaryIds || [])].filter(Boolean);
        value = allIds.join('; ');
        break;
      case 'Start Date':
        value = study.protocolSection?.statusModule?.startDateStruct?.date || '';
        break;
      case 'Primary Completion Date':
        value = study.protocolSection?.statusModule?.primaryCompletionDateStruct?.date || '';
        break;
      case 'Completion Date':
        value = study.protocolSection?.statusModule?.completionDateStruct?.date || '';
        break;
      case 'First Posted':
        value = study.protocolSection?.statusModule?.studyFirstPostDateStruct?.date || '';
        break;
      case 'Results First Posted':
        value = study.protocolSection?.statusModule?.resultsFirstPostDateStruct?.date || '';
        break;
      case 'Last Update Posted':
        value = study.protocolSection?.statusModule?.lastUpdatePostDateStruct?.date || '';
        break;
      case 'Locations':
        if (study.protocolSection?.contactsLocationsModule?.locations) {
          value = study.protocolSection.contactsLocationsModule.locations
            .map(loc => {
              const parts = [loc.facility, loc.city, loc.state, loc.country].filter(Boolean);
              return parts.join(', ');
            })
            .join('; ');
        }
        break;
      case 'Study Documents':
        if (study.documentSection?.largeDocumentModule?.largeDocs) {
          value = study.documentSection.largeDocumentModule.largeDocs
            .map(doc => doc.label || doc.filename)
            .join('; ');
        }
        break;
      default:
        value = '';
    }
    
    // Escape any commas and quotes for CSV format
    return `"${value.replace(/"/g, '""')}"`;
  }

  /**
   * Parse a CSV string into an array of objects
   * @param csv CSV data as a string
   * @returns Array of objects with properties for each column
   */
  static parseCSV(csv: string): Record<string, string>[] {
    const lines = csv.split('\n');
    const headers = this.parseCSVLine(lines[0]);
    
    const results: Record<string, string>[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      
      const values = this.parseCSVLine(lines[i]);
      const obj: Record<string, string> = {};
      
      for (let j = 0; j < headers.length; j++) {
        obj[headers[j]] = values[j] || '';
      }
      
      results.push(obj);
    }
    
    return results;
  }

  /**
   * Parse a CSV line into an array of values
   * @param line CSV line
   * @returns Array of values
   */
  private static parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      
      if (char === '"') {
        // Check for escaped quotes
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        result.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    
    result.push(current);
    return result;
  }
}

/**
 * RIS Export Utilities
 */
export class RISExporter {
  /**
   * Convert an array of studies to RIS format
   * @param studies Array of studies to convert
   * @returns RIS-formatted string
   */
  static exportStudies(studies: Study[]): string {
    let ris = '';
    
    for (const study of studies) {
      ris += this.studyToRIS(study) + '\n';
    }
    
    return ris;
  }

  /**
   * Convert a single study to RIS format
   * @param study The study to convert
   * @returns RIS-formatted string
   */
  private static studyToRIS(study: Study): string {
    const id = study.protocolSection?.identificationModule;
    const status = study.protocolSection?.statusModule;
    const sponsor = study.protocolSection?.sponsorCollaboratorsModule;
    const design = study.protocolSection?.designModule;
    const contact = study.protocolSection?.contactsLocationsModule;
    const docs = study.documentSection?.largeDocumentModule;
    
    let ris = '';
    
    // Required RIS fields
    ris += 'TY  - DBASE\n';
    ris += 'DP  - National Library of Medicine (US)\n';
    ris += 'PP  - Bethesda (MD)\n';
    
    // Study identifiers
    if (id?.nctId) {
      ris += `ID  - ${id.nctId}\n`;
      ris += `AN  - ${id.nctId}\n`;
    }
    
    // Secondary IDs
    if (id?.secondaryIdInfos) {
      for (const secId of id.secondaryIdInfos) {
        if (secId.id && secId.type) {
          ris += `U1  - ${secId.id} [${secId.type}]\n`;
        }
      }
    }
    
    // Title info
    if (id?.briefTitle) {
      ris += `ST  - ${id.briefTitle}`;
      if (id.acronym) {
        ris += ` (${id.acronym})`;
      }
      ris += '\n';
    }
    
    if (id?.officialTitle) {
      ris += `TI  - ${id.officialTitle}\n`;
    }
    
    // Dates
    if (status?.studyFirstSubmitDate) {
      ris += `Y1  - ${status.studyFirstSubmitDate}\n`;
    }
    
    if (status?.startDateStruct?.date) {
      ris += `Y2  - ${status.startDateStruct.date}\n`;
    }
    
    // Collaborators
    if (sponsor?.collaborators) {
      for (const collaborator of sponsor.collaborators) {
        if (collaborator.name) {
          ris += `A2  - ${collaborator.name}\n`;
        }
      }
    }
    
    // Sponsor
    if (sponsor?.leadSponsor?.name) {
      ris += `C1  - ${sponsor.leadSponsor.name}\n`;
    }
    
    // Status
    if (status?.overallStatus) {
      ris += `C2  - ${status.overallStatus}\n`;
    }
    
    // Update dates
    if (status?.lastUpdatePostDateStruct?.date) {
      ris += `C3  - ${status.lastUpdatePostDateStruct.date}\n`;
    }
    
    if (status?.lastUpdateSubmitDate) {
      ris += `C4  - ${status.lastUpdateSubmitDate}\n`;
    }
    
    // Study type
    if (design?.studyType) {
      ris += `C5  - ${design.studyType}\n`;
    }
    
    // Has results
    ris += `C6  - ${study.hasResults ? 'Has Results' : 'No Results'}\n`;
    
    // Study documents
    if (docs?.largeDocs) {
      const docTypes = [];
      if (docs.largeDocs.some(doc => doc.hasProtocol)) docTypes.push('Protocol');
      if (docs.largeDocs.some(doc => doc.hasSap)) docTypes.push('SAP');
      if (docs.largeDocs.some(doc => doc.hasIcf)) docTypes.push('ICF');
      
      if (docTypes.length > 0) {
        ris += `C7  - ${docTypes.join(', ')}\n`;
      }
    }
    
    // Central contact
    if (contact?.centralContacts && contact.centralContacts.length > 0) {
      const centralContact = contact.centralContacts[0];
      
      if (centralContact.name) {
        ris += `C8  - ${centralContact.name}`;
        if (centralContact.role) {
          ris += ` [${centralContact.role}]`;
        }
        ris += '\n';
      }
      
      if (centralContact.phone) {
        ris += `C8  - ${centralContact.phone}`;
        if (centralContact.phoneExt) {
          ris += ` ext. ${centralContact.phoneExt}`;
        }
        ris += '\n';
      }
      
      if (centralContact.email) {
        ris += `C8  - ${centralContact.email}\n`;
      }
    }
    
    // Brief summary
    if (study.protocolSection?.descriptionModule?.briefSummary) {
      ris += `AB  - ${study.protocolSection.descriptionModule.briefSummary.replace(/\n/g, ' ')}\n`;
    }
    
    // URL
    if (id?.nctId) {
      ris += `UR  - https://clinicaltrials.gov/study/${id.nctId}\n`;
    }
    
    // ClinicalTrials.gov database indicator
    ris += 'SF  - ClinicalTrials.gov\n';
    
    // Retrieved date (current date)
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    ris += `RD  - ${year}/${month}/${day}\n`;
    
    // End record
    ris += 'ER  - \n';
    
    return ris;
  }

  /**
   * Parse a RIS string into an array of objects
   * @param ris RIS data as a string
   * @returns Array of objects with properties for each RIS tag
   */
  static parseRIS(ris: string): Record<string, string[]>[] {
    const records: Record<string, string[]>[] = [];
    let currentRecord: Record<string, string[]> = {};
    
    const lines = ris.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      // Skip empty lines
      if (!line) continue;
      
      // Check for end of record
      if (line === 'ER  - ') {
        if (Object.keys(currentRecord).length > 0) {
          records.push(currentRecord);
          currentRecord = {};
        }
        continue;
      }
      
      // Parse RIS tag and content
      const match = line.match(/^([A-Z][A-Z0-9])  - (.*)$/);
      if (match) {
        const [, tag, content] = match;
        
        if (!currentRecord[tag]) {
          currentRecord[tag] = [];
        }
        
        currentRecord[tag].push(content);
      }
    }
    
    // Add the last record if not already added
    if (Object.keys(currentRecord).length > 0) {
      records.push(currentRecord);
    }
    
    return records;
  }
}