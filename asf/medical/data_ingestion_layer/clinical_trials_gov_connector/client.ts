/**
 * ClinicalTrials.gov API Client
 * 
 * This client provides methods to interact with the ClinicalTrials.gov API v2
 * Documentation: https://clinicaltrials.gov/data-api/
 */

export class ClinicalTrialsClient {
    private baseUrl: string;
    private defaultHeaders: Record<string, string>;
  
    /**
     * Create a new ClinicalTrials.gov API client
     * @param options Configuration options
     */
    constructor(options: { 
      baseUrl?: string,
      headers?: Record<string, string>
    } = {}) {
      this.baseUrl = options.baseUrl || 'https://clinicaltrials.gov/api/v2';
      this.defaultHeaders = {
        'Accept': 'application/json',
        ...(options.headers || {})
      };
    }
  
    /**
     * Get API version information
     */
    async getVersion(): Promise<VersionInfo> {
      return this.get('/version');
    }
  
    /**
     * Search for studies using the query builder or raw query string
     * @param query Search query (raw string or SearchQuery object)
     * @param params Additional parameters (fields, format, etc.)
     */
    async searchStudies(
      query: string | SearchQuery, 
      params: SearchParams = {}
    ): Promise<SearchResponse> {
      const queryString = typeof query === 'string' 
        ? query 
        : this.buildSearchQuery(query);
      
      return this.get('/studies', {
        query: queryString,
        ...params
      });
    }
  
    /**
     * Get a specific study by NCT ID
     * @param nctId The NCT identifier
     * @param params Additional parameters (fields, etc.)
     */
    async getStudy(nctId: string, params: StudyParams = {}): Promise<Study> {
      return this.get(`/studies/${nctId}`, params);
    }
  
    /**
     * Get study metadata (information about available fields)
     */
    async getMetadata(): Promise<MetadataResponse> {
      return this.get('/studies/metadata');
    }
  
    /**
     * Get available search areas
     */
    async getSearchAreas(): Promise<SearchAreasResponse> {
      return this.get('/studies/search-areas');
    }
  
    /**
     * Get enumeration types
     */
    async getEnums(): Promise<EnumsResponse> {
      return this.get('/studies/enums');
    }
  
    /**
     * Get statistical information about study sizes
     */
    async getStudySizes(): Promise<StudySizesResponse> {
      return this.get('/stats/size');
    }
  
    /**
     * Get statistical information about field values
     */
    async getFieldValues(params: FieldValuesParams): Promise<FieldValuesResponse> {
      return this.get('/stats/field/values', params);
    }
  
    /**
     * Get statistical information about list field sizes
     */
    async getFieldSizes(params: FieldSizesParams): Promise<FieldSizesResponse> {
      return this.get('/stats/field/sizes', params);
    }
  
    /**
     * Download studies in CSV format
     * @param ids Array of NCT IDs to download
     * @param columns Columns to include
     */
    async downloadCSV(ids: string[], columns?: string[]): Promise<string> {
      // Implementation uses the CSV download endpoint
      const params: any = { ids: ids.join(',') };
      if (columns) {
        params.columns = columns.join(',');
      }
      
      const response = await this.request('/studies/download', {
        method: 'GET',
        params,
        headers: {
          ...this.defaultHeaders,
          'Accept': 'text/csv'
        }
      });
      
      return response.text();
    }
  
    /**
     * Download studies in RIS format
     * @param ids Array of NCT IDs to download
     */
    async downloadRIS(ids: string[]): Promise<string> {
      const response = await this.request('/studies/download', {
        method: 'GET',
        params: { 
          ids: ids.join(','),
          format: 'ris'
        },
        headers: {
          ...this.defaultHeaders,
          'Accept': 'application/x-research-info-systems'
        }
      });
      
      return response.text();
    }
  
    /**
     * Build a search query string from a SearchQuery object
     * @param query The search query object
     */
    private buildSearchQuery(query: SearchQuery): string {
      // Implementation of the query builder
      if (query.expression) {
        return query.expression;
      }
      
      // Build from components
      let queryParts: string[] = [];
      
      if (query.term) {
        queryParts.push(query.term);
      }
      
      if (query.condition) {
        queryParts.push(`AREA[ConditionSearch]${query.condition}`);
      }
      
      if (query.intervention) {
        queryParts.push(`AREA[InterventionSearch]${query.intervention}`);
      }
      
      if (query.title) {
        queryParts.push(`AREA[TitleSearch]${query.title}`);
      }
      
      if (query.outcome) {
        queryParts.push(`AREA[OutcomeSearch]${query.outcome}`);
      }
      
      if (query.sponsor) {
        queryParts.push(`AREA[SponsorSearch]${query.sponsor}`);
      }
      
      if (query.location) {
        queryParts.push(`AREA[LocationSearch]${query.location}`);
      }
      
      if (query.id) {
        queryParts.push(`AREA[IdSearch]${query.id}`);
      }
      
      if (query.phase) {
        queryParts.push(`AREA[Phase]${query.phase}`);
      }
      
      if (query.studyType) {
        queryParts.push(`AREA[StudyType]${query.studyType}`);
      }
      
      // Add custom field searches
      if (query.fields) {
        for (const [field, value] of Object.entries(query.fields)) {
          queryParts.push(`AREA[${field}]${value}`);
        }
      }
      
      // Combine with AND by default
      return queryParts.join(' AND ');
    }
  
    /**
     * Make a GET request to the API
     * @param endpoint API endpoint
     * @param params Query parameters
     */
    private async get<T>(endpoint: string, params: Record<string, any> = {}): Promise<T> {
      return this.request(endpoint, {
        method: 'GET',
        params
      });
    }
  
    /**
     * Make a request to the API
     * @param endpoint API endpoint
     * @param options Request options
     */
    private async request<T>(
      endpoint: string, 
      options: {
        method: string,
        params?: Record<string, any>,
        headers?: Record<string, string>,
        body?: any
      }
    ): Promise<any> {
      const url = new URL(this.baseUrl + endpoint);
      
      // Add query parameters
      if (options.params) {
        Object.entries(options.params).forEach(([key, value]) => {
          if (value !== undefined && value !== null) {
            url.searchParams.append(key, String(value));
          }
        });
      }
      
      // Merge headers
      const headers = {
        ...this.defaultHeaders,
        ...(options.headers || {})
      };
      
      // Make the request
      const response = await fetch(url.toString(), {
        method: options.method,
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined
      });
      
      // Handle errors
      if (!response.ok) {
        let errorMessage: string;
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || `API error: ${response.status}`;
        } catch (e) {
          errorMessage = `API error: ${response.status} ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }
      
      // Parse JSON response if applicable
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return response.json();
      }
      
      return response;
    }
  }
  
  // Type definitions
  
  export interface VersionInfo {
    version: string;
    dataTimestamp: string;
  }
  
  export interface SearchParams {
    fields?: string[];
    pageSize?: number;
    pageToken?: string;
    sort?: {
      field: string;
      order: 'asc' | 'desc';
    };
    countTotal?: boolean;
  }
  
  export interface StudyParams {
    fields?: string[];
  }
  
  export interface SearchQuery {
    expression?: string;
    term?: string;
    condition?: string;
    intervention?: string;
    title?: string;
    outcome?: string;
    sponsor?: string;
    location?: string;
    id?: string;
    phase?: string;
    studyType?: string;
    fields?: Record<string, string>;
  }
  
  export interface SearchResponse {
    studies: Study[];
    nextPageToken?: string;
    totalCount?: number;
  }
  
  export interface Study {
    protocolSection: ProtocolSection;
    resultsSection?: ResultsSection;
    documentSection?: DocumentSection;
    derivedSection?: DerivedSection;
    annotationSection?: AnnotationSection;
    hasResults: boolean;
  }
  
  // These interfaces would be expanded with complete field definitions
  // from the Study Data Structure documentation
  export interface ProtocolSection {
    identificationModule: {
      nctId: string;
      briefTitle: string;
      officialTitle?: string;
      acronym?: string;
      organization?: any;
      // ...other fields
    };
    statusModule?: any;
    sponsorCollaboratorsModule?: any;
    oversightModule?: any;
    descriptionModule?: any;
    conditionsModule?: any;
    designModule?: any;
    armsInterventionsModule?: any;
    outcomesModule?: any;
    eligibilityModule?: any;
    contactsLocationsModule?: any;
    referencesModule?: any;
    ipdSharingStatementModule?: any;
    // ...other modules
  }
  
  export interface ResultsSection {
    // Implementation based on API documentation
  }
  
  export interface DocumentSection {
    // Implementation based on API documentation
  }
  
  export interface DerivedSection {
    // Implementation based on API documentation
  }
  
  export interface AnnotationSection {
    // Implementation based on API documentation
  }
  
  export interface MetadataResponse {
    // Implementation based on API documentation
  }
  
  export interface SearchAreasResponse {
    // Implementation based on API documentation
  }
  
  export interface EnumsResponse {
    // Implementation based on API documentation
  }
  
  export interface StudySizesResponse {
    // Implementation based on API documentation
  }
  
  export interface FieldValuesParams {
    field: string;
    term?: string;
    limit?: number;
  }
  
  export interface FieldValuesResponse {
    // Implementation based on API documentation
  }
  
  export interface FieldSizesParams {
    field: string;
  }
  
  export interface FieldSizesResponse {
    // Implementation based on API documentation
  }