/**
 * ClinicalTrials.gov Search Query Builder
 * 
 * Provides a fluent interface for building complex search queries for the ClinicalTrials.gov API
 * Based on the query syntax documented in the API.
 */

export class SearchQueryBuilder {
    private queryParts: string[] = [];
  
    /**
     * Creates a new search query builder
     */
    constructor() {}
  
    /**
     * Add a term to search across all default fields
     * @param term The term to search for
     */
    term(term: string): SearchQueryBuilder {
      this.queryParts.push(this.formatTerm(term));
      return this;
    }
  
    /**
     * Search for a specific condition/disease
     * @param condition The condition to search for
     */
    condition(condition: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[ConditionSearch]${this.formatTerm(condition)}`);
      return this;
    }
  
    /**
     * Search for a specific intervention/treatment
     * @param intervention The intervention to search for
     */
    intervention(intervention: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[InterventionSearch]${this.formatTerm(intervention)}`);
      return this;
    }
  
    /**
     * Search for a specific title or acronym
     * @param title The title or acronym to search for
     */
    title(title: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[TitleSearch]${this.formatTerm(title)}`);
      return this;
    }
  
    /**
     * Search for a specific outcome measure
     * @param outcome The outcome measure to search for
     */
    outcome(outcome: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[OutcomeSearch]${this.formatTerm(outcome)}`);
      return this;
    }
  
    /**
     * Search for a specific sponsor or collaborator
     * @param sponsor The sponsor or collaborator to search for
     */
    sponsor(sponsor: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[SponsorSearch]${this.formatTerm(sponsor)}`);
      return this;
    }
  
    /**
     * Search for a specific location (city, state, country)
     * @param location The location to search for
     */
    location(location: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[LocationSearch]${this.formatTerm(location)}`);
      return this;
    }
  
    /**
     * Search for a specific NCT ID or other study identifier
     * @param id The ID to search for
     */
    id(id: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[IdSearch]${this.formatTerm(id)}`);
      return this;
    }
  
    /**
     * Filter by study type
     * @param type The study type (e.g., INTERVENTIONAL, OBSERVATIONAL)
     */
    studyType(type: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[StudyType]${type}`);
      return this;
    }
  
    /**
     * Filter by study phase
     * @param phase The study phase (e.g., PHASE1, PHASE2)
     */
    phase(phase: string | string[]): SearchQueryBuilder {
      if (Array.isArray(phase)) {
        const phases = phase.map(p => `AREA[Phase]${p}`);
        this.queryParts.push(`(${phases.join(' OR ')})`);
      } else {
        this.queryParts.push(`AREA[Phase]${phase}`);
      }
      return this;
    }
  
    /**
     * Filter by recruitment status
     * @param status The recruitment status (e.g., RECRUITING, COMPLETED)
     */
    status(status: string | string[]): SearchQueryBuilder {
      if (Array.isArray(status)) {
        const statuses = status.map(s => `AREA[OverallStatus]${s}`);
        this.queryParts.push(`(${statuses.join(' OR ')})`);
      } else {
        this.queryParts.push(`AREA[OverallStatus]${status}`);
      }
      return this;
    }
  
    /**
     * Filter by eligible gender
     * @param gender The eligible gender (MALE, FEMALE, ALL)
     */
    gender(gender: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[Sex]${gender}`);
      return this;
    }
  
    /**
     * Filter by age range
     * @param min Minimum age (e.g., "18 Years")
     * @param max Maximum age (e.g., "65 Years")
     */
    ageRange(min: string, max: string): SearchQueryBuilder {
      this.queryParts.push(`AREA[MinimumAge]RANGE[${min}, ${max}]`);
      return this;
    }
  
    /**
     * Filter by standard age group
     * @param ageGroup The age group (e.g., CHILD, ADULT, OLDER_ADULT)
     */
    ageGroup(ageGroup: string | string[]): SearchQueryBuilder {
      if (Array.isArray(ageGroup)) {
        const groups = ageGroup.map(g => `AREA[StdAge]${g}`);
        this.queryParts.push(`(${groups.join(' OR ')})`);
      } else {
        this.queryParts.push(`AREA[StdAge]${ageGroup}`);
      }
      return this;
    }
  
    /**
     * Filter by funder type
     * @param funderType The funder type (e.g., NIH, INDUSTRY)
     */
    funderType(funderType: string | string[]): SearchQueryBuilder {
      if (Array.isArray(funderType)) {
        const funders = funderType.map(f => `AREA[LeadSponsorClass]${f}`);
        this.queryParts.push(`(${funders.join(' OR ')})`);
      } else {
        this.queryParts.push(`AREA[LeadSponsorClass]${funderType}`);
      }
      return this;
    }
  
    /**
     * Filter by study start date range
     * @param from Start date from (format: MM/DD/YYYY)
     * @param to Start date to (format: MM/DD/YYYY)
     */
    startDateRange(from: string | 'MIN', to: string | 'MAX'): SearchQueryBuilder {
      this.queryParts.push(`AREA[StartDate]RANGE[${from}, ${to}]`);
      return this;
    }
  
    /**
     * Filter by primary completion date range
     * @param from Completion date from (format: MM/DD/YYYY)
     * @param to Completion date to (format: MM/DD/YYYY)
     */
    completionDateRange(from: string | 'MIN', to: string | 'MAX'): SearchQueryBuilder {
      this.queryParts.push(`AREA[PrimaryCompletionDate]RANGE[${from}, ${to}]`);
      return this;
    }
  
    /**
     * Filter by first posted date range
     * @param from First posted date from (format: MM/DD/YYYY)
     * @param to First posted date to (format: MM/DD/YYYY)
     */
    firstPostedDateRange(from: string | 'MIN', to: string | 'MAX'): SearchQueryBuilder {
      this.queryParts.push(`AREA[StudyFirstPostDate]RANGE[${from}, ${to}]`);
      return this;
    }
  
    /**
     * Filter for studies with results
     */
    hasResults(hasResults: boolean = true): SearchQueryBuilder {
      this.queryParts.push(`AREA[HasResults]${hasResults ? 'true' : 'false'}`);
      return this;
    }
  
    /**
     * Create a search for a specific field value
     * @param field The field name
     * @param value The value to search for
     * @param expansion Optional expansion type
     */
    field(field: string, value: string, expansion?: ExpansionType): SearchQueryBuilder {
      let query = `AREA[${field}]`;
      
      if (expansion) {
        query += `EXPANSION[${expansion}]`;
      }
      
      query += this.formatTerm(value);
      this.queryParts.push(query);
      
      return this;
    }
  
    /**
     * Add a raw expression to the query
     * @param expression The raw expression to add
     */
    raw(expression: string): SearchQueryBuilder {
      this.queryParts.push(expression);
      return this;
    }
  
    /**
     * Combine with AND operator
     * @param builder Another builder or expression to combine with
     */
    and(builder: SearchQueryBuilder | string): SearchQueryBuilder {
      const expr = typeof builder === 'string' 
        ? builder 
        : builder.build();
      
      if (this.queryParts.length === 0) {
        if (typeof builder === 'string') {
          this.queryParts.push(builder);
        } else {
          this.queryParts = builder.queryParts;
        }
      } else if (expr) {
        this.queryParts.push(`AND (${expr})`);
      }
      
      return this;
    }
  
    /**
     * Combine with OR operator
     * @param builder Another builder or expression to combine with
     */
    or(builder: SearchQueryBuilder | string): SearchQueryBuilder {
      const expr = typeof builder === 'string' 
        ? builder 
        : builder.build();
      
      if (this.queryParts.length === 0) {
        if (typeof builder === 'string') {
          this.queryParts.push(builder);
        } else {
          this.queryParts = builder.queryParts;
        }
      } else if (expr) {
        this.queryParts.push(`OR (${expr})`);
      }
      
      return this;
    }
  
    /**
     * Add a NOT operator
     * @param builder Another builder or expression to negate
     */
    not(builder: SearchQueryBuilder | string): SearchQueryBuilder {
      const expr = typeof builder === 'string' 
        ? builder 
        : builder.build();
      
      if (expr) {
        this.queryParts.push(`NOT (${expr})`);
      }
      
      return this;
    }
  
    /**
     * Build a location search that matches all criteria within a single location
     * @param city Optional city name
     * @param state Optional state name
     * @param country Optional country name
     * @param status Optional recruitment status for the location
     */
    locationSearch(
      city?: string, 
      state?: string, 
      country?: string,
      status?: string
    ): SearchQueryBuilder {
      let locationParts: string[] = [];
      
      if (city) {
        locationParts.push(`AREA[LocationCity]${this.formatTerm(city)}`);
      }
      
      if (state) {
        locationParts.push(`AREA[LocationState]${this.formatTerm(state)}`);
      }
      
      if (country) {
        locationParts.push(`AREA[LocationCountry]${this.formatTerm(country)}`);
      }
      
      if (status) {
        locationParts.push(`AREA[LocationStatus]${status}`);
      }
      
      if (locationParts.length > 0) {
        this.queryParts.push(`SEARCH[Location](${locationParts.join(' AND ')})`);
      }
      
      return this;
    }
  
    /**
     * Build the complete search query string
     */
    build(): string {
      return this.queryParts.join(' ');
    }
  
    /**
     * Format a search term, adding quotes if it contains spaces
     * @param term The term to format
     */
    private formatTerm(term: string): string {
      // Escape any quotes in the term
      term = term.replace(/"/g, '\\"');
      
      // Add quotes if the term contains spaces
      if (term.includes(' ')) {
        return `"${term}"`;
      }
      
      return term;
    }
  }
  
  /**
   * Expansion types for search terms
   */
  export type ExpansionType = 'None' | 'Term' | 'Concept' | 'Relaxation' | 'Lossy';
  
  /**
   * Coverage types for search terms
   */
  export type CoverageType = 'FullMatch' | 'StartsWith' | 'EndsWith' | 'Contains';