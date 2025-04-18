/**
 * ClinicalTrials.gov Study Data Model
 * 
 * Type definitions for the ClinicalTrials.gov API study data structure
 * Based on the Study Data Structure documentation.
 */

export interface Study {
    protocolSection: ProtocolSection;
    resultsSection?: ResultsSection;
    annotationSection?: AnnotationSection;
    documentSection?: DocumentSection;
    derivedSection?: DerivedSection;
    hasResults: boolean;
  }
  
  // Protocol Section
  export interface ProtocolSection {
    identificationModule: IdentificationModule;
    statusModule?: StatusModule;
    sponsorCollaboratorsModule?: SponsorCollaboratorsModule;
    oversightModule?: OversightModule;
    descriptionModule?: DescriptionModule;
    conditionsModule?: ConditionsModule;
    designModule?: DesignModule;
    armsInterventionsModule?: ArmsInterventionsModule;
    outcomesModule?: OutcomesModule;
    eligibilityModule?: EligibilityModule;
    contactsLocationsModule?: ContactsLocationsModule;
    referencesModule?: ReferencesModule;
    ipdSharingStatementModule?: IPDSharingStatementModule;
  }
  
  // Identification Module
  export interface IdentificationModule {
    nctId: string;
    nctIdAliases?: string[];
    orgStudyIdInfo?: OrgStudyIdInfo;
    secondaryIdInfos?: SecondaryIdInfo[];
    briefTitle?: string;
    officialTitle?: string;
    acronym?: string;
    organization?: Organization;
  }
  
  export interface OrgStudyIdInfo {
    id?: string;
    type?: string;
    link?: string;
  }
  
  export interface SecondaryIdInfo {
    id?: string;
    type?: string;
    domain?: string;
    link?: string;
  }
  
  export interface Organization {
    fullName?: string;
    class?: string;
  }
  
  // Status Module
  export interface StatusModule {
    statusVerifiedDate?: string;
    overallStatus?: string;
    lastKnownStatus?: string;
    delayedPosting?: boolean;
    whyStopped?: string;
    expandedAccessInfo?: ExpandedAccessInfo;
    startDateStruct?: PartialDateStruct;
    primaryCompletionDateStruct?: PartialDateStruct;
    completionDateStruct?: PartialDateStruct;
    studyFirstSubmitDate?: string;
    studyFirstSubmitQcDate?: string;
    studyFirstPostDateStruct?: DateStruct;
    resultsFirstSubmitDate?: string;
    resultsFirstSubmitQcDate?: string;
    resultsFirstPostDateStruct?: DateStruct;
    dispFirstSubmitDate?: string;
    dispFirstSubmitQcDate?: string;
    dispFirstPostDateStruct?: DateStruct;
    lastUpdateSubmitDate?: string;
    lastUpdatePostDateStruct?: DateStruct;
  }
  
  export interface ExpandedAccessInfo {
    hasExpandedAccess?: boolean;
    nctId?: string;
    statusForNctId?: string;
  }
  
  export interface PartialDateStruct {
    date?: string;
    type?: string;
  }
  
  export interface DateStruct {
    date?: string;
    type?: string;
  }
  
  // Sponsor/Collaborators Module
  export interface SponsorCollaboratorsModule {
    responsibleParty?: ResponsibleParty;
    leadSponsor?: Sponsor;
    collaborators?: Sponsor[];
  }
  
  export interface ResponsibleParty {
    type?: string;
    investigatorFullName?: string;
    investigatorTitle?: string;
    investigatorAffiliation?: string;
    oldNameTitle?: string;
    oldOrganization?: string;
  }
  
  export interface Sponsor {
    name?: string;
    class?: string;
  }
  
  // Oversight Module
  export interface OversightModule {
    oversightHasDmc?: boolean;
    isFdaRegulatedDrug?: boolean;
    isFdaRegulatedDevice?: boolean;
    isUnapprovedDevice?: boolean;
    isPpsd?: boolean;
    isUsExport?: boolean;
    fdaaa801Violation?: boolean;
  }
  
  // Description Module
  export interface DescriptionModule {
    briefSummary?: string;
    detailedDescription?: string;
  }
  
  // Conditions Module
  export interface ConditionsModule {
    conditions?: string[];
    keywords?: string[];
  }
  
  // Design Module
  export interface DesignModule {
    studyType?: string;
    patientRegistry?: boolean;
    targetDuration?: string;
    phases?: string[];
    designInfo?: DesignInfo;
    bioSpec?: BioSpec;
    enrollmentInfo?: EnrollmentInfo;
  }
  
  export interface DesignInfo {
    allocation?: string;
    interventionModel?: string;
    interventionModelDescription?: string;
    primaryPurpose?: string;
    observationalModel?: string;
    timePerspective?: string;
    maskingInfo?: MaskingBlock;
  }
  
  export interface MaskingBlock {
    masking?: string;
    maskingDescription?: string;
    whoMasked?: string[];
  }
  
  export interface BioSpec {
    retention?: string;
    description?: string;
  }
  
  export interface EnrollmentInfo {
    count?: number;
    type?: string;
  }
  
  // Arms/Interventions Module
  export interface ArmsInterventionsModule {
    armGroups?: ArmGroup[];
    interventions?: Intervention[];
  }
  
  export interface ArmGroup {
    label?: string;
    type?: string;
    description?: string;
    interventionNames?: string[];
  }
  
  export interface Intervention {
    type?: string;
    name?: string;
    description?: string;
    armGroupLabels?: string[];
    otherNames?: string[];
  }
  
  // Outcomes Module
  export interface OutcomesModule {
    primaryOutcomes?: Outcome[];
    secondaryOutcomes?: Outcome[];
    otherOutcomes?: Outcome[];
  }
  
  export interface Outcome {
    measure?: string;
    description?: string;
    timeFrame?: string;
  }
  
  // Eligibility Module
  export interface EligibilityModule {
    eligibilityCriteria?: string;
    healthyVolunteers?: boolean;
    sex?: string;
    genderBased?: boolean;
    genderDescription?: string;
    minimumAge?: string;
    maximumAge?: string;
    stdAges?: string[];
    studyPopulation?: string;
    samplingMethod?: string;
  }
  
  // Contacts/Locations Module
  export interface ContactsLocationsModule {
    centralContacts?: Contact[];
    overallOfficials?: Official[];
    locations?: Location[];
  }
  
  export interface Contact {
    name?: string;
    role?: string;
    phone?: string;
    phoneExt?: string;
    email?: string;
  }
  
  export interface Official {
    name?: string;
    affiliation?: string;
    role?: string;
  }
  
  export interface Location {
    facility?: string;
    status?: string;
    city?: string;
    state?: string;
    zip?: string;
    country?: string;
    contacts?: Contact[];
    geoPoint?: GeoPoint;
  }
  
  export interface GeoPoint {
    lat: number;
    lon: number;
  }
  
  // References Module
  export interface ReferencesModule {
    references?: Reference[];
    seeAlsoLinks?: SeeAlsoLink[];
    availIpds?: AvailIpd[];
  }
  
  export interface Reference {
    pmid?: string;
    type?: string;
    citation?: string;
    retractions?: Retraction[];
  }
  
  export interface Retraction {
    pmid?: string;
    source?: string;
  }
  
  export interface SeeAlsoLink {
    label?: string;
    url?: string;
  }
  
  export interface AvailIpd {
    id?: string;
    type?: string;
    url?: string;
    comment?: string;
  }
  
  // IPD Sharing Statement Module
  export interface IPDSharingStatementModule {
    ipdSharing?: string;
    description?: string;
    infoTypes?: string[];
    timeFrame?: string;
    accessCriteria?: string;
    url?: string;
  }
  
  // Results Section
  export interface ResultsSection {
    participantFlowModule?: ParticipantFlowModule;
    baselineCharacteristicsModule?: BaselineCharacteristicsModule;
    outcomeMeasuresModule?: OutcomeMeasuresModule;
    adverseEventsModule?: AdverseEventsModule;
    moreInfoModule?: MoreInfoModule;
  }
  
  // Participant Flow Module
  export interface ParticipantFlowModule {
    preAssignmentDetails?: string;
    recruitmentDetails?: string;
    typeUnitsAnalyzed?: string;
    groups?: FlowGroup[];
    periods?: FlowPeriod[];
  }
  
  export interface FlowGroup {
    id?: string;
    title?: string;
    description?: string;
  }
  
  export interface FlowPeriod {
    title?: string;
    milestones?: FlowMilestone[];
    dropWithdraws?: DropWithdraw[];
  }
  
  export interface FlowMilestone {
    type?: string;
    comment?: string;
    achievements?: FlowStats[];
  }
  
  export interface FlowStats {
    groupId?: string;
    comment?: string;
    numSubjects?: string;
    numUnits?: string;
  }
  
  export interface DropWithdraw {
    type?: string;
    comment?: string;
    reasons?: FlowStats[];
  }
  
  // Baseline Characteristics Module
  export interface BaselineCharacteristicsModule {
    populationDescription?: string;
    typeUnitsAnalyzed?: string;
    groups?: MeasureGroup[];
    denoms?: Denom[];
    measures?: BaselineMeasure[];
  }
  
  export interface MeasureGroup {
    id?: string;
    title?: string;
    description?: string;
  }
  
  export interface Denom {
    units?: string;
    counts?: DenomCount[];
  }
  
  export interface DenomCount {
    groupId?: string;
    value?: string;
  }
  
  export interface BaselineMeasure {
    title?: string;
    description?: string;
    populationDescription?: string;
    paramType?: string;
    dispersionType?: string;
    unitOfMeasure?: string;
    calculatePct?: boolean;
    denomUnitsSelected?: string;
    denoms?: Denom[];
    classes?: MeasureClass[];
  }
  
  export interface MeasureClass {
    title?: string;
    denoms?: Denom[];
    categories?: MeasureCategory[];
  }
  
  export interface MeasureCategory {
    title?: string;
    measurements?: Measurement[];
  }
  
  export interface Measurement {
    groupId?: string;
    value?: string;
    spread?: string;
    lowerLimit?: string;
    upperLimit?: string;
    comment?: string;
  }
  
  // Outcome Measures Module
  export interface OutcomeMeasuresModule {
    outcomeMeasures?: OutcomeMeasure[];
  }
  
  export interface OutcomeMeasure {
    type?: string;
    title?: string;
    description?: string;
    populationDescription?: string;
    reportingStatus?: string;
    anticipatedPostingDate?: string;
    paramType?: string;
    dispersionType?: string;
    unitOfMeasure?: string;
    calculatePct?: boolean;
    timeFrame?: string;
    typeUnitsAnalyzed?: string;
    denomUnitsSelected?: string;
    groups?: MeasureGroup[];
    denoms?: Denom[];
    classes?: MeasureClass[];
    analyses?: MeasureAnalysis[];
  }
  
  export interface MeasureAnalysis {
    paramType?: string;
    paramValue?: string;
    dispersionType?: string;
    dispersionValue?: string;
    statisticalMethod?: string;
    statisticalComment?: string;
    pValue?: string;
    pValueComment?: string;
    ciNumSides?: string;
    ciPctValue?: string;
    ciLowerLimit?: string;
    ciUpperLimit?: string;
    ciLowerLimitComment?: string;
    ciUpperLimitComment?: string;
    estimateComment?: string;
    testedNonInferiority?: boolean;
    nonInferiorityType?: string;
    nonInferiorityComment?: string;
    otherAnalysisDescription?: string;
    groupDescription?: string;
    groupIds?: string[];
  }
  
  // Adverse Events Module
  export interface AdverseEventsModule {
    frequencyThreshold?: string;
    timeFrame?: string;
    description?: string;
    allCauseMortalityComment?: string;
    eventGroups?: EventGroup[];
    seriousEvents?: AdverseEvent[];
    otherEvents?: AdverseEvent[];
  }
  
  export interface EventGroup {
    id?: string;
    title?: string;
    description?: string;
    deathsNumAffected?: number;
    deathsNumAtRisk?: number;
    seriousNumAffected?: number;
    seriousNumAtRisk?: number;
    otherNumAffected?: number;
    otherNumAtRisk?: number;
  }
  
  export interface AdverseEvent {
    term?: string;
    organSystem?: string;
    sourceVocabulary?: string;
    assessmentType?: string;
    notes?: string;
    stats?: EventStats[];
  }
  
  export interface EventStats {
    groupId?: string;
    numEvents?: number;
    numAffected?: number;
    numAtRisk?: number;
  }
  
  // More Info Module
  export interface MoreInfoModule {
    limitationsAndCaveats?: LimitationsAndCaveats;
    certainAgreement?: CertainAgreement;
    pointOfContact?: PointOfContact;
  }
  
  export interface LimitationsAndCaveats {
    description?: string;
  }
  
  export interface CertainAgreement {
    piSponsorEmployee?: boolean;
    restrictionType?: string;
    restrictiveAgreement?: boolean;
    otherDetails?: string;
  }
  
  export interface PointOfContact {
    title?: string;
    organization?: string;
    email?: string;
    phone?: string;
    phoneExt?: string;
  }
  
  // Annotation Section
  export interface AnnotationSection {
    annotationModule?: AnnotationModule;
  }
  
  export interface AnnotationModule {
    unpostedAnnotation?: UnpostedAnnotation;
    violationAnnotation?: ViolationAnnotation;
  }
  
  export interface UnpostedAnnotation {
    unpostedResponsibleParty?: string;
    unpostedEvents?: UnpostedEvent[];
  }
  
  export interface UnpostedEvent {
    type?: string;
    date?: string;
    dateUnknown?: boolean;
  }
  
  export interface ViolationAnnotation {
    violationEvents?: ViolationEvent[];
  }
  
  export interface ViolationEvent {
    type?: string;
    description?: string;
    creationDate?: string;
    issuedDate?: string;
    releaseDate?: string;
    postedDate?: string;
  }
  
  // Document Section
  export interface DocumentSection {
    largeDocumentModule?: LargeDocumentModule;
  }
  
  export interface LargeDocumentModule {
    noSap?: boolean;
    largeDocs?: LargeDoc[];
  }
  
  export interface LargeDoc {
    typeAbbrev?: string;
    hasProtocol?: boolean;
    hasSap?: boolean;
    hasIcf?: boolean;
    label?: string;
    date?: string;
    uploadDate?: string;
    filename?: string;
    size?: number;
  }
  
  // Derived Section
  export interface DerivedSection {
    miscInfoModule?: MiscInfoModule;
    conditionBrowseModule?: BrowseModule;
    interventionBrowseModule?: BrowseModule;
  }
  
  export interface MiscInfoModule {
    versionHolder?: string;
    removedCountries?: string[];
    submissionTracking?: SubmissionTracking;
  }
  
  export interface SubmissionTracking {
    estimatedResultsFirstSubmitDate?: string;
    firstMcpInfo?: FirstMcpInfo;
    submissionInfos?: SubmissionInfo[];
  }
  
  export interface FirstMcpInfo {
    postDateStruct?: DateStruct;
  }
  
  export interface SubmissionInfo {
    releaseDate?: string;
    unreleaseDate?: string;
    unreleaseDateUnknown?: boolean;
    resetDate?: string;
    mcpReleaseN?: number;
  }
  
  export interface BrowseModule {
    meshes?: Mesh[];
    ancestors?: Mesh[];
    browseLeaves?: BrowseLeaf[];
    browseBranches?: BrowseBranch[];
  }
  
  export interface Mesh {
    id?: string;
    term?: string;
  }
  
  export interface BrowseLeaf {
    id?: string;
    name?: string;
    asFound?: string;
    relevance?: string;
  }
  
  export interface BrowseBranch {
    abbrev?: string;
    name?: string;
  }