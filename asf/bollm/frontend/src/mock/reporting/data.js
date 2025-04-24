// Mock data for the reporting system

export const dimensions = [
  {
    id: 'patient_id',
    name: 'Patient ID',
    description: 'Unique identifier for the patient',
    category: 'patient',
    data_type: 'string',
    table: 'patients',
    column: 'id'
  },
  {
    id: 'patient_age',
    name: 'Patient Age',
    description: 'Age of the patient',
    category: 'patient',
    data_type: 'number',
    table: 'patients',
    column: 'age'
  },
  {
    id: 'patient_gender',
    name: 'Patient Gender',
    description: 'Gender of the patient',
    category: 'patient',
    data_type: 'string',
    table: 'patients',
    column: 'gender'
  },
  {
    id: 'diagnosis_code',
    name: 'Diagnosis Code',
    description: 'ICD-10 diagnosis code',
    category: 'diagnosis',
    data_type: 'string',
    table: 'diagnoses',
    column: 'code'
  },
  {
    id: 'diagnosis_date',
    name: 'Diagnosis Date',
    description: 'Date of diagnosis',
    category: 'diagnosis',
    data_type: 'date',
    table: 'diagnoses',
    column: 'date'
  },
  {
    id: 'medication_name',
    name: 'Medication Name',
    description: 'Name of the medication',
    category: 'medication',
    data_type: 'string',
    table: 'medications',
    column: 'name'
  },
  {
    id: 'medication_dosage',
    name: 'Medication Dosage',
    description: 'Dosage of the medication',
    category: 'medication',
    data_type: 'string',
    table: 'medications',
    column: 'dosage'
  },
  {
    id: 'provider_id',
    name: 'Provider ID',
    description: 'Unique identifier for the provider',
    category: 'provider',
    data_type: 'string',
    table: 'providers',
    column: 'id'
  },
  {
    id: 'provider_specialty',
    name: 'Provider Specialty',
    description: 'Specialty of the provider',
    category: 'provider',
    data_type: 'string',
    table: 'providers',
    column: 'specialty'
  }
];

export const metrics = [
  {
    id: 'visit_count',
    name: 'Visit Count',
    description: 'Number of visits',
    category: 'utilization',
    data_type: 'number',
    table: 'visits',
    column: 'id',
    aggregation: 'count',
    format: 'number'
  },
  {
    id: 'total_cost',
    name: 'Total Cost',
    description: 'Total cost of care',
    category: 'financial',
    data_type: 'number',
    table: 'claims',
    column: 'amount',
    aggregation: 'sum',
    format: 'currency'
  },
  {
    id: 'average_cost',
    name: 'Average Cost',
    description: 'Average cost per visit',
    category: 'financial',
    data_type: 'number',
    table: 'claims',
    column: 'amount',
    aggregation: 'avg',
    format: 'currency'
  },
  {
    id: 'medication_count',
    name: 'Medication Count',
    description: 'Number of medications prescribed',
    category: 'medication',
    data_type: 'number',
    table: 'medications',
    column: 'id',
    aggregation: 'count',
    format: 'number'
  },
  {
    id: 'readmission_rate',
    name: 'Readmission Rate',
    description: 'Rate of readmission within 30 days',
    category: 'quality',
    data_type: 'number',
    table: 'admissions',
    column: 'readmission',
    aggregation: 'avg',
    format: 'percent'
  },
  {
    id: 'length_of_stay',
    name: 'Length of Stay',
    description: 'Average length of stay in days',
    category: 'utilization',
    data_type: 'number',
    table: 'admissions',
    column: 'length_of_stay',
    aggregation: 'avg',
    format: 'number'
  }
];

export const dimensionCategories = [
  {
    id: 'patient',
    name: 'Patient',
    description: 'Patient demographics and attributes'
  },
  {
    id: 'diagnosis',
    name: 'Diagnosis',
    description: 'Diagnosis information'
  },
  {
    id: 'medication',
    name: 'Medication',
    description: 'Medication information'
  },
  {
    id: 'provider',
    name: 'Provider',
    description: 'Provider information'
  }
];

export const metricCategories = [
  {
    id: 'utilization',
    name: 'Utilization',
    description: 'Metrics related to healthcare utilization'
  },
  {
    id: 'financial',
    name: 'Financial',
    description: 'Metrics related to costs and payments'
  },
  {
    id: 'medication',
    name: 'Medication',
    description: 'Metrics related to medications'
  },
  {
    id: 'quality',
    name: 'Quality',
    description: 'Metrics related to quality of care'
  }
];

export const savedReports = [
  {
    id: 1,
    name: 'Patient Demographics Report',
    description: 'Overview of patient demographics',
    report_definition: {
      dimensions: ['patient_gender', 'patient_age'],
      metrics: ['visit_count', 'total_cost'],
      filters: {},
      sort_by: 'visit_count',
      sort_order: 'desc',
      limit: 100
    },
    owner: 'admin',
    created_at: '2023-01-15T10:30:00Z',
    updated_at: '2023-01-15T10:30:00Z'
  },
  {
    id: 2,
    name: 'Provider Performance Report',
    description: 'Performance metrics by provider',
    report_definition: {
      dimensions: ['provider_id', 'provider_specialty'],
      metrics: ['visit_count', 'average_cost', 'readmission_rate'],
      filters: {},
      sort_by: 'readmission_rate',
      sort_order: 'asc',
      limit: 50
    },
    owner: 'admin',
    created_at: '2023-02-20T14:45:00Z',
    updated_at: '2023-03-10T09:15:00Z'
  },
  {
    id: 3,
    name: 'Medication Usage Report',
    description: 'Analysis of medication usage patterns',
    report_definition: {
      dimensions: ['medication_name', 'medication_dosage'],
      metrics: ['medication_count', 'total_cost'],
      filters: {},
      sort_by: 'medication_count',
      sort_order: 'desc',
      limit: 20
    },
    owner: 'admin',
    created_at: '2023-03-05T11:20:00Z',
    updated_at: '2023-03-05T11:20:00Z'
  }
];

export const scheduledReports = [
  {
    id: 1,
    name: 'Weekly Patient Demographics',
    description: 'Weekly report of patient demographics',
    report_id: 1,
    frequency: 'weekly',
    time: '08:00:00',
    days_of_week: [1], // Monday
    export_format: 'excel',
    recipients: ['admin@example.com', 'manager@example.com'],
    owner: 'admin',
    created_at: '2023-01-20T15:30:00Z',
    updated_at: '2023-01-20T15:30:00Z',
    last_run: '2023-04-17T08:00:00Z',
    next_run: '2023-04-24T08:00:00Z'
  },
  {
    id: 2,
    name: 'Monthly Provider Performance',
    description: 'Monthly report of provider performance',
    report_id: 2,
    frequency: 'monthly',
    time: '09:00:00',
    days_of_month: [1], // First day of month
    export_format: 'pdf',
    recipients: ['admin@example.com', 'director@example.com'],
    owner: 'admin',
    created_at: '2023-02-25T10:15:00Z',
    updated_at: '2023-02-25T10:15:00Z',
    last_run: '2023-04-01T09:00:00Z',
    next_run: '2023-05-01T09:00:00Z'
  }
];

export const sampleReportData = {
  dimensions: ['patient_gender', 'patient_age'],
  metrics: ['visit_count', 'total_cost'],
  rows: [
    { patient_gender: 'Male', patient_age: '0-17', visit_count: 120, total_cost: 15000 },
    { patient_gender: 'Male', patient_age: '18-34', visit_count: 250, total_cost: 35000 },
    { patient_gender: 'Male', patient_age: '35-50', visit_count: 310, total_cost: 48000 },
    { patient_gender: 'Male', patient_age: '51-65', visit_count: 280, total_cost: 52000 },
    { patient_gender: 'Male', patient_age: '65+', visit_count: 350, total_cost: 75000 },
    { patient_gender: 'Female', patient_age: '0-17', visit_count: 140, total_cost: 16500 },
    { patient_gender: 'Female', patient_age: '18-34', visit_count: 320, total_cost: 42000 },
    { patient_gender: 'Female', patient_age: '35-50', visit_count: 380, total_cost: 56000 },
    { patient_gender: 'Female', patient_age: '51-65', visit_count: 290, total_cost: 54000 },
    { patient_gender: 'Female', patient_age: '65+', visit_count: 410, total_cost: 82000 }
  ],
  totals: { visit_count: 2850, total_cost: 475500 }
};

export const sampleReportResult = {
  report_definition: {
    dimensions: ['patient_gender', 'patient_age'],
    metrics: ['visit_count', 'total_cost'],
    filters: {},
    sort_by: 'visit_count',
    sort_order: 'desc',
    limit: 100
  },
  data: sampleReportData,
  execution_time: '0.35 seconds',
  row_count: 10
};
