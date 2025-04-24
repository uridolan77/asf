// Types for the reporting system

export enum ReportExportFormat {
  CSV = 'csv',
  EXCEL = 'excel',
  PDF = 'pdf',
  JSON = 'json'
}

export interface Dimension {
  id: string;
  name: string;
  description?: string;
  category?: string;
  data_type: string;
  table: string;
  column: string;
}

export interface Metric {
  id: string;
  name: string;
  description?: string;
  category?: string;
  data_type: string;
  table: string;
  column: string;
  aggregation: string;
  format?: string;
}

export interface DimensionCategory {
  id: string;
  name: string;
  description?: string;
}

export interface MetricCategory {
  id: string;
  name: string;
  description?: string;
}

export interface ReportDefinition {
  dimensions: string[];
  metrics: string[];
  filters?: Record<string, any>;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  limit?: number;
}

export interface SavedReport {
  id?: number;
  name: string;
  description?: string;
  report_definition: ReportDefinition | Record<string, any>;
  owner?: string;
  created_at?: string;
  updated_at?: string;
}

export interface ReportData {
  dimensions: string[];
  metrics: string[];
  rows: Record<string, any>[];
  totals: Record<string, any>;
}

export interface ReportResult {
  report_definition: ReportDefinition;
  data: ReportData;
  execution_time: string;
  row_count: number;
}

export interface ReportSchedule {
  id?: number;
  name: string;
  description?: string;
  report_id: number;
  frequency: 'daily' | 'weekly' | 'monthly';
  time: string;  // HH:MM:SS
  days_of_week?: number[];  // 0=Monday to 6=Sunday
  days_of_month?: number[];  // 1-31
  export_format: ReportExportFormat;
  recipients: string[];
  owner?: string;
  created_at?: string;
  updated_at?: string;
  last_run?: string;
  next_run?: string;
}

export interface ExportStatus {
  task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  username: string;
  format: string;
  created_at: string;
  completed_at?: string;
  download_url?: string;
  error?: string;
}
