// Type declarations for the mock reporting API

import { 
  Dimension, 
  Metric, 
  DimensionCategory, 
  MetricCategory, 
  ReportDefinition, 
  SavedReport, 
  ReportResult, 
  ReportSchedule,
  ExportStatus
} from '../../types/reporting';

export function mockGetDimensions(category?: string): Promise<Dimension[]>;
export function mockGetDimensionCategories(): Promise<DimensionCategory[]>;
export function mockGetDimensionValues(dimensionId: string, search?: string, limit?: number): Promise<any[]>;

export function mockGetMetrics(category?: string): Promise<Metric[]>;
export function mockGetMetricCategories(): Promise<MetricCategory[]>;

export function mockExecuteReport(reportDefinition: ReportDefinition): Promise<ReportResult>;
export function mockSaveReport(report: SavedReport): Promise<SavedReport>;
export function mockGetSavedReports(): Promise<SavedReport[]>;
export function mockGetSavedReport(reportId: number): Promise<SavedReport>;
export function mockUpdateReport(reportId: number, report: SavedReport): Promise<SavedReport>;
export function mockDeleteReport(reportId: number): Promise<void>;

export function mockScheduleReport(schedule: ReportSchedule): Promise<ReportSchedule>;
export function mockGetScheduledReports(): Promise<ReportSchedule[]>;
export function mockGetScheduledReport(scheduleId: number): Promise<ReportSchedule>;
export function mockUpdateSchedule(scheduleId: number, schedule: ReportSchedule): Promise<ReportSchedule>;
export function mockDeleteSchedule(scheduleId: number): Promise<void>;

export function mockExportReport(reportId: number, format: string): Promise<ExportStatus>;
export function mockGetExportStatus(taskId: string): Promise<ExportStatus>;
