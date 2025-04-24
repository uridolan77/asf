// API client for the reporting system
import axios from 'axios';
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

// Import mock API implementations
import {
  mockGetDimensions,
  mockGetDimensionCategories,
  mockGetDimensionValues,
  mockGetMetrics,
  mockGetMetricCategories,
  mockExecuteReport,
  mockSaveReport,
  mockGetSavedReports,
  mockGetSavedReport,
  mockUpdateReport,
  mockDeleteReport,
  mockScheduleReport,
  mockGetScheduledReports,
  mockGetScheduledReport,
  mockUpdateSchedule,
  mockDeleteSchedule,
  mockExportReport,
  mockGetExportStatus
} from '../../mock/reporting/api';

const API_BASE_URL = '/api';

// Flag to use mock API instead of real API
const USE_MOCK_API = true;

// Dimensions API
export const getDimensions = async (category?: string): Promise<Dimension[]> => {
  if (USE_MOCK_API) {
    return mockGetDimensions(category);
  }
  const params = category ? { category } : {};
  const response = await axios.get(`${API_BASE_URL}/dimensions`, { params });
  return response.data;
};

export const getDimensionCategories = async (): Promise<DimensionCategory[]> => {
  if (USE_MOCK_API) {
    return mockGetDimensionCategories();
  }
  const response = await axios.get(`${API_BASE_URL}/dimensions/categories`);
  return response.data;
};

export const getDimensionValues = async (dimensionId: string, search?: string, limit: number = 100): Promise<any[]> => {
  if (USE_MOCK_API) {
    return mockGetDimensionValues(dimensionId, search, limit);
  }
  const params = { search, limit };
  const response = await axios.get(`${API_BASE_URL}/dimensions/${dimensionId}/values`, { params });
  return response.data;
};

// Metrics API
export const getMetrics = async (category?: string): Promise<Metric[]> => {
  if (USE_MOCK_API) {
    return mockGetMetrics(category);
  }
  const params = category ? { category } : {};
  const response = await axios.get(`${API_BASE_URL}/metrics`, { params });
  return response.data;
};

export const getMetricCategories = async (): Promise<MetricCategory[]> => {
  if (USE_MOCK_API) {
    return mockGetMetricCategories();
  }
  const response = await axios.get(`${API_BASE_URL}/metrics/categories`);
  return response.data;
};

// Reports API
export const executeReport = async (reportDefinition: ReportDefinition): Promise<ReportResult> => {
  if (USE_MOCK_API) {
    return mockExecuteReport(reportDefinition);
  }
  const response = await axios.post(`${API_BASE_URL}/reports/execute`, reportDefinition);
  return response.data;
};

export const saveReport = async (report: SavedReport): Promise<SavedReport> => {
  if (USE_MOCK_API) {
    return mockSaveReport(report);
  }
  const response = await axios.post(`${API_BASE_URL}/reports/save`, report);
  return response.data;
};

export const getSavedReports = async (): Promise<SavedReport[]> => {
  if (USE_MOCK_API) {
    return mockGetSavedReports();
  }
  const response = await axios.get(`${API_BASE_URL}/reports/saved`);
  return response.data;
};

export const getSavedReport = async (reportId: number): Promise<SavedReport> => {
  if (USE_MOCK_API) {
    return mockGetSavedReport(reportId);
  }
  const response = await axios.get(`${API_BASE_URL}/reports/saved/${reportId}`);
  return response.data;
};

export const updateReport = async (reportId: number, report: SavedReport): Promise<SavedReport> => {
  if (USE_MOCK_API) {
    return mockUpdateReport(reportId, report);
  }
  const response = await axios.put(`${API_BASE_URL}/reports/saved/${reportId}`, report);
  return response.data;
};

export const deleteReport = async (reportId: number): Promise<void> => {
  if (USE_MOCK_API) {
    return mockDeleteReport(reportId);
  }
  await axios.delete(`${API_BASE_URL}/reports/saved/${reportId}`);
};

// Report Scheduling API
export const scheduleReport = async (schedule: ReportSchedule): Promise<ReportSchedule> => {
  if (USE_MOCK_API) {
    return mockScheduleReport(schedule);
  }
  const response = await axios.post(`${API_BASE_URL}/reports/schedule`, schedule);
  return response.data;
};

export const getScheduledReports = async (): Promise<ReportSchedule[]> => {
  if (USE_MOCK_API) {
    return mockGetScheduledReports();
  }
  const response = await axios.get(`${API_BASE_URL}/reports/scheduled`);
  return response.data;
};

export const getScheduledReport = async (scheduleId: number): Promise<ReportSchedule> => {
  if (USE_MOCK_API) {
    return mockGetScheduledReport(scheduleId);
  }
  const response = await axios.get(`${API_BASE_URL}/reports/scheduled/${scheduleId}`);
  return response.data;
};

export const updateSchedule = async (scheduleId: number, schedule: ReportSchedule): Promise<ReportSchedule> => {
  if (USE_MOCK_API) {
    return mockUpdateSchedule(scheduleId, schedule);
  }
  const response = await axios.put(`${API_BASE_URL}/reports/scheduled/${scheduleId}`, schedule);
  return response.data;
};

export const deleteSchedule = async (scheduleId: number): Promise<void> => {
  if (USE_MOCK_API) {
    return mockDeleteSchedule(scheduleId);
  }
  await axios.delete(`${API_BASE_URL}/reports/scheduled/${scheduleId}`);
};

// Report Export API
export const exportReport = async (reportId: number, format: string): Promise<ExportStatus> => {
  if (USE_MOCK_API) {
    return mockExportReport(reportId, format);
  }
  const response = await axios.post(`${API_BASE_URL}/reports/export`, { report_id: reportId, format });
  return response.data;
};

export const getExportStatus = async (taskId: string): Promise<ExportStatus> => {
  if (USE_MOCK_API) {
    return mockGetExportStatus(taskId);
  }
  const response = await axios.get(`${API_BASE_URL}/reports/export/${taskId}`);
  return response.data;
};
