// Mock API implementation for the reporting system
import { 
  dimensions, 
  metrics, 
  dimensionCategories, 
  metricCategories, 
  savedReports, 
  scheduledReports, 
  sampleReportResult 
} from './data';

// Helper function to simulate API delay
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Dimensions API
export const mockGetDimensions = async (category) => {
  await delay(300);
  if (category) {
    return dimensions.filter(d => d.category === category);
  }
  return dimensions;
};

export const mockGetDimensionCategories = async () => {
  await delay(200);
  return dimensionCategories;
};

export const mockGetDimensionValues = async (dimensionId, search, limit) => {
  await delay(300);
  
  // Mock dimension values based on dimension ID
  const valuesByDimension = {
    patient_gender: ['Male', 'Female', 'Other', 'Unknown'].map(v => ({ value: v, count: Math.floor(Math.random() * 1000) })),
    patient_age: ['0-17', '18-34', '35-50', '51-65', '65+'].map(v => ({ value: v, count: Math.floor(Math.random() * 1000) })),
    provider_specialty: ['Cardiology', 'Neurology', 'Oncology', 'Pediatrics', 'Primary Care', 'Surgery'].map(v => ({ value: v, count: Math.floor(Math.random() * 100) })),
    diagnosis_code: ['A00', 'B01', 'C02', 'D03', 'E04', 'F05'].map(v => ({ value: v, count: Math.floor(Math.random() * 500) })),
    medication_name: ['Aspirin', 'Lisinopril', 'Metformin', 'Atorvastatin', 'Levothyroxine'].map(v => ({ value: v, count: Math.floor(Math.random() * 800) }))
  };
  
  let values = valuesByDimension[dimensionId] || [];
  
  // Apply search filter if provided
  if (search) {
    const searchLower = search.toLowerCase();
    values = values.filter(v => v.value.toLowerCase().includes(searchLower));
  }
  
  // Apply limit
  if (limit && values.length > limit) {
    values = values.slice(0, limit);
  }
  
  return values;
};

// Metrics API
export const mockGetMetrics = async (category) => {
  await delay(300);
  if (category) {
    return metrics.filter(m => m.category === category);
  }
  return metrics;
};

export const mockGetMetricCategories = async () => {
  await delay(200);
  return metricCategories;
};

// Reports API
export const mockExecuteReport = async (reportDefinition) => {
  await delay(800);
  
  // For simplicity, we'll return the sample report result
  // In a real implementation, this would process the report definition
  // and generate a result based on the actual data
  return {
    ...sampleReportResult,
    report_definition: reportDefinition
  };
};

export const mockSaveReport = async (report) => {
  await delay(500);
  
  if (report.id) {
    // Update existing report
    const index = savedReports.findIndex(r => r.id === report.id);
    if (index !== -1) {
      const updatedReport = {
        ...savedReports[index],
        ...report,
        updated_at: new Date().toISOString()
      };
      savedReports[index] = updatedReport;
      return updatedReport;
    }
  }
  
  // Create new report
  const newReport = {
    ...report,
    id: Math.max(...savedReports.map(r => r.id), 0) + 1,
    owner: 'admin',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };
  savedReports.push(newReport);
  return newReport;
};

export const mockGetSavedReports = async () => {
  await delay(400);
  return savedReports;
};

export const mockGetSavedReport = async (reportId) => {
  await delay(300);
  const report = savedReports.find(r => r.id === reportId);
  if (!report) {
    throw new Error(`Report with ID ${reportId} not found`);
  }
  return report;
};

export const mockUpdateReport = async (reportId, report) => {
  await delay(500);
  const index = savedReports.findIndex(r => r.id === reportId);
  if (index === -1) {
    throw new Error(`Report with ID ${reportId} not found`);
  }
  
  const updatedReport = {
    ...savedReports[index],
    ...report,
    id: reportId,
    updated_at: new Date().toISOString()
  };
  savedReports[index] = updatedReport;
  return updatedReport;
};

export const mockDeleteReport = async (reportId) => {
  await delay(400);
  const index = savedReports.findIndex(r => r.id === reportId);
  if (index === -1) {
    throw new Error(`Report with ID ${reportId} not found`);
  }
  
  savedReports.splice(index, 1);
};

// Report Scheduling API
export const mockScheduleReport = async (schedule) => {
  await delay(500);
  
  if (schedule.id) {
    // Update existing schedule
    const index = scheduledReports.findIndex(s => s.id === schedule.id);
    if (index !== -1) {
      const updatedSchedule = {
        ...scheduledReports[index],
        ...schedule,
        updated_at: new Date().toISOString()
      };
      scheduledReports[index] = updatedSchedule;
      return updatedSchedule;
    }
  }
  
  // Create new schedule
  const newSchedule = {
    ...schedule,
    id: Math.max(...scheduledReports.map(s => s.id), 0) + 1,
    owner: 'admin',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    last_run: null,
    next_run: calculateNextRun(schedule)
  };
  scheduledReports.push(newSchedule);
  return newSchedule;
};

export const mockGetScheduledReports = async () => {
  await delay(400);
  return scheduledReports;
};

export const mockGetScheduledReport = async (scheduleId) => {
  await delay(300);
  const schedule = scheduledReports.find(s => s.id === scheduleId);
  if (!schedule) {
    throw new Error(`Schedule with ID ${scheduleId} not found`);
  }
  return schedule;
};

export const mockUpdateSchedule = async (scheduleId, schedule) => {
  await delay(500);
  const index = scheduledReports.findIndex(s => s.id === scheduleId);
  if (index === -1) {
    throw new Error(`Schedule with ID ${scheduleId} not found`);
  }
  
  const updatedSchedule = {
    ...scheduledReports[index],
    ...schedule,
    id: scheduleId,
    updated_at: new Date().toISOString(),
    next_run: calculateNextRun(schedule)
  };
  scheduledReports[index] = updatedSchedule;
  return updatedSchedule;
};

export const mockDeleteSchedule = async (scheduleId) => {
  await delay(400);
  const index = scheduledReports.findIndex(s => s.id === scheduleId);
  if (index === -1) {
    throw new Error(`Schedule with ID ${scheduleId} not found`);
  }
  
  scheduledReports.splice(index, 1);
};

// Report Export API
export const mockExportReport = async (reportId, format) => {
  await delay(600);
  
  // Simulate starting an export task
  const taskId = `export-${reportId}-${Date.now()}`;
  return {
    task_id: taskId,
    status: 'pending',
    username: 'admin',
    format,
    created_at: new Date().toISOString()
  };
};

export const mockGetExportStatus = async (taskId) => {
  await delay(300);
  
  // Simulate a completed export after a delay
  return {
    task_id: taskId,
    status: 'completed',
    username: 'admin',
    format: 'pdf',
    created_at: new Date(Date.now() - 60000).toISOString(),
    completed_at: new Date().toISOString(),
    download_url: `/api/reports/export/${taskId}/download`
  };
};

// Helper function to calculate the next run time for a schedule
function calculateNextRun(schedule) {
  const now = new Date();
  let nextRun = new Date();
  
  // Set the time
  const [hours, minutes] = schedule.time.split(':').map(Number);
  nextRun.setHours(hours, minutes, 0, 0);
  
  // If the time has already passed today, move to the next occurrence
  if (nextRun <= now) {
    nextRun.setDate(nextRun.getDate() + 1);
  }
  
  // Adjust based on frequency
  if (schedule.frequency === 'daily') {
    // Already set for the next day
  } else if (schedule.frequency === 'weekly' && schedule.days_of_week) {
    // Find the next day of the week that matches
    const currentDay = nextRun.getDay(); // 0 = Sunday, 1 = Monday, etc.
    const daysOfWeek = schedule.days_of_week.map(d => d === 0 ? 7 : d); // Convert 0 (Sunday) to 7 for easier calculation
    
    // Find the next day of the week that's in the schedule
    let daysToAdd = 0;
    for (let i = 0; i < 7; i++) {
      const checkDay = (currentDay + i) % 7 || 7; // Convert 0 to 7 (Sunday)
      if (daysOfWeek.includes(checkDay)) {
        daysToAdd = i;
        break;
      }
    }
    
    nextRun.setDate(nextRun.getDate() + daysToAdd);
  } else if (schedule.frequency === 'monthly' && schedule.days_of_month) {
    // Find the next day of the month that matches
    const currentDate = nextRun.getDate();
    const daysOfMonth = schedule.days_of_month;
    
    // If there's a day in the current month that's still upcoming
    const validDaysThisMonth = daysOfMonth.filter(d => d > currentDate && d <= new Date(nextRun.getFullYear(), nextRun.getMonth() + 1, 0).getDate());
    
    if (validDaysThisMonth.length > 0) {
      nextRun.setDate(Math.min(...validDaysThisMonth));
    } else {
      // Move to the next month and use the first valid day
      nextRun.setMonth(nextRun.getMonth() + 1);
      nextRun.setDate(Math.min(...daysOfMonth.filter(d => d <= new Date(nextRun.getFullYear(), nextRun.getMonth() + 1, 0).getDate())));
    }
  }
  
  return nextRun.toISOString();
}
