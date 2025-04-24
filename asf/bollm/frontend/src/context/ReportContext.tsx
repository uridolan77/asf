import React, { createContext, useContext, useState, ReactNode } from 'react';
import { ReportDefinition, SavedReport } from '../types/reporting';
import { getSavedReport } from '../api/reporting';

interface ReportContextType {
  reportDefinition: ReportDefinition;
  setReportDefinition: React.Dispatch<React.SetStateAction<ReportDefinition>>;
  savedReport: SavedReport | null;
  setSavedReport: React.Dispatch<React.SetStateAction<SavedReport | null>>;
  loadSavedReport: (reportId: number) => Promise<SavedReport | null>;
}

const defaultReportDefinition: ReportDefinition = {
  dimensions: [],
  metrics: [],
  filters: {},
  sort_by: undefined,
  sort_order: 'asc',
  limit: 100,
};

const ReportContext = createContext<ReportContextType | undefined>(undefined);

export const ReportProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [reportDefinition, setReportDefinition] = useState<ReportDefinition>(defaultReportDefinition);
  const [savedReport, setSavedReport] = useState<SavedReport | null>(null);

  const loadSavedReport = async (reportId: number): Promise<SavedReport | null> => {
    try {
      const report = await getSavedReport(reportId);
      setSavedReport(report);
      
      // If the report definition is a string, parse it
      if (typeof report.report_definition === 'string') {
        report.report_definition = JSON.parse(report.report_definition as unknown as string);
      }
      
      setReportDefinition(report.report_definition as ReportDefinition);
      return report;
    } catch (error) {
      console.error('Error loading saved report:', error);
      return null;
    }
  };

  return (
    <ReportContext.Provider
      value={{
        reportDefinition,
        setReportDefinition,
        savedReport,
        setSavedReport,
        loadSavedReport,
      }}
    >
      {children}
    </ReportContext.Provider>
  );
};

export const useReport = (): ReportContextType => {
  const context = useContext(ReportContext);
  if (context === undefined) {
    throw new Error('useReport must be used within a ReportProvider');
  }
  return context;
};

export default ReportContext;
