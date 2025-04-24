import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Grid,
  Paper,
  Typography,
  Divider,
  Tabs,
  Tab,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Save as SaveIcon,
  PlayArrow as RunIcon,
  GetApp as ExportIcon,
  Schedule as ScheduleIcon
} from '@mui/icons-material';

import DimensionSelector from '../../components/reporting/DimensionSelector';
import MetricSelector from '../../components/reporting/MetricSelector';
import FilterBuilder from '../../components/reporting/FilterBuilder';
import ReportTable from '../../components/reporting/ReportTable';
import ReportChart from '../../components/reporting/ReportChart';
import SaveReportDialog from '../../components/reporting/SaveReportDialog';
import ExportReportDialog from '../../components/reporting/ExportReportDialog';
import ScheduleReportDialog from '../../components/reporting/ScheduleReportDialog';

import { useReport } from '../../context/ReportContext';
import { useDimensions } from '../../hooks/useDimensions';
import { useMetrics } from '../../hooks/useMetrics';
import { executeReport } from '../../api/reporting';
import { ReportResult, SavedReport, ReportSchedule } from '../../types/reporting';

enum ResultView {
  TABLE = 'table',
  CHART = 'chart',
}

const ReportBuilder: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const {
    reportDefinition,
    setReportDefinition,
    savedReport,
    setSavedReport,
    loadSavedReport
  } = useReport();
  
  const [selectedDimensions, setSelectedDimensions] = useState<string[]>(reportDefinition.dimensions || []);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(reportDefinition.metrics || []);
  const [filters, setFilters] = useState<Record<string, any>>(reportDefinition.filters || {});
  const [sortBy, setSortBy] = useState<string | undefined>(reportDefinition.sort_by);
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>(reportDefinition.sort_order || 'asc');
  const [limit, setLimit] = useState<number | undefined>(reportDefinition.limit);
  
  const [reportResult, setReportResult] = useState<ReportResult | null>(null);
  const [resultView, setResultView] = useState<ResultView>(ResultView.TABLE);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  const [saveDialogOpen, setSaveDialogOpen] = useState<boolean>(false);
  const [exportDialogOpen, setExportDialogOpen] = useState<boolean>(false);
  const [scheduleDialogOpen, setScheduleDialogOpen] = useState<boolean>(false);
  
  const { dimensions } = useDimensions();
  const { metrics } = useMetrics();
  
  // Load saved report if ID is provided
  useEffect(() => {
    const fetchSavedReport = async () => {
      if (id) {
        try {
          setLoading(true);
          const report = await loadSavedReport(parseInt(id));
          if (report) {
            setSelectedDimensions(report.report_definition.dimensions || []);
            setSelectedMetrics(report.report_definition.metrics || []);
            setFilters(report.report_definition.filters || {});
            setSortBy(report.report_definition.sort_by);
            setSortOrder(report.report_definition.sort_order || 'asc');
            setLimit(report.report_definition.limit);
          }
        } catch (err) {
          console.error('Error loading saved report:', err);
          setError('Failed to load the saved report');
        } finally {
          setLoading(false);
        }
      }
    };
    
    fetchSavedReport();
  }, [id]);
  
  // Update report definition when selections change
  useEffect(() => {
    setReportDefinition({
      dimensions: selectedDimensions,
      metrics: selectedMetrics,
      filters,
      sort_by: sortBy,
      sort_order: sortOrder,
      limit
    });
  }, [selectedDimensions, selectedMetrics, filters, sortBy, sortOrder, limit]);
  
  const handleDimensionToggle = (dimensionId: string) => {
    setSelectedDimensions(prev => {
      if (prev.includes(dimensionId)) {
        return prev.filter(id => id !== dimensionId);
      } else {
        return [...prev, dimensionId];
      }
    });
  };
  
  const handleMetricToggle = (metricId: string) => {
    setSelectedMetrics(prev => {
      if (prev.includes(metricId)) {
        return prev.filter(id => id !== metricId);
      } else {
        return [...prev, metricId];
      }
    });
  };
  
  const handleFiltersChange = (newFilters: Record<string, any>) => {
    setFilters(newFilters);
  };
  
  const handleSortChange = (field: string, direction: 'asc' | 'desc') => {
    setSortBy(field);
    setSortOrder(direction);
  };
  
  const handleResultViewChange = (_event: React.SyntheticEvent, newValue: ResultView) => {
    setResultView(newValue);
  };
  
  const handleRunReport = async () => {
    if (selectedDimensions.length === 0) {
      setError('Please select at least one dimension');
      return;
    }
    
    if (selectedMetrics.length === 0) {
      setError('Please select at least one metric');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const result = await executeReport(reportDefinition);
      setReportResult(result);
    } catch (err) {
      console.error('Error executing report:', err);
      setError('Failed to execute report. Please check your selections and try again.');
    } finally {
      setLoading(false);
    }
  };
  
  const handleSaveDialogOpen = () => {
    setSaveDialogOpen(true);
  };
  
  const handleSaveDialogClose = () => {
    setSaveDialogOpen(false);
  };
  
  const handleExportDialogOpen = () => {
    if (!savedReport?.id) {
      setError('Please save the report before exporting');
      return;
    }
    
    setExportDialogOpen(true);
  };
  
  const handleExportDialogClose = () => {
    setExportDialogOpen(false);
  };
  
  const handleScheduleDialogOpen = () => {
    if (!savedReport?.id) {
      setError('Please save the report before scheduling');
      return;
    }
    
    setScheduleDialogOpen(true);
  };
  
  const handleScheduleDialogClose = () => {
    setScheduleDialogOpen(false);
  };
  
  const handleSaveSuccess = (report: SavedReport) => {
    setSavedReport(report);
    
    // If this is a new report, redirect to the edit URL
    if (!id && report.id) {
      navigate(`/reports/edit/${report.id}`, { replace: true });
    }
  };
  
  const handleScheduleSuccess = (schedule: ReportSchedule) => {
    // Optionally show a success message or redirect to scheduled reports
  };
  
  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          {id ? `Edit Report: ${savedReport?.name || ''}` : 'Create New Report'}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="contained"
            color="primary"
            startIcon={<RunIcon />}
            onClick={handleRunReport}
            disabled={loading || selectedDimensions.length === 0 || selectedMetrics.length === 0}
          >
            Run Report
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<SaveIcon />}
            onClick={handleSaveDialogOpen}
            disabled={loading}
          >
            Save
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<ExportIcon />}
            onClick={handleExportDialogOpen}
            disabled={loading || !savedReport?.id}
          >
            Export
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<ScheduleIcon />}
            onClick={handleScheduleDialogOpen}
            disabled={loading || !savedReport?.id}
          >
            Schedule
          </Button>
        </Box>
      </Box>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, mb: 3 }}>
            <DimensionSelector
              selectedDimensions={selectedDimensions}
              onDimensionToggle={handleDimensionToggle}
            />
          </Paper>
          
          <Paper sx={{ p: 2 }}>
            <MetricSelector
              selectedMetrics={selectedMetrics}
              onMetricToggle={handleMetricToggle}
            />
          </Paper>
        </Grid>
        
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2, mb: 3 }}>
            <FilterBuilder
              filters={filters}
              onFiltersChange={handleFiltersChange}
            />
          </Paper>
          
          <Paper sx={{ p: 2 }}>
            <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
              <Tabs
                value={resultView}
                onChange={handleResultViewChange}
                aria-label="report results view"
              >
                <Tab label="Table" value={ResultView.TABLE} />
                <Tab label="Chart" value={ResultView.CHART} />
              </Tabs>
            </Box>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                <CircularProgress />
              </Box>
            ) : (
              <Box>
                {resultView === ResultView.TABLE ? (
                  <ReportTable
                    result={reportResult}
                    loading={loading}
                    dimensions={dimensions}
                    metrics={metrics}
                    onSortChange={handleSortChange}
                  />
                ) : (
                  <ReportChart
                    result={reportResult}
                    loading={loading}
                    dimensions={dimensions}
                    metrics={metrics}
                  />
                )}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
      
      {/* Dialogs */}
      <SaveReportDialog
        open={saveDialogOpen}
        onClose={handleSaveDialogClose}
        reportDefinition={reportDefinition}
        savedReport={savedReport}
        onSaveSuccess={handleSaveSuccess}
      />
      
      <ExportReportDialog
        open={exportDialogOpen}
        onClose={handleExportDialogClose}
        reportId={savedReport?.id}
      />
      
      <ScheduleReportDialog
        open={scheduleDialogOpen}
        onClose={handleScheduleDialogClose}
        reportId={savedReport?.id}
        onScheduleSuccess={handleScheduleSuccess}
      />
    </Box>
  );
};

export default ReportBuilder;
