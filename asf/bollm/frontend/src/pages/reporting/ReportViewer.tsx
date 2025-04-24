import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Button,
  Paper,
  Typography,
  Tabs,
  Tab,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Edit as EditIcon,
  Refresh as RefreshIcon,
  GetApp as ExportIcon,
  Schedule as ScheduleIcon
} from '@mui/icons-material';

import ReportTable from '../../components/reporting/ReportTable';
import ReportChart from '../../components/reporting/ReportChart';
import ExportReportDialog from '../../components/reporting/ExportReportDialog';
import ScheduleReportDialog from '../../components/reporting/ScheduleReportDialog';

import { useReport } from '../../context/ReportContext';
import { useDimensions } from '../../hooks/useDimensions';
import { useMetrics } from '../../hooks/useMetrics';
import { executeReport } from '../../api/reporting';
import { ReportResult } from '../../types/reporting';

enum ResultView {
  TABLE = 'table',
  CHART = 'chart',
}

const ReportViewer: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const {
    reportDefinition,
    savedReport,
    loadSavedReport
  } = useReport();
  
  const [reportResult, setReportResult] = useState<ReportResult | null>(null);
  const [resultView, setResultView] = useState<ResultView>(ResultView.TABLE);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  const [exportDialogOpen, setExportDialogOpen] = useState<boolean>(false);
  const [scheduleDialogOpen, setScheduleDialogOpen] = useState<boolean>(false);
  
  const { dimensions } = useDimensions();
  const { metrics } = useMetrics();
  
  // Check if we should open the schedule dialog from URL params
  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    if (searchParams.get('schedule') === 'true') {
      setScheduleDialogOpen(true);
    }
  }, [location]);
  
  // Load saved report and execute it
  useEffect(() => {
    const fetchReportAndExecute = async () => {
      if (!id) return;
      
      try {
        setLoading(true);
        const report = await loadSavedReport(parseInt(id));
        
        if (report) {
          const result = await executeReport(report.report_definition);
          setReportResult(result);
        }
      } catch (err) {
        console.error('Error loading or executing report:', err);
        setError('Failed to load or execute the report');
      } finally {
        setLoading(false);
      }
    };
    
    fetchReportAndExecute();
  }, [id]);
  
  const handleResultViewChange = (_event: React.SyntheticEvent, newValue: ResultView) => {
    setResultView(newValue);
  };
  
  const handleRefreshReport = async () => {
    if (!reportDefinition) return;
    
    try {
      setLoading(true);
      setError(null);
      
      const result = await executeReport(reportDefinition);
      setReportResult(result);
    } catch (err) {
      console.error('Error refreshing report:', err);
      setError('Failed to refresh the report');
    } finally {
      setLoading(false);
    }
  };
  
  const handleEditReport = () => {
    if (!id) return;
    navigate(`/reports/edit/${id}`);
  };
  
  const handleExportDialogOpen = () => {
    setExportDialogOpen(true);
  };
  
  const handleExportDialogClose = () => {
    setExportDialogOpen(false);
  };
  
  const handleScheduleDialogOpen = () => {
    setScheduleDialogOpen(true);
  };
  
  const handleScheduleDialogClose = () => {
    setScheduleDialogOpen(false);
    
    // Remove the schedule param from URL if it exists
    if (location.search.includes('schedule=true')) {
      navigate(`/reports/view/${id}`, { replace: true });
    }
  };
  
  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">
          {savedReport?.name || 'Report Viewer'}
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={handleRefreshReport}
            disabled={loading}
          >
            Refresh
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<EditIcon />}
            onClick={handleEditReport}
            disabled={loading}
          >
            Edit
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<ExportIcon />}
            onClick={handleExportDialogOpen}
            disabled={loading}
          >
            Export
          </Button>
          
          <Button
            variant="outlined"
            startIcon={<ScheduleIcon />}
            onClick={handleScheduleDialogOpen}
            disabled={loading}
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
      
      {savedReport?.description && (
        <Typography variant="body1" sx={{ mb: 3 }}>
          {savedReport.description}
        </Typography>
      )}
      
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
      
      {/* Dialogs */}
      <ExportReportDialog
        open={exportDialogOpen}
        onClose={handleExportDialogClose}
        reportId={savedReport?.id}
      />
      
      <ScheduleReportDialog
        open={scheduleDialogOpen}
        onClose={handleScheduleDialogClose}
        reportId={savedReport?.id}
      />
    </Box>
  );
};

export default ReportViewer;
