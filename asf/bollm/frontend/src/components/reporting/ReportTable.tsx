import React, { useState } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Typography,
  Paper,
  TablePagination
} from '@mui/material';
import { ReportResult, Dimension, Metric } from '../../types/reporting';

interface ReportTableProps {
  result: ReportResult | null;
  loading: boolean;
  dimensions: Dimension[];
  metrics: Metric[];
  onSortChange?: (field: string, direction: 'asc' | 'desc') => void;
}

const ReportTable: React.FC<ReportTableProps> = ({
  result,
  loading,
  dimensions,
  metrics,
  onSortChange
}) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [orderBy, setOrderBy] = useState<string | undefined>(undefined);
  const [order, setOrder] = useState<'asc' | 'desc'>('asc');
  
  if (!result || !result.data || loading) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No data to display
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Run the report to see results
        </Typography>
      </Box>
    );
  }
  
  const { data } = result;
  const { dimensions: dimensionIds, metrics: metricIds, rows, totals } = data;
  
  const getDimensionName = (id: string): string => {
    const dimension = dimensions.find(d => d.id === id);
    return dimension ? dimension.name : id;
  };
  
  const getMetricName = (id: string): string => {
    const metric = metrics.find(m => m.id === id);
    return metric ? metric.name : id;
  };
  
  const getMetricFormat = (id: string): string | undefined => {
    const metric = metrics.find(m => m.id === id);
    return metric ? metric.format : undefined;
  };
  
  const formatValue = (value: any, format?: string): string => {
    if (value === null || value === undefined) return '-';
    
    if (typeof value === 'number') {
      if (format === 'currency') {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);
      } else if (format === 'percent') {
        return new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(value / 100);
      } else {
        return new Intl.NumberFormat('en-US').format(value);
      }
    }
    
    return String(value);
  };
  
  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };
  
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };
  
  const handleRequestSort = (field: string) => {
    const isAsc = orderBy === field && order === 'asc';
    const newOrder = isAsc ? 'desc' : 'asc';
    setOrder(newOrder);
    setOrderBy(field);
    
    if (onSortChange) {
      onSortChange(field, newOrder);
    }
  };
  
  // Apply pagination
  const paginatedRows = rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);
  
  return (
    <Box>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              {dimensionIds.map(id => (
                <TableCell key={id}>
                  {onSortChange ? (
                    <TableSortLabel
                      active={orderBy === id}
                      direction={orderBy === id ? order : 'asc'}
                      onClick={() => handleRequestSort(id)}
                    >
                      {getDimensionName(id)}
                    </TableSortLabel>
                  ) : (
                    getDimensionName(id)
                  )}
                </TableCell>
              ))}
              
              {metricIds.map(id => (
                <TableCell key={id} align="right">
                  {onSortChange ? (
                    <TableSortLabel
                      active={orderBy === id}
                      direction={orderBy === id ? order : 'asc'}
                      onClick={() => handleRequestSort(id)}
                    >
                      {getMetricName(id)}
                    </TableSortLabel>
                  ) : (
                    getMetricName(id)
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          
          <TableBody>
            {paginatedRows.map((row, index) => (
              <TableRow key={index}>
                {dimensionIds.map(id => (
                  <TableCell key={id}>
                    {formatValue(row[id])}
                  </TableCell>
                ))}
                
                {metricIds.map(id => (
                  <TableCell key={id} align="right">
                    {formatValue(row[id], getMetricFormat(id))}
                  </TableCell>
                ))}
              </TableRow>
            ))}
            
            {/* Totals row */}
            {Object.keys(totals).length > 0 && (
              <TableRow sx={{ '& td': { fontWeight: 'bold', borderTop: '2px solid rgba(224, 224, 224, 1)' } }}>
                {dimensionIds.map((id, index) => (
                  <TableCell key={id}>
                    {index === 0 ? 'Total' : ''}
                  </TableCell>
                ))}
                
                {metricIds.map(id => (
                  <TableCell key={id} align="right">
                    {formatValue(totals[id], getMetricFormat(id))}
                  </TableCell>
                ))}
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      
      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
        component="div"
        count={rows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Box>
  );
};

export default ReportTable;
