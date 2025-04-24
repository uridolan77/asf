// frontend/src/components/UI/SkeletonLoaders.js
import React from 'react';
import { Skeleton, Box, Card, CardContent, Grid, Typography } from '@mui/material';

/**
 * Card Skeleton - for loading card-based components
 */
export const CardSkeleton = ({ count = 1, height = 200 }) => {
  return (
    <Grid container spacing={2}>
      {Array.from(new Array(count)).map((_, index) => (
        <Grid item xs={12} md={4} key={index}>
          <Card>
            <CardContent>
              <Skeleton variant="rectangular" height={height} />
              <Box sx={{ pt: 0.5 }}>
                <Skeleton />
                <Skeleton width="60%" />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
};

/**
 * Table Skeleton - for loading table-based components
 */
export const TableSkeleton = ({ rowCount = 5, columnCount = 4 }) => {
  return (
    <Box sx={{ width: '100%' }}>
      {/* Header row */}
      <Box sx={{ display: 'flex', mb: 1 }}>
        {Array.from(new Array(columnCount)).map((_, index) => (
          <Box key={index} sx={{ flex: 1, px: 1 }}>
            <Skeleton variant="rectangular" height={30} />
          </Box>
        ))}
      </Box>
      
      {/* Data rows */}
      {Array.from(new Array(rowCount)).map((_, rowIndex) => (
        <Box key={rowIndex} sx={{ display: 'flex', my: 2 }}>
          {Array.from(new Array(columnCount)).map((_, colIndex) => (
            <Box key={colIndex} sx={{ flex: 1, px: 1 }}>
              <Skeleton variant="text" />
            </Box>
          ))}
        </Box>
      ))}
    </Box>
  );
};

/**
 * Dashboard Skeleton - for loading dashboard components
 */
export const DashboardSkeleton = () => {
  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Skeleton variant="rectangular" height={40} width="50%" />
        <Skeleton variant="text" width="70%" sx={{ mt: 1 }} />
      </Box>
      
      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {Array.from(new Array(4)).map((_, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Skeleton variant="text" width="60%" />
                <Skeleton variant="rectangular" height={60} sx={{ mt: 2 }} />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
      
      {/* Main content area */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Skeleton variant="rectangular" height={20} width="40%" sx={{ mb: 2 }} />
              <Skeleton variant="rectangular" height={250} />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Skeleton variant="rectangular" height={20} width="60%" sx={{ mb: 2 }} />
              <Skeleton variant="rectangular" height={250} />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

/**
 * Detail View Skeleton - for loading detail views
 */
export const DetailViewSkeleton = () => {
  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Skeleton variant="rectangular" height={40} width="70%" sx={{ mb: 2 }} />
      
      {/* Meta information */}
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Skeleton variant="text" width={100} />
        <Skeleton variant="text" width={100} />
        <Skeleton variant="text" width={100} />
      </Box>
      
      {/* Content sections */}
      <Box sx={{ mb: 3 }}>
        <Skeleton variant="rectangular" height={30} width="30%" sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={100} />
      </Box>
      
      <Box sx={{ mb: 3 }}>
        <Skeleton variant="rectangular" height={30} width="25%" sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={150} />
      </Box>
      
      <Box>
        <Skeleton variant="rectangular" height={30} width="35%" sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={120} />
      </Box>
    </Box>
  );
};

/**
 * Form Skeleton - for loading forms
 */
export const FormSkeleton = ({ fieldCount = 4 }) => {
  return (
    <Box sx={{ width: '100%' }}>
      <Skeleton variant="rectangular" height={30} width="40%" sx={{ mb: 4 }} />
      
      {Array.from(new Array(fieldCount)).map((_, index) => (
        <Box key={index} sx={{ mb: 3 }}>
          <Skeleton variant="text" width="30%" height={20} sx={{ mb: 1 }} />
          <Skeleton variant="rectangular" height={40} />
        </Box>
      ))}
      
      <Skeleton variant="rectangular" height={40} width="30%" sx={{ mt: 2 }} />
    </Box>
  );
};