import React from 'react';
import { 
  Box, CircularProgress, Typography, Backdrop, 
  Fade, Skeleton, LinearProgress 
} from '@mui/material';

/**
 * Full page loading backdrop
 */
export const FullPageLoader = ({ open, message = 'Loading...' }) => {
  return (
    <Backdrop
      sx={{
        color: '#fff',
        zIndex: (theme) => theme.zIndex.drawer + 1,
        flexDirection: 'column',
        gap: 2
      }}
      open={open}
    >
      <CircularProgress color="inherit" />
      {message && <Typography variant="h6">{message}</Typography>}
    </Backdrop>
  );
};

/**
 * Content area loading indicator
 */
export const ContentLoader = ({ height = 400, message = 'Loading content...' }) => {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: height,
        width: '100%',
      }}
    >
      <CircularProgress size={40} sx={{ mb: 2 }} />
      {message && <Typography variant="body1" color="text.secondary">{message}</Typography>}
    </Box>
  );
};

/**
 * Button loading indicator
 */
export const ButtonLoader = ({ size = 20 }) => {
  return <CircularProgress size={size} color="inherit" />;
};

/**
 * Card loading skeleton
 */
export const CardSkeleton = ({ count = 1 }) => {
  return (
    <>
      {[...Array(count)].map((_, index) => (
        <Box key={index} sx={{ mb: 2 }}>
          <Skeleton variant="rectangular" width="100%" height={40} sx={{ mb: 1, borderRadius: 1 }} />
          <Skeleton variant="rectangular" width="60%" height={20} sx={{ mb: 1, borderRadius: 1 }} />
          <Skeleton variant="rectangular" width="100%" height={100} sx={{ borderRadius: 1 }} />
        </Box>
      ))}
    </>
  );
};

/**
 * Table loading skeleton
 */
export const TableSkeleton = ({ rows = 5, columns = 4 }) => {
  return (
    <Box sx={{ width: '100%' }}>
      <Skeleton variant="rectangular" width="100%" height={50} sx={{ mb: 1, borderRadius: 1 }} />
      {[...Array(rows)].map((_, rowIndex) => (
        <Box key={rowIndex} sx={{ display: 'flex', mb: 1 }}>
          {[...Array(columns)].map((_, colIndex) => (
            <Skeleton
              key={colIndex}
              variant="rectangular"
              width={`${100 / columns}%`}
              height={40}
              sx={{ mx: 0.5, borderRadius: 1 }}
            />
          ))}
        </Box>
      ))}
    </Box>
  );
};

/**
 * Linear progress indicator for the top of the page
 */
export const TopProgressBar = ({ loading }) => {
  return (
    <Fade in={loading} unmountOnExit>
      <LinearProgress
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          zIndex: (theme) => theme.zIndex.appBar + 1,
        }}
      />
    </Fade>
  );
};

/**
 * Form field loading skeleton
 */
export const FormFieldSkeleton = ({ count = 1 }) => {
  return (
    <>
      {[...Array(count)].map((_, index) => (
        <Box key={index} sx={{ mb: 2 }}>
          <Skeleton variant="rectangular" width="30%" height={20} sx={{ mb: 1, borderRadius: 1 }} />
          <Skeleton variant="rectangular" width="100%" height={56} sx={{ borderRadius: 1 }} />
        </Box>
      ))}
    </>
  );
};

export default {
  FullPageLoader,
  ContentLoader,
  ButtonLoader,
  CardSkeleton,
  TableSkeleton,
  TopProgressBar,
  FormFieldSkeleton
};
