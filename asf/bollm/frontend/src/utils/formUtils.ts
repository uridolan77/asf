// Form utility types and functions
import { FieldValues, UseFormReturn, FieldErrors } from 'react-hook-form';
import * as yup from 'yup';

// Utility function to get error message from form errors
export const getErrorMessage = <T extends FieldValues>(
  errors: FieldErrors<T>,
  field: keyof T
): string => {
  const error = errors[field];
  return error ? (error.message as string) : '';
};

// Common validation schemas
export const validationSchemas = {
  // Common validation patterns
  email: yup.string()
    .email('Please enter a valid email')
    .required('Email is required'),
    
  password: yup.string()
    .min(8, 'Password must be at least 8 characters')
    .matches(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
      'Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character'
    )
    .required('Password is required'),
    
  username: yup.string()
    .min(3, 'Username must be at least 3 characters')
    .max(20, 'Username cannot exceed 20 characters')
    .matches(
      /^[a-zA-Z0-9_-]+$/,
      'Username can only contain letters, numbers, underscores, and hyphens'
    )
    .required('Username is required'),
    
  name: yup.string()
    .min(2, 'Name must be at least 2 characters')
    .required('Name is required'),
    
  // Add more common validations as needed
};

// Custom form types
export interface FormState<T extends FieldValues> {
  form: UseFormReturn<T>;
  isSubmitting: boolean;
  isSubmitted: boolean;
  isValid: boolean;
  errors: FieldErrors<T>;
  reset: () => void;
  getErrorMessage: (field: keyof T) => string;
}

export type ValidationSchema<T extends FieldValues> = yup.ObjectSchema<{
  [K in keyof T]: any;
}>;