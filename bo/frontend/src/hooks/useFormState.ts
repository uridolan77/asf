// Custom hook to unify form handling
import { useState } from 'react';
import { useForm, UseFormProps, FieldValues } from 'react-hook-form';
import { yupResolver } from '@hookform/resolvers/yup';
import { ValidationSchema, FormState, getErrorMessage } from '../utils/formUtils';

interface UseFormStateOptions<T extends FieldValues> extends UseFormProps<T> {
  validationSchema?: ValidationSchema<T>;
}

/**
 * Custom hook that combines react-hook-form with Yup validation
 * 
 * @param options - Form options including validation schema
 * @returns FormState object with form methods and state
 */
export function useFormState<T extends FieldValues>(
  options: UseFormStateOptions<T> = {}
): FormState<T> {
  const { validationSchema, ...formOptions } = options;
  
  // Configure resolver if validation schema is provided
  const formConfig = validationSchema
    ? {
        ...formOptions,
        resolver: yupResolver(validationSchema),
      }
    : formOptions;
  
  const form = useForm<T>(formConfig);
  const { formState, reset } = form;
  const { isSubmitting, isSubmitted, isValid, errors } = formState;
  
  // Create a function to get error message for a specific field
  const getFieldErrorMessage = (field: keyof T) => getErrorMessage(errors, field);
  
  return {
    form,
    isSubmitting,
    isSubmitted,
    isValid,
    errors,
    reset,
    getErrorMessage: getFieldErrorMessage,
  };
}