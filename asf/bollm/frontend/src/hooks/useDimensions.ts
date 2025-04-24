import { useState, useEffect } from 'react';
import { Dimension, DimensionCategory } from '../types/reporting';
import { getDimensions, getDimensionCategories, getDimensionValues } from '../api/reporting';

export const useDimensions = (category?: string) => {
  const [dimensions, setDimensions] = useState<Dimension[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchDimensions = async () => {
      try {
        setLoading(true);
        const data = await getDimensions(category);
        setDimensions(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('An error occurred while fetching dimensions'));
        console.error('Error fetching dimensions:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDimensions();
  }, [category]);

  return { dimensions, loading, error };
};

export const useDimensionCategories = () => {
  const [categories, setCategories] = useState<DimensionCategory[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchCategories = async () => {
      try {
        setLoading(true);
        const data = await getDimensionCategories();
        setCategories(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('An error occurred while fetching dimension categories'));
        console.error('Error fetching dimension categories:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchCategories();
  }, []);

  return { categories, loading, error };
};

export const useDimensionValues = (dimensionId: string, search?: string, limit: number = 100) => {
  const [values, setValues] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const fetchValues = async () => {
      if (!dimensionId) {
        setValues([]);
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const data = await getDimensionValues(dimensionId, search, limit);
        setValues(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err : new Error('An error occurred while fetching dimension values'));
        console.error('Error fetching dimension values:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchValues();
  }, [dimensionId, search, limit]);

  return { values, loading, error };
};
