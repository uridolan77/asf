/**
 * Mock data generator for CL-PEFT components
 * This file provides functions to generate mock data for testing the CL-PEFT components
 */

import { faker } from '@faker-js/faker';

/**
 * Generate a mock adapter
 * @param {Object} overrides - Properties to override in the generated adapter
 * @returns {Object} - A mock adapter
 */
export const generateMockAdapter = (overrides = {}) => {
  const adapterId = overrides.adapter_id || `adapter_${faker.string.alphanumeric(8)}`;
  const createdAt = overrides.created_at || faker.date.past().toISOString();
  const updatedAt = overrides.updated_at || faker.date.recent().toISOString();
  
  const clStrategies = ['naive', 'ewc', 'replay', 'generative_replay', 'orthogonal_lora', 'adaptive_svd', 'mask_based'];
  const peftMethods = ['lora', 'qlora'];
  const statuses = ['initializing', 'ready', 'training', 'error'];
  const baseModels = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'mistralai/Mistral-7B-v0.1',
    'google/gemma-2b',
    'google/gemma-7b',
    'tiiuae/falcon-7b'
  ];
  
  return {
    adapter_id: adapterId,
    adapter_name: overrides.adapter_name || faker.word.words({ count: { min: 1, max: 3 } }),
    base_model_name: overrides.base_model_name || faker.helpers.arrayElement(baseModels),
    description: overrides.description || faker.lorem.sentence(),
    cl_strategy: overrides.cl_strategy || faker.helpers.arrayElement(clStrategies),
    peft_method: overrides.peft_method || faker.helpers.arrayElement(peftMethods),
    status: overrides.status || faker.helpers.arrayElement(statuses),
    created_at: createdAt,
    updated_at: updatedAt,
    task_history: overrides.task_history || generateMockTaskHistory(adapterId, 0, 5),
    tags: overrides.tags || Array.from({ length: faker.number.int({ min: 0, max: 5 }) }, () => faker.word.sample())
  };
};

/**
 * Generate a list of mock adapters
 * @param {number} count - Number of adapters to generate
 * @returns {Array} - List of mock adapters
 */
export const generateMockAdapters = (count = 10) => {
  return Array.from({ length: count }, () => generateMockAdapter());
};

/**
 * Generate a mock task
 * @param {string} adapterId - Adapter ID
 * @param {Object} overrides - Properties to override in the generated task
 * @returns {Object} - A mock task
 */
export const generateMockTask = (adapterId, overrides = {}) => {
  const taskId = overrides.task_id || `task_${faker.string.alphanumeric(8)}`;
  const trainedAt = overrides.trained_at || faker.date.recent().toISOString();
  
  // Generate random metrics
  const metrics = {
    loss: faker.number.float({ min: 0.1, max: 2.0, precision: 0.0001 }),
    accuracy: faker.number.float({ min: 0.7, max: 0.99, precision: 0.0001 }),
    f1: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
    precision: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
    recall: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
    runtime: faker.number.float({ min: 10, max: 1000, precision: 0.01 }),
    samples_per_second: faker.number.float({ min: 1, max: 100, precision: 0.01 })
  };
  
  // Generate random evaluation metrics (50% chance)
  const hasEvalMetrics = faker.datatype.boolean();
  const evalMetrics = hasEvalMetrics ? {
    eval_loss: faker.number.float({ min: 0.1, max: 2.0, precision: 0.0001 }),
    eval_accuracy: faker.number.float({ min: 0.7, max: 0.99, precision: 0.0001 }),
    eval_f1: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
    eval_precision: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
    eval_recall: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 })
  } : null;
  
  return {
    task_id: taskId,
    trained_at: trainedAt,
    metrics: overrides.metrics || metrics,
    eval_metrics: overrides.eval_metrics || evalMetrics
  };
};

/**
 * Generate a list of mock tasks
 * @param {string} adapterId - Adapter ID
 * @param {number} min - Minimum number of tasks to generate
 * @param {number} max - Maximum number of tasks to generate
 * @returns {Array} - List of mock tasks
 */
export const generateMockTaskHistory = (adapterId, min = 0, max = 10) => {
  const count = faker.number.int({ min, max });
  return Array.from({ length: count }, () => generateMockTask(adapterId));
};

/**
 * Generate mock evaluation results
 * @param {string} adapterId - Adapter ID
 * @param {string} taskId - Task ID
 * @returns {Object} - Mock evaluation results
 */
export const generateMockEvaluationResults = (adapterId, taskId) => {
  return {
    adapter_id: adapterId,
    task_id: taskId,
    results: {
      eval_loss: faker.number.float({ min: 0.1, max: 2.0, precision: 0.0001 }),
      eval_accuracy: faker.number.float({ min: 0.7, max: 0.99, precision: 0.0001 }),
      eval_f1: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
      eval_precision: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
      eval_recall: faker.number.float({ min: 0.6, max: 0.95, precision: 0.0001 }),
      eval_runtime: faker.number.float({ min: 10, max: 1000, precision: 0.01 }),
      eval_samples_per_second: faker.number.float({ min: 1, max: 100, precision: 0.01 }),
      eval_steps_per_second: faker.number.float({ min: 0.1, max: 10, precision: 0.01 })
    }
  };
};

/**
 * Generate mock forgetting results
 * @param {string} adapterId - Adapter ID
 * @param {string} taskId - Task ID
 * @param {string} metricKey - Metric key
 * @returns {Object} - Mock forgetting results
 */
export const generateMockForgettingResults = (adapterId, taskId, metricKey = 'eval_loss') => {
  return {
    adapter_id: adapterId,
    task_id: taskId,
    forgetting: faker.number.float({ min: 0.01, max: 0.3, precision: 0.0001 }),
    metric_key: metricKey
  };
};
