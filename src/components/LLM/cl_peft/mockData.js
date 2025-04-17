/**
 * Mock data for CL-PEFT components
 */

// Generate a random ID
const generateId = () => Math.random().toString(36).substring(2, 15);

// Generate a random date in the past
const generatePastDate = (daysAgo = 30) => {
  const date = new Date();
  date.setDate(date.getDate() - Math.floor(Math.random() * daysAgo));
  return date.toISOString();
};

// Generate mock adapter
export const generateMockAdapter = (overrides = {}) => {
  const id = generateId();
  const createdAt = generatePastDate(60);
  const updatedAt = generatePastDate(30);
  
  // Generate random task history
  const taskHistory = [];
  const numTasks = Math.floor(Math.random() * 5) + 1;
  
  for (let i = 0; i < numTasks; i++) {
    taskHistory.push({
      task_id: `task_${generateId()}`,
      trained_at: generatePastDate(30 - i * 5),
      metrics: {
        loss: 0.1 + Math.random() * 0.5,
        accuracy: 0.7 + Math.random() * 0.3,
        perplexity: 1.1 + Math.random() * 0.5
      },
      eval_metrics: Math.random() > 0.3 ? {
        eval_loss: 0.2 + Math.random() * 0.5,
        eval_accuracy: 0.65 + Math.random() * 0.3,
        eval_perplexity: 1.2 + Math.random() * 0.5
      } : null
    });
  }
  
  return {
    adapter_id: `adapter_${id}`,
    adapter_name: overrides.adapter_name || `Adapter ${id.substring(0, 5)}`,
    base_model_name: overrides.base_model_name || 'meta-llama/Llama-2-7b-hf',
    cl_strategy: overrides.cl_strategy || 'ewc',
    peft_method: overrides.peft_method || 'lora',
    description: overrides.description || 'A mock adapter for testing purposes',
    status: overrides.status || ['ready', 'initializing', 'training', 'error'][Math.floor(Math.random() * 4)],
    created_at: createdAt,
    updated_at: updatedAt,
    task_history: taskHistory,
    tags: overrides.tags || ['medical', 'test', 'demo'].slice(0, Math.floor(Math.random() * 3) + 1),
    ...overrides
  };
};

// Generate mock adapters
export const generateMockAdapters = (count = 5) => {
  const adapters = [];
  
  // Predefined adapters for consistency
  const predefinedAdapters = [
    {
      adapter_name: 'Medical Knowledge Adapter',
      base_model_name: 'meta-llama/Llama-2-7b-hf',
      cl_strategy: 'ewc',
      peft_method: 'lora',
      description: 'Specialized adapter for medical knowledge and terminology',
      status: 'ready',
      tags: ['medical', 'production']
    },
    {
      adapter_name: 'Clinical Guidelines Adapter',
      base_model_name: 'mistralai/Mistral-7B-v0.1',
      cl_strategy: 'replay_experience',
      peft_method: 'adalora',
      description: 'Adapter fine-tuned on clinical guidelines and protocols',
      status: 'ready',
      tags: ['clinical', 'guidelines']
    },
    {
      adapter_name: 'Research Literature Adapter',
      base_model_name: 'meta-llama/Llama-2-13b-hf',
      cl_strategy: 'orthogonal',
      peft_method: 'ia3',
      description: 'Specialized for scientific research literature comprehension',
      status: 'training',
      tags: ['research', 'literature']
    }
  ];
  
  // Add predefined adapters
  for (let i = 0; i < Math.min(count, predefinedAdapters.length); i++) {
    adapters.push(generateMockAdapter(predefinedAdapters[i]));
  }
  
  // Add random adapters if needed
  for (let i = predefinedAdapters.length; i < count; i++) {
    adapters.push(generateMockAdapter());
  }
  
  return adapters;
};

// Generate mock evaluation results
export const generateMockEvaluationResults = (adapterId, taskId) => {
  return {
    adapter_id: adapterId,
    task_id: taskId,
    timestamp: new Date().toISOString(),
    results: {
      eval_loss: 0.1 + Math.random() * 0.5,
      eval_accuracy: 0.7 + Math.random() * 0.3,
      eval_perplexity: 1.1 + Math.random() * 0.5,
      eval_f1: 0.75 + Math.random() * 0.2,
      eval_precision: 0.7 + Math.random() * 0.3,
      eval_recall: 0.7 + Math.random() * 0.3
    }
  };
};

// Generate mock forgetting results
export const generateMockForgettingResults = (adapterId, taskId, metricKey = 'eval_loss') => {
  // Generate forgetting curve
  const tasks = ['task_1', 'task_2', 'task_3', 'task_4'];
  const forgettingCurve = {};
  
  for (const task of tasks) {
    if (metricKey === 'eval_loss') {
      // For loss, lower is better, so forgetting means increasing values
      forgettingCurve[task] = 0.1 + Math.random() * 0.1 * tasks.indexOf(task);
    } else {
      // For accuracy/f1/etc., higher is better, so forgetting means decreasing values
      forgettingCurve[task] = 0.9 - Math.random() * 0.1 * tasks.indexOf(task);
    }
  }
  
  return {
    adapter_id: adapterId,
    task_id: taskId,
    metric_key: metricKey,
    timestamp: new Date().toISOString(),
    forgetting: metricKey === 'eval_loss' ? 0.15 : 0.12,
    forgetting_curve: forgettingCurve
  };
};
