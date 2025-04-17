/**
 * Service for interacting with the CL-PEFT API
 */

import {
  generateMockAdapters,
  generateMockAdapter,
  generateMockEvaluationResults,
  generateMockForgettingResults
} from '../components/LLM/cl_peft/mockData';

// Set to true to use mock data instead of real API calls
const USE_MOCK_DATA = true;

// API base URL - use environment variable or default
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// API endpoints
const ENDPOINTS = {
  adapters: `${API_BASE_URL}/cl_peft/adapters`,
  adapter: (id) => `${API_BASE_URL}/cl_peft/adapters/${id}`,
  evaluate: (id) => `${API_BASE_URL}/cl_peft/adapters/${id}/evaluate`,
  forgetting: (id) => `${API_BASE_URL}/cl_peft/adapters/${id}/forgetting`,
  generate: (id) => `${API_BASE_URL}/cl_peft/adapters/${id}/generate`,
  train: (id) => `${API_BASE_URL}/cl_peft/adapters/${id}/train`
};

// Mock data storage
const mockStorage = {
  adapters: generateMockAdapters(5)
};

/**
 * Get all adapters
 */
export const getAdapters = async () => {
  if (USE_MOCK_DATA) {
    return { adapters: mockStorage.adapters };
  }

  const response = await fetch(ENDPOINTS.adapters);

  if (!response.ok) {
    throw new Error(`Failed to fetch adapters: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Get adapter by ID
 */
export const getAdapter = async (adapterId) => {
  if (USE_MOCK_DATA) {
    const adapter = mockStorage.adapters.find(a => a.adapter_id === adapterId);

    if (!adapter) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    return { adapter };
  }

  const response = await fetch(ENDPOINTS.adapter(adapterId));

  if (!response.ok) {
    throw new Error(`Failed to fetch adapter: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Create a new adapter
 */
export const createAdapter = async (adapterData) => {
  if (USE_MOCK_DATA) {
    const newAdapter = generateMockAdapter(adapterData);
    mockStorage.adapters.push(newAdapter);

    return { adapter: newAdapter };
  }

  const response = await fetch(ENDPOINTS.adapters, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(adapterData)
  });

  if (!response.ok) {
    throw new Error(`Failed to create adapter: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Update an adapter
 */
export const updateAdapter = async (adapterId, adapterData) => {
  if (USE_MOCK_DATA) {
    const adapterIndex = mockStorage.adapters.findIndex(a => a.adapter_id === adapterId);

    if (adapterIndex === -1) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    const updatedAdapter = {
      ...mockStorage.adapters[adapterIndex],
      ...adapterData,
      updated_at: new Date().toISOString()
    };

    mockStorage.adapters[adapterIndex] = updatedAdapter;

    return { adapter: updatedAdapter };
  }

  const response = await fetch(ENDPOINTS.adapter(adapterId), {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(adapterData)
  });

  if (!response.ok) {
    throw new Error(`Failed to update adapter: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Delete an adapter
 */
export const deleteAdapter = async (adapterId) => {
  if (USE_MOCK_DATA) {
    const adapterIndex = mockStorage.adapters.findIndex(a => a.adapter_id === adapterId);

    if (adapterIndex === -1) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    mockStorage.adapters.splice(adapterIndex, 1);

    return { success: true };
  }

  const response = await fetch(ENDPOINTS.adapter(adapterId), {
    method: 'DELETE'
  });

  if (!response.ok) {
    throw new Error(`Failed to delete adapter: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Evaluate an adapter
 */
export const evaluateAdapter = async (adapterId, evaluationData) => {
  if (USE_MOCK_DATA) {
    const adapter = mockStorage.adapters.find(a => a.adapter_id === adapterId);

    if (!adapter) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    return generateMockEvaluationResults(adapterId, evaluationData.task_id);
  }

  const response = await fetch(ENDPOINTS.evaluate(adapterId), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(evaluationData)
  });

  if (!response.ok) {
    throw new Error(`Failed to evaluate adapter: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Compute forgetting for an adapter
 */
export const computeForgetting = async (adapterId, forgettingData) => {
  if (USE_MOCK_DATA) {
    const adapter = mockStorage.adapters.find(a => a.adapter_id === adapterId);

    if (!adapter) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 1500));

    return generateMockForgettingResults(
      adapterId,
      forgettingData.task_id,
      forgettingData.metric_key
    );
  }

  const response = await fetch(ENDPOINTS.forgetting(adapterId), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(forgettingData)
  });

  if (!response.ok) {
    throw new Error(`Failed to compute forgetting: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Generate text using an adapter
 */
export const generateText = async (adapterId, generationData) => {
  if (USE_MOCK_DATA) {
    const adapter = mockStorage.adapters.find(a => a.adapter_id === adapterId);

    if (!adapter) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Generate mock text based on prompt
    const prompt = generationData.prompt.toLowerCase();
    let response = '';

    if (prompt.includes('diabetes')) {
      response = "# Type 2 Diabetes Mellitus\n\nType 2 diabetes mellitus is a chronic metabolic disorder characterized by insulin resistance and relative insulin deficiency. The pathophysiology involves:\n\n1. **Insulin Resistance**: Cells in muscle, fat, and liver become resistant to insulin, requiring more insulin to facilitate glucose uptake.\n\n2. **Beta Cell Dysfunction**: Initially, pancreatic beta cells increase insulin production to overcome resistance, but eventually fail to keep up, leading to relative insulin deficiency.\n\n3. **Hepatic Glucose Production**: Increased glucose production by the liver contributes to hyperglycemia.\n\n4. **Incretin Effect Impairment**: Reduced incretin hormone effect, which normally stimulates insulin release after meals.\n\n5. **Adipose Tissue Dysfunction**: Dysregulated fat metabolism leads to increased free fatty acids, which worsen insulin resistance.\n\nGenetic factors, obesity, sedentary lifestyle, and aging are key contributors to the development of type 2 diabetes.";
    } else if (prompt.includes('alzheimer')) {
      response = "# Latest Treatment Options for Alzheimer's Disease\n\nAlzheimer's disease treatment has seen significant advancements in recent years:\n\n## FDA-Approved Medications\n\n1. **Aducanumab (Aduhelm)**: First disease-modifying therapy approved in 2021, targeting amyloid plaques.\n\n2. **Lecanemab (Leqembi)**: Approved in 2023, showing modest slowing of cognitive decline by targeting amyloid beta.\n\n3. **Donanemab**: Recently showed promising results in phase 3 trials, awaiting FDA approval.\n\n## Symptomatic Treatments\n\n- **Cholinesterase inhibitors**: Donepezil, rivastigmine, and galantamine\n- **NMDA receptor antagonist**: Memantine\n\n## Emerging Approaches\n\n- **Tau-targeting therapies**: Several in clinical trials\n- **Anti-inflammatory approaches**: Targeting neuroinflammation\n- **Metabolic interventions**: Addressing insulin resistance in the brain\n- **Stem cell therapies**: Early-stage research\n\nCombination therapies and personalized medicine approaches are showing promise for more effective treatment.";
    } else if (prompt.includes('quantum')) {
      response = "# Quantum Entanglement Explained Simply\n\nQuantum entanglement is like having two coins that are magically connected. When you flip one coin and it lands on heads, the other coin instantly lands on tails, no matter how far apart they are.\n\nIn quantum physics terms:\n\n1. Two or more particles become linked so that the quantum state of each particle cannot be described independently.\n\n2. When you measure one particle, you instantly know information about its entangled partner, even if they're separated by vast distances.\n\n3. Einstein called this \"spooky action at a distance\" because it seems to violate the principle that information can't travel faster than light.\n\n4. However, no actual information is transmitted faster than light - the correlation was established when the particles first interacted.\n\nThis phenomenon is fundamental to quantum computing, quantum cryptography, and our understanding of the universe at its most fundamental level.";
    } else {
      response = "I'm a specialized medical and scientific language model. I can provide information about medical conditions, treatments, scientific concepts, and research. Please ask me a specific question in these domains for a more detailed response.";
    }

    return {
      generated_text: response,
      adapter_id: adapterId,
      prompt: generationData.prompt,
      generation_params: generationData
    };
  }

  const response = await fetch(ENDPOINTS.generate(adapterId), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(generationData)
  });

  if (!response.ok) {
    throw new Error(`Failed to generate text: ${response.statusText}`);
  }

  return response.json();
};

/**
 * Train an adapter on a new task
 */
export const trainAdapter = async (adapterId, trainingData) => {
  if (USE_MOCK_DATA) {
    const adapterIndex = mockStorage.adapters.findIndex(a => a.adapter_id === adapterId);

    if (adapterIndex === -1) {
      throw new Error(`Adapter not found: ${adapterId}`);
    }

    // Simulate delay
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Create a new task
    const newTask = {
      task_id: trainingData.task_id || `task_${Math.random().toString(36).substring(2, 15)}`,
      trained_at: new Date().toISOString(),
      metrics: {
        loss: 0.1 + Math.random() * 0.5,
        accuracy: 0.7 + Math.random() * 0.3,
        perplexity: 1.1 + Math.random() * 0.5
      }
    };

    // Add task to adapter
    const updatedAdapter = {
      ...mockStorage.adapters[adapterIndex],
      task_history: [
        newTask,
        ...(mockStorage.adapters[adapterIndex].task_history || [])
      ],
      updated_at: new Date().toISOString()
    };

    mockStorage.adapters[adapterIndex] = updatedAdapter;

    return {
      adapter: updatedAdapter,
      task: newTask
    };
  }

  const response = await fetch(ENDPOINTS.train(adapterId), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(trainingData)
  });

  if (!response.ok) {
    throw new Error(`Failed to train adapter: ${response.statusText}`);
  }

  return response.json();
};
