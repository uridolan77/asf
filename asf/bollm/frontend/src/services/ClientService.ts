import api from './api';

class ClientService {
  // Existing Medical Client methods
  getAllClients() {
    return api.get('/api/clients').then(response => response.data);
  }

  getClient(clientId: string) {
    return api.get(`/api/clients/${clientId}`).then(response => response.data);
  }

  updateClientConfig(clientId: string, config: any) {
    return api.put(`/api/clients/${clientId}/config`, config).then(response => response.data);
  }

  testClientConnection(clientId: string) {
    return api.post(`/api/clients/${clientId}/test-connection`).then(response => response.data);
  }

  getClientUsage(clientId: string, days: number = 30) {
    return api.get(`/api/clients/${clientId}/usage?days=${days}`).then(response => response.data);
  }

  getClientStatusHistory(clientId: string, limit: number = 100) {
    return api.get(`/api/clients/${clientId}/status-history?limit=${limit}`).then(response => response.data);
  }

  // DSPy Client methods
  getAllDSPyClients() {
    return api.get('/api/dspy/clients').then(response => response.data);
  }

  getDSPyClient(clientId: string) {
    return api.get(`/api/dspy/clients/${clientId}`).then(response => response.data);
  }

  createDSPyClient(clientData: any) {
    return api.post('/api/dspy/clients', clientData).then(response => response.data);
  }

  updateDSPyClient(clientId: string, clientData: any) {
    return api.put(`/api/dspy/clients/${clientId}`, clientData).then(response => response.data);
  }

  deleteDSPyClient(clientId: string) {
    return api.delete(`/api/dspy/clients/${clientId}`).then(response => response.data);
  }

  updateDSPyClientConfig(clientId: string, config: any) {
    return api.put(`/api/dspy/clients/${clientId}/config`, config).then(response => response.data);
  }

  testDSPyClientConnection(clientId: string) {
    return api.post(`/api/dspy/clients/${clientId}/test-connection`).then(response => response.data);
  }

  getDSPyModules(clientId: string) {
    return api.get(`/api/dspy/clients/${clientId}/modules`).then(response => response.data);
  }

  getDSPyModule(moduleId: string) {
    return api.get(`/api/dspy/modules/${moduleId}`).then(response => response.data);
  }

  createDSPyModule(moduleData: any) {
    return api.post('/api/dspy/modules', moduleData).then(response => response.data);
  }

  syncDSPyModules(clientId: string) {
    return api.post(`/api/dspy/clients/${clientId}/sync-modules`).then(response => response.data);
  }

  getDSPyClientUsage(clientId: string, days: number = 30) {
    return api.get(`/api/dspy/clients/${clientId}/usage?days=${days}`).then(response => response.data);
  }

  getDSPyClientStatusHistory(clientId: string, limit: number = 100) {
    return api.get(`/api/dspy/clients/${clientId}/status-history?limit=${limit}`).then(response => response.data);
  }

  getDSPyAuditLogs(clientId: string, options: any = {}) {
    const { moduleId, eventType, startDate, endDate, limit = 100, offset = 0 } = options;
    
    let url = `/api/dspy/clients/${clientId}/audit-logs?limit=${limit}&offset=${offset}`;
    
    if (moduleId) url += `&module_id=${moduleId}`;
    if (eventType) url += `&event_type=${eventType}`;
    if (startDate) url += `&start_date=${startDate}`;
    if (endDate) url += `&end_date=${endDate}`;
    
    return api.get(url).then(response => response.data);
  }

  getCircuitBreakerStatus(clientId: string) {
    return api.get(`/api/dspy/clients/${clientId}/circuit-breakers`).then(response => response.data);
  }
}

export default new ClientService();