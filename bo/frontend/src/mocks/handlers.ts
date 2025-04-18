import { rest } from 'msw';
import { 
  generateMockAdapters, 
  generateMockAdapter, 
  generateMockEvaluationResults 
} from '../components/LLM/cl_peft/mockData';

// Base URL for API
const API_URL = '/api';

/**
 * MSW handlers for mocking API responses
 */
export const handlers = [
  // Auth endpoints
  rest.post(`${API_URL}/login`, (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: {
          access_token: 'mock-token-12345',
          token_type: 'bearer',
          user: {
            id: 1,
            username: 'admin',
            email: 'admin@example.com',
            role_id: 1
          }
        }
      })
    );
  }),

  rest.get(`${API_URL}/me`, (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: {
          id: 1,
          username: 'admin',
          email: 'admin@example.com',
          role_id: 1
        }
      })
    );
  }),

  // LLM endpoints
  rest.get(`${API_URL}/llm/status`, (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: {
          status: 'operational',
          components: {
            gateway: {
              status: 'operational',
              models: {
                'gpt-4o': 2580,
                'claude-3-opus': 1420,
                'biomedlm-2-7b': 3850,
                'mistralai/Mixtral-8x7B': 980
              }
            },
            mcp: {
              status: 'operational',
              providers: 3
            },
            cl_peft: {
              status: 'operational',
              adapters: 8
            },
            dspy: {
              status: 'operational',
              modules: 5
            }
          }
        }
      })
    );
  }),

  // MCP providers
  rest.get(`${API_URL}/llm/mcp/providers`, (req, res, ctx) => {
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: [
          {
            id: 'openai',
            name: 'OpenAI',
            transport_type: 'http',
            status: 'connected',
            config: {
              api_key: '••••••••••••••••',
              base_url: 'https://api.openai.com/v1'
            }
          },
          {
            id: 'anthropic',
            name: 'Anthropic',
            transport_type: 'http',
            status: 'connected',
            config: {
              api_key: '••••••••••••••••',
              base_url: 'https://api.anthropic.com'
            }
          },
          {
            id: 'huggingface',
            name: 'Hugging Face',
            transport_type: 'http',
            status: 'connected',
            config: {
              api_key: '••••••••••••••••',
              base_url: 'https://api-inference.huggingface.co/models'
            }
          }
        ]
      })
    );
  }),

  // CL-PEFT adapters
  rest.get(`${API_URL}/llm/cl-peft/adapters`, (req, res, ctx) => {
    const mockAdapters = generateMockAdapters(8);
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: mockAdapters
      })
    );
  }),

  rest.get(`${API_URL}/llm/cl-peft/adapters/:adapterId`, (req, res, ctx) => {
    const { adapterId } = req.params;
    const mockAdapter = generateMockAdapter({ adapter_id: adapterId as string });
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: mockAdapter
      })
    );
  }),

  rest.post(`${API_URL}/llm/cl-peft/adapters/:adapterId/evaluate`, async (req, res, ctx) => {
    const { adapterId } = req.params;
    const body = await req.json();
    const mockResults = generateMockEvaluationResults(adapterId as string, body.task_id);
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: mockResults
      })
    );
  }),

  // Medical search
  rest.post(`${API_URL}/medical/search`, async (req, res, ctx) => {
    const body = await req.json();
    const { query, max_results = 20 } = body;
    
    // Generate mock search results
    const articles = Array.from({ length: Math.min(max_results, 20) }, (_, i) => ({
      id: `article_${i}`,
      title: `Research on ${query} - Part ${i + 1}`,
      authors: ['Author A', 'Author B'],
      journal: 'Journal of Medical Research',
      year: 2023 - i % 5,
      abstract: `This study investigates ${query} and its implications for healthcare.`,
      relevance_score: 0.95 - (i * 0.02),
      source: 'Mock Data'
    }));
    
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        message: `Found ${articles.length} results for query: ${query}`,
        data: {
          articles,
          query,
          total_results: articles.length
        }
      })
    );
  }),

  // Medical search history
  rest.get(`${API_URL}/medical/search/history`, (req, res, ctx) => {
    const now = new Date();
    const searches = [
      {
        id: 'search_1',
        query: 'pneumonia treatment',
        type: 'standard',
        timestamp: new Date(now.getTime() - 24 * 60 * 60 * 1000).toISOString(),
        result_count: 42
      },
      {
        id: 'search_2',
        query: 'P: adults with hypertension, I: ACE inhibitors, C: ARBs, O: blood pressure reduction',
        type: 'pico',
        timestamp: new Date(now.getTime() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        result_count: 15
      },
      {
        id: 'search_3',
        query: 'diabetes management',
        type: 'standard',
        timestamp: new Date(now.getTime() - 5 * 24 * 60 * 60 * 1000).toISOString(),
        result_count: 78
      }
    ];
    
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        data: {
          searches
        }
      })
    );
  }),

  // Medical clients
  rest.get(`${API_URL}/medical/clients`, (req, res, ctx) => {
    const clients = [
      {
        id: 'ncbi',
        name: 'NCBI',
        description: 'National Center for Biotechnology Information',
        status: 'connected',
        last_checked: new Date().toISOString(),
        api_version: '2.0',
        endpoints: ['pubmed', 'pmc', 'gene']
      },
      {
        id: 'umls',
        name: 'UMLS',
        description: 'Unified Medical Language System',
        status: 'connected',
        last_checked: new Date().toISOString(),
        api_version: '1.5',
        endpoints: ['search', 'semantic-network', 'metathesaurus']
      },
      {
        id: 'clinicaltrials',
        name: 'ClinicalTrials.gov',
        description: 'Database of clinical studies',
        status: 'connected',
        last_checked: new Date().toISOString(),
        api_version: '1.0',
        endpoints: ['search', 'study', 'condition']
      }
    ];
    
    return res(
      ctx.status(200),
      ctx.json({
        success: true,
        message: `Retrieved ${clients.length} medical clients`,
        data: {
          clients,
          total: clients.length
        }
      })
    );
  })
];
