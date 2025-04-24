describe('LLM Management UI', () => {
  beforeEach(() => {
    // Login via API for faster tests
    cy.loginViaApi('testuser', 'testpassword');
    cy.visit('/llm-management');
  });

  it('should display the LLM management page', () => {
    cy.get('[data-testid="llm-management-title"]').should('be.visible');
    cy.get('[data-testid="llm-models-list"]').should('be.visible');
  });

  it('should allow adding a new LLM model', () => {
    // Click add model button
    cy.get('[data-testid="add-llm-button"]').click();
    
    // Fill in the form
    cy.get('[data-testid="model-name-input"]').type('Test GPT Model');
    cy.get('[data-testid="model-endpoint-input"]').type('https://api.test.com/completions');
    cy.get('[data-testid="model-api-key-input"]').type('test-api-key-12345');
    cy.get('[data-testid="model-type-select"]').click();
    cy.get('[data-value="openai"]').click();
    
    // Submit the form
    cy.get('[data-testid="submit-model-button"]').click();
    
    // Verify the model was added
    cy.get('[data-testid="llm-models-list"]').should('contain', 'Test GPT Model');
    cy.get('[data-testid="success-notification"]').should('be.visible');
  });

  it('should allow editing an existing LLM model', () => {
    // Find and click edit button for first model
    cy.get('[data-testid="edit-model-button"]').first().click();
    
    // Update model name
    cy.get('[data-testid="model-name-input"]').clear().type('Updated Model Name');
    
    // Submit the form
    cy.get('[data-testid="submit-model-button"]').click();
    
    // Verify the model was updated
    cy.get('[data-testid="llm-models-list"]').should('contain', 'Updated Model Name');
    cy.get('[data-testid="success-notification"]').should('be.visible');
  });

  it('should allow deleting an LLM model', () => {
    // Get the name of the first model for later verification
    cy.get('[data-testid="model-name"]').first().invoke('text').as('modelName');
    
    // Find and click delete button for first model
    cy.get('[data-testid="delete-model-button"]').first().click();
    
    // Confirm deletion
    cy.get('[data-testid="confirm-delete-button"]').click();
    
    // Verify the model was deleted
    cy.get('@modelName').then((modelName) => {
      cy.get('[data-testid="llm-models-list"]').should('not.contain', modelName);
    });
    cy.get('[data-testid="success-notification"]').should('be.visible');
  });

  it('should display LLM model performance metrics', () => {
    // Click on metrics tab
    cy.get('[data-testid="metrics-tab"]').click();
    
    // Verify metrics charts are visible
    cy.get('[data-testid="response-time-chart"]').should('be.visible');
    cy.get('[data-testid="token-usage-chart"]').should('be.visible');
    cy.get('[data-testid="model-comparison-chart"]').should('be.visible');
  });
});