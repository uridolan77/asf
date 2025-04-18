describe('Claim Visualizer with Backend Mocks', () => {
  beforeEach(() => {
    // Login via API for faster tests
    cy.loginViaApi('testuser', 'testpassword');
    
    // Mock the API response for claims data
    cy.intercept('GET', '/api/claims/visualize*', { fixture: 'claims-data.json' }).as('getClaimsData');
    
    cy.visit('/analysis');
    cy.get('[data-testid="claim-visualizer-tab"]').click();
    cy.wait('@getClaimsData');
  });

  it('should display the claim visualizer component', () => {
    cy.get('[data-testid="claim-visualizer"]').should('be.visible');
    cy.get('[data-testid="claim-node"]').should('have.length.at.least', 1);
    cy.get('[data-testid="claim-connections"]').should('be.visible');
  });

  it('should detect and highlight contradictions', () => {
    // Select contradiction detection mode
    cy.get('[data-testid="contradiction-detection-toggle"]').click();
    cy.get('[data-testid="contradiction-detection-toggle"]').should('have.class', 'active');
    
    // Check if contradictions are highlighted
    cy.get('[data-testid="contradiction-node"]').should('be.visible');
    cy.get('[data-testid="contradiction-edge"]').should('be.visible');
  });

  it('should allow filtering claims by source', () => {
    // Open source filter dropdown
    cy.get('[data-testid="source-filter-dropdown"]').click();
    
    // Select a specific source
    cy.get('[data-testid="source-filter-option"]').first().click();
    
    // Verify that only claims from the selected source are shown
    cy.get('[data-testid="claim-source-tag"]').each(($el) => {
      cy.wrap($el).invoke('text').should('include', 'Source 1');
    });
  });

  it('should allow zooming and panning', () => {
    // Get initial transform value
    cy.get('[data-testid="visualizer-svg"]')
      .invoke('attr', 'transform')
      .as('initialTransform');
    
    // Zoom in
    cy.get('[data-testid="zoom-in-button"]').click();
    
    // Assert transform changed
    cy.get('[data-testid="visualizer-svg"]')
      .invoke('attr', 'transform')
      .should('not.eq', '@initialTransform');
    
    // Pan the visualizer
    cy.get('[data-testid="visualizer-svg"]')
      .trigger('mousedown', { which: 1 })
      .trigger('mousemove', { clientX: 300, clientY: 200 })
      .trigger('mouseup');
    
    // Assert transform changed again
    cy.get('[data-testid="visualizer-svg"]')
      .invoke('attr', 'transform')
      .should('not.eq', '@initialTransform');
  });

  it('should show detailed information when a claim is clicked', () => {
    // Click on a claim node
    cy.get('[data-testid="claim-node"]').first().click();
    
    // Check that claim details panel is visible
    cy.get('[data-testid="claim-details-panel"]').should('be.visible');
    cy.get('[data-testid="claim-text"]').should('be.visible');
    cy.get('[data-testid="claim-source"]').should('be.visible');
    cy.get('[data-testid="claim-evidence"]').should('be.visible');
  });

  it('should allow exporting the visualization', () => {
    // Click export button
    cy.get('[data-testid="export-button"]').click();
    
    // Check export options are available
    cy.get('[data-testid="export-dropdown"]').should('be.visible');
    cy.get('[data-testid="export-as-png"]').should('be.visible');
    cy.get('[data-testid="export-as-svg"]').should('be.visible');
    cy.get('[data-testid="export-as-json"]').should('be.visible');
    
    // Select PNG export
    cy.get('[data-testid="export-as-png"]').click();
    
    // Verify download happened (this is a simplified check)
    cy.get('[data-testid="export-success-notification"]').should('be.visible');
  });
});