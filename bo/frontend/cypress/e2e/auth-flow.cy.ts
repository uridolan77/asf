describe('Authentication Flow', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('should display login form', () => {
    cy.get('[data-testid="login-form"]').should('be.visible');
    cy.get('[data-testid="username-input"]').should('be.visible');
    cy.get('[data-testid="password-input"]').should('be.visible');
    cy.get('[data-testid="login-button"]').should('be.visible');
  });

  it('should show error with invalid credentials', () => {
    cy.get('[data-testid="username-input"]').type('invaliduser');
    cy.get('[data-testid="password-input"]').type('invalidpass');
    cy.get('[data-testid="login-button"]').click();
    
    cy.get('[data-testid="error-message"]')
      .should('be.visible')
      .and('contain', 'Invalid credentials');
    
    cy.url().should('not.include', '/dashboard');
  });

  it('should login successfully with valid credentials', () => {
    // Using test account - in real implementation, use cy.intercept to mock API
    cy.get('[data-testid="username-input"]').type('testuser');
    cy.get('[data-testid="password-input"]').type('testpassword');
    cy.get('[data-testid="login-button"]').click();
    
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="user-menu"]').should('be.visible');
  });

  it('should maintain session after page refresh', () => {
    // Login first
    cy.login('testuser', 'testpassword');
    
    // Verify logged in state
    cy.get('[data-testid="user-menu"]').should('be.visible');
    
    // Refresh the page
    cy.reload();
    
    // Should still be logged in
    cy.url().should('include', '/dashboard');
    cy.get('[data-testid="user-menu"]').should('be.visible');
  });

  it('should redirect to login after logout', () => {
    // Login first
    cy.login('testuser', 'testpassword');
    
    // Logout
    cy.get('[data-testid="user-menu"]').click();
    cy.get('[data-testid="logout-button"]').click();
    
    // Should redirect to login
    cy.url().should('not.include', '/dashboard');
    cy.get('[data-testid="login-form"]').should('be.visible');
  });
});