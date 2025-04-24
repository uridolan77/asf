// ***********************************************
// Custom commands for Cypress tests
// ***********************************************

// Custom command for UI login
Cypress.Commands.add('login', (username: string, password: string) => {
  cy.visit('/');
  cy.get('[data-testid="username-input"]').type(username);
  cy.get('[data-testid="password-input"]').type(password);
  cy.get('[data-testid="login-button"]').click();
  cy.url().should('include', '/dashboard');
});

// Custom command for API login (faster)
Cypress.Commands.add('loginViaApi', (username: string, password: string) => {
  cy.request({
    method: 'POST',
    url: `${Cypress.env('apiUrl')}/auth/login`,
    body: {
      username,
      password
    }
  }).then((response) => {
    expect(response.status).to.eq(200);
    window.localStorage.setItem('token', response.body.access_token);
    cy.visit('/dashboard');
  });
});

// Custom command for selecting claim options in visualizer
Cypress.Commands.add('selectClaimOption', (option: string) => {
  cy.get(`[data-testid="claim-option-${option}"]`).click();
  cy.get(`[data-testid="claim-option-${option}"]`).should('have.class', 'selected');
});