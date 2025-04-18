// Type definitions for Cypress commands
/// <reference types="cypress" />

declare namespace Cypress {
  interface Chainable {
    /**
     * Custom command to login using UI
     * @example cy.login('username', 'password')
     */
    login(username: string, password: string): Chainable<Element>;
    
    /**
     * Custom command to login via API directly (faster than UI)
     * @example cy.loginViaApi('username', 'password')
     */
    loginViaApi(username: string, password: string): Chainable<Element>;
    
    /**
     * Select an option in the claim visualizer
     * @example cy.selectClaimOption('contradiction')
     */
    selectClaimOption(option: string): Chainable<Element>;
  }
}