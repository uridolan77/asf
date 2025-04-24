#!/usr/bin/env python
"""
Direct Backoffice Login Script

This script bypasses the normal authentication flow by:
1. Creating a minimal app with only necessary components
2. Explicitly setting up models to avoid conflicts
3. Creating a demo user with admin privileges
4. Starting a server that will automatically log you in

Usage: 
python direct_backoffice_login.py
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import jwt
from datetime import datetime, timedelta, timezone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("backoffice-login")

# Create minimal FastAPI app
app = FastAPI(title="Backoffice Direct Access")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT constants - just for demo
SECRET_KEY = "demo_secret_key_not_for_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Demo user
DEMO_USER = {
    "id": 1,
    "email": "admin@example.com",
    "full_name": "Demo Admin",
    "role": "admin"
}

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.get("/", response_class=HTMLResponse)
async def root():
    """Display login page with auto-login button"""
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Direct Backoffice Access</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f4f7f9;
                }
                .container {
                    background: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    text-align: center;
                    max-width: 500px;
                }
                h1 { color: #333; }
                .warning {
                    background: #fff3cd;
                    border: 1px solid #ffeeba;
                    color: #856404;
                    padding: 1rem;
                    border-radius: 4px;
                    margin-bottom: 1.5rem;
                }
                .button {
                    display: inline-block;
                    background-color: #4CAF50;
                    color: white;
                    padding: 12px 24px;
                    text-align: center;
                    text-decoration: none;
                    font-size: 16px;
                    border-radius: 4px;
                    border: none;
                    cursor: pointer;
                    margin-top: 1rem;
                    font-weight: bold;
                }
                .button:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Direct Backoffice Access</h1>
                <div class="warning">
                    <strong>Demo Mode</strong>: This is a demonstration bypass that logs you in as an admin user.
                </div>
                <p>Click the button below to automatically log in to the backoffice with admin privileges.</p>
                <a href="/auto-login" class="button">Login as Admin</a>
            </div>
        </body>
    </html>
    """

@app.get("/auto-login")
async def auto_login():
    """Create a token and redirect to the backoffice"""
    # Create token with user claims
    token_data = {
        "sub": DEMO_USER["email"],
        "id": DEMO_USER["id"],
        "role": DEMO_USER["role"]
    }
    access_token = create_access_token(token_data)
    
    # In a real app, we would store this token in secure cookies
    # But for this demo, we'll add it as a query parameter
    logger.info(f"Created demo access token for {DEMO_USER['email']}")
    
    # Redirect to the backoffice with the token
    return RedirectResponse(url=f"/backoffice?token={access_token}")

@app.get("/backoffice", response_class=HTMLResponse)
async def backoffice(request: Request, token: str = None):
    """Mock backoffice page that will use the token"""
    if not token:
        return RedirectResponse(url="/")
    
    try:
        # Verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        user_role = payload.get("role")
        
        # Here you would normally validate the token in more detail
        if user_role != "admin":
            return HTMLResponse(content="Unauthorized: Admin role required", status_code=403)
            
        # Show backoffice interface
        return f"""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Backoffice Dashboard</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        margin: 0;
                        padding: 0;
                    }}
                    .header {{
                        background-color: #2c3e50;
                        color: white;
                        padding: 1rem;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }}
                    .main {{
                        padding: 2rem;
                    }}
                    .user-info {{
                        display: flex;
                        align-items: center;
                    }}
                    .avatar {{
                        width: 36px;
                        height: 36px;
                        border-radius: 50%;
                        background-color: #3498db;
                        color: white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 1rem;
                        font-weight: bold;
                    }}
                    .menu {{
                        display: flex;
                    }}
                    .menu-item {{
                        color: white;
                        margin-left: 1.5rem;
                        cursor: pointer;
                    }}
                    .card {{
                        background: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        padding: 1.5rem;
                        margin-bottom: 1.5rem;
                    }}
                    .stats {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                        gap: 1.5rem;
                        margin-bottom: 2rem;
                    }}
                    .stat-card {{
                        background: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        padding: 1.5rem;
                    }}
                    .stat-value {{
                        font-size: 2rem;
                        font-weight: bold;
                        color: #2c3e50;
                    }}
                    .stat-label {{
                        color: #7f8c8d;
                        margin-top: 0.5rem;
                    }}
                    .token-info {{
                        background: #f8f9fa;
                        border: 1px solid #e9ecef;
                        border-radius: 4px;
                        padding: 0.75rem;
                        font-family: monospace;
                        font-size: 0.9rem;
                        overflow: auto;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Medical Research Synthesizer Backoffice</h1>
                    <div class="user-info">
                        <div class="avatar">{user_email[0].upper()}</div>
                        <span>{user_email} ({user_role})</span>
                    </div>
                </div>
                
                <div class="main">
                    <h2>Welcome to the Backoffice Dashboard</h2>
                    <p>You are now logged in as an admin user. In a real application, you would see administrative controls here.</p>
                    
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-value">42</div>
                            <div class="stat-label">Active Users</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">187</div>
                            <div class="stat-label">Documents Processed</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">93%</div>
                            <div class="stat-label">System Uptime</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value">12</div>
                            <div class="stat-label">Active Models</div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>Authentication Information</h3>
                        <p>For demonstration purposes, here is information about your authentication:</p>
                        <div class="token-info">
                            Token is valid and you are properly authenticated as {user_email} with {user_role} role.
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>Next Steps</h3>
                        <p>In the real application, you would be able to:</p>
                        <ul>
                            <li>Manage users and permissions</li>
                            <li>Configure system settings</li>
                            <li>View analytics and performance metrics</li>
                            <li>Manage models and knowledge bases</li>
                            <li>Monitor system health</li>
                        </ul>
                    </div>
                </div>
            </body>
        </html>
        """
    except jwt.PyJWTError:
        return HTMLResponse(content="Invalid token", status_code=401)

if __name__ == "__main__":
    logger.info("Starting direct backoffice login server")
    uvicorn.run(app, host="0.0.0.0", port=8000)