"""
Simple FastAPI server for testing connectivity.
This only sets up the minimal routes needed to verify connectivity.
"""
import os
import sys
from fastapi import FastAPI
import uvicorn

# Create a standalone FastAPI app
app = FastAPI(title="Connectivity Test Server")

@app.get("/")
async def root():
    return {
        "message": "Test server is running correctly",
        "status": "online"
    }

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

if __name__ == "__main__":
    print("Starting connectivity test server on http://127.0.0.1:9000")
    try:
        uvicorn.run(
            app, 
            host="127.0.0.1",
            port=9000,
            log_level="info"
        )
    except Exception as e:
        print(f"Error starting server: {e}")