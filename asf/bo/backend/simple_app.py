"""
Simple FastAPI application without any problematic imports.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI(title="Simple API")

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

@app.post("/items/")
async def create_item(item: Item):
    """Create a new item."""
    return item

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """Get an item by ID."""
    if item_id < 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "name": f"Item {item_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
