from setuptools import setup, find_packages

setup(
    name="asf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
        "httpx",
        "aiohttp",
        "redis",
        # Don't include pymemgraph as it's causing issues
    ],
)