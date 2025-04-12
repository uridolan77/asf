Setup script for the DSPy integration package.

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README
with open('README.md') as f:
    long_description = f.read()

setup(
    name="asf-medical-dspy",
    version="1.0.0",
    description="DSPy Integration for Medical Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ASF Medical Team",
    author_email="info@example.com",
    url="https://github.com/example/asf-medical-dspy",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires=">=3.8",
)
