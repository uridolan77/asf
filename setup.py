"""
Setup script for the Medical Research Synthesizer.
"""

from setuptools import setup, find_packages

# Read requirements
with open("asf/medical/requirements.txt") as f:
    requirements = f.read().splitlines()

# Read README
with open("asf/medical/README.md") as f:
    long_description = f.read()

setup(
    name="medical-research-synthesizer",
    version="1.0.0",
    description="A comprehensive platform for searching, analyzing, and synthesizing medical research literature",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ASF Team",
    author_email="info@example.com",
    url="https://github.com/yourusername/medical-research-synthesizer",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "medical-research-synthesizer=asf.medical.scripts.run_app:main",
            "mrs-api=asf.medical.scripts.run_api:main",
            "mrs-tests=asf.medical.scripts.run_tests:main",
            "mrs-docs=asf.medical.scripts.generate_docs:main",
        ],
    },
)
