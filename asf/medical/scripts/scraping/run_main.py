import requests
import yaml
import json
from pathlib import Path

def download_openai_api_spec():
    """Download and process the OpenAI API specification."""
    # Create output directory
    output_dir = Path("openai_docs")
    output_dir.mkdir(exist_ok=True)
    
    # Download the OpenAPI specification
    spec_url = "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"
    print(f"Downloading OpenAI API specification...")
    response = requests.get(spec_url)
    response.raise_for_status()
    
    # Save the raw specification
    spec_path = output_dir / "openai_api_spec.yaml"
    with open(spec_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    
    # Convert to markdown
    spec_data = yaml.safe_load(response.text)
    
    md_path = output_dir / "openai_api_reference.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# OpenAI API Reference\n\n")
        
        # Write various sections of the API documentation
        if 'info' in spec_data:
            f.write(f"## {spec_data['info'].get('title', 'Overview')}\n\n")
            f.write(f"{spec_data['info'].get('description', '')}\n\n")
        
        # Process and write all endpoints
        f.write("## API Endpoints\n\n")
        for path, methods in spec_data.get('paths', {}).items():
            f.write(f"### {path}\n\n")
            
            for method, details in methods.items():
                f.write(f"#### {method.upper()}\n\n")
                f.write(f"{details.get('description', '')}\n\n")
    
    print(f"API reference created at {md_path}")
    return md_path