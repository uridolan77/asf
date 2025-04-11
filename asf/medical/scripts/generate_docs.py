Script to generate documentation for the Medical Research Synthesizer.

This script generates documentation using Sphinx.

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def main():
    """Generate documentation.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    parser = argparse.ArgumentParser(description="Generate documentation for the Medical Research Synthesizer")
    parser.add_argument("--clean", action="store_true", help="Clean build directory before generating")
    parser.add_argument("--format", choices=["html", "pdf", "all"], default="html", help="Output format")
    args = parser.parse_args()
    
    docs_dir = Path(__file__).parent.parent.parent.parent / "docs"
    build_dir = docs_dir / "build"
    
    if not docs_dir.exists():
        print(f"Creating docs directory: {docs_dir}")
        docs_dir.mkdir(parents=True)
    
    conf_py = docs_dir / "conf.py"
    if not conf_py.exists():
        print(f"Creating Sphinx configuration: {conf_py}")
        os.system(f"sphinx-quickstart --quiet --project='Medical Research Synthesizer' --author='ASF Team' {docs_dir}")
    
    if args.clean and build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        os.system(f"rm -rf {build_dir}")
    
    print("Generating API documentation")
    os.system(f"sphinx-apidoc -o {docs_dir}/api asf/medical")
    
    if args.format == "html" or args.format == "all":
        print("Building HTML documentation")
        os.system(f"sphinx-build -b html {docs_dir} {build_dir}/html")
    
    if args.format == "pdf" or args.format == "all":
        print("Building PDF documentation")
        os.system(f"sphinx-build -b latex {docs_dir} {build_dir}/latex")
        os.system(f"cd {build_dir}/latex && make")
    
    print("Documentation generated successfully")

if __name__ == "__main__":
    main()
