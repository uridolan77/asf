#!/usr/bin/env python3
from pathlib import Path

# Define all files to create under the new root "agentor"
files = [
    # agents
    "agentor/agents/__init__.py",
    "agentor/agents/base.py",
    "agentor/agents/rule_based.py",
    "agentor/agents/utility_based.py",
    "agentor/agents/learning.py",

    # components
    "agentor/components/__init__.py",
    "agentor/components/memory.py",
    "agentor/components/decision.py",
    "agentor/components/sensors.py",
    "agentor/components/actions.py",
    "agentor/components/learning.py",
    "agentor/components/coordination.py",

    # interfaces
    "agentor/interfaces/__init__.py",
    "agentor/interfaces/api.py",
    "agentor/interfaces/filesystem.py",
    "agentor/interfaces/database.py",
    "agentor/interfaces/network.py",

    # environments
    "agentor/environments/__init__.py",
    "agentor/environments/simple.py",
    "agentor/environments/grid_world.py",
    "agentor/environments/custom.py",

    # utils
    "agentor/utils/__init__.py",
    "agentor/utils/logging.py",
    "agentor/utils/serialization.py",
    "agentor/utils/visualization.py",

    # tests
    "agentor/tests/__init__.py",
    "agentor/tests/test_agents.py",
    "agentor/tests/test_components.py",
    "agentor/tests/test_interfaces.py",
    "agentor/tests/test_utils.py",

    # examples
    "agentor/examples/simple_agent.py",
    "agentor/examples/multi_agent.py",
    "agentor/examples/learning_agent.py",

    # root files
    "agentor/README.md",
    "agentor/setup.py",
    "agentor/requirements.txt",
]

def create_structure(file_list):
    for file in file_list:
        path = Path(file)
        # create any missing parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        # create the file if it doesn't exist
        if not path.exists():
            path.touch()
            print(f"Created: {path}")
        else:
            print(f"Already exists: {path}")

if __name__ == "__main__":
    create_structure(files)
