import os

# Define the folder structure as dictionary where keys are directory paths and values are lists of files to create.
project_structure = {
    "backend": {
        "subdirs": {
            "backend/api": ["__init__.py", "endpoints.py"],
            "backend/config": ["config.py"],
            "backend/models": ["__init__.py"],
        },
        "files": {
            "backend/run.py": "# Entry point for backend server\n\nif __name__ == '__main__':\n    print('Running backend server...')\n",
            "backend/requirements.txt": "# Add your Python dependencies here\nflask\n"
        },
    },
    "frontend": {
        "subdirs": {
            "frontend/public": ["index.html"],
            "frontend/src": [],
            "frontend/src/components": [],
            "frontend/src/components/Dashboard": ["DashboardWidget.js"],
            "frontend/src/components/Navigation": ["SidebarMenu.js"],
            "frontend/src/components/Common": ["Header.js"],
            "frontend/src/pages": ["Login.js", "Dashboard.js", "Settings.js"],
            "frontend/src/services": ["apiService.js"],
        },
        "files": {
            "frontend/src/App.js":
            """// Main App Component
import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>Back Office Dashboard</h1>
    </div>
  );
}

export default App;
""",
            "frontend/src/index.js":
            """// Entry point for React app
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);
""",
            "frontend/public/index.html":
            """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Back Office Dashboard</title>
</head>
<body>
  <div id="root"></div>
</body>
</html>
""",
            "frontend/package.json":
            """{
  "name": "backoffice-frontend",
  "version": "1.0.0",
  "description": "Front-end for Back Office Dashboard",
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "latest"
  }
}
""",
            "frontend/README.md": "# Front-end\nThis folder contains the React-based back-office UI.\n"
        },
    },
    "docs": {
        "subdirs": {},
        "files": {
            "docs/architecture.md": "# Architecture Documentation\n\nDescribe the system architecture here.\n"
        },
    },
    "": {  # root-level files
        "subdirs": {},
        "files": {
            "README.md": "# Project Back Office\nThis repository includes the backend and frontend code for the Back Office Dashboard.\n"
        },
    },
}

def create_structure(base_path, structure):
    for dir_path, contents in structure.items():
        # Create subdirectories first
        subdirs = contents.get("subdirs", {})
        for subdir, files in subdirs.items():
            full_path = os.path.join(base_path, subdir)
            os.makedirs(full_path, exist_ok=True)
            # Create files within the subdirectory
            for filename in files:
                file_path = os.path.join(full_path, filename)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write(f"// {filename} - starter file\n")
                    print(f"Created file: {file_path}")

        # Then create files directly under this level
        files = contents.get("files", {})
        for filepath, content in files.items():
            full_file_path = os.path.join(base_path, filepath)
            # Make sure the folder exists
            os.makedirs(os.path.dirname(full_file_path), exist_ok=True)
            with open(full_file_path, 'w') as f:
                f.write(content)
            print(f"Created file: {full_file_path}")

if __name__ == '__main__':
    base = os.getcwd()  # Run this script from the project root
    create_structure(base, project_structure)
    print("Project structure created successfully!")