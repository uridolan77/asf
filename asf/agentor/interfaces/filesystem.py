

class FileSystemInterface:
    """Interface for interacting with the file system."""
    
    def __init__(self, base_dir=None):
        """
        Initialize a file system interface.
        
        Args:
            base_dir: Optional base directory for operations
        """
        import os
        self.base_dir = base_dir or os.getcwd()
    
    def read_file(self, filename):
        """Read the contents of a file."""
        import os
        path = os.path.join(self.base_dir, filename)
        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read {path}: {str(e)}")
    
    def write_file(self, filename, content):
        """Write content to a file."""
        import os
        path = os.path.join(self.base_dir, filename)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return True
        except Exception as e:
            raise IOError(f"Failed to write to {path}: {str(e)}")
    
    def list_files(self, directory=None):
        """List files in a directory."""
        import os
        path = os.path.join(self.base_dir, directory or '')
        try:
            return os.listdir(path)
        except Exception as e:
            raise IOError(f"Failed to list files in {path}: {str(e)}")