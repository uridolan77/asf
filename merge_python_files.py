import os

def merge_python_files(input_dir, output_file, include_init=False, recursive=True, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ['node_modules']  # Default to excluding node_modules
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(input_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in sorted(files):
                if (file.endswith(".py") or file.endswith(".js") or file.endswith(".ts")) and (include_init or file != "__init__.py"):   
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, input_dir)

                    outfile.write("\n\n")
                    outfile.write("#" * 80 + "\n")
                    outfile.write(f"# FILE: {relative_path}\n")
                    outfile.write("#" * 80 + "\n\n")

                    with open(file_path, 'r', encoding='utf-8') as infile:
                        contents = infile.read()
                        outfile.write(contents)
                        outfile.write("\n\n")

            if not recursive:
                break

    print(f"✅ Merged all Python files from '{input_dir}' into '{output_file}'")
    print(f"📂 Excluded directories: {exclude_dirs}")

# Example usage
if __name__ == "__main__":
    merge_python_files(
        input_dir="c:\\code\\asf\\",      # 📁 Your source folder
        output_file="c:\\code\\asf-code.txt",   # 📄 Destination file
        include_init=True,               # ⛔ Skip __init__.py by default
        recursive=True,                  # ✅ Traverse subfolders
        exclude_dirs=['node_modules', 'venv', '__pycache__']  # 🚫 Directories to exclude
    )
