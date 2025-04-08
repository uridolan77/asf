# Script to create the new 'SocratesNS_Refactored' project structure

import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define the name for the new project directory
new_project_root = "SocratesNS"

# Define the base source directory within the new project
base_src_dir = os.path.join(new_project_root, "src")

# Define the directory structure to create *within* the new project root
# These paths are relative to the location where the script is run
dirs_to_create = [
    # Top-level conceptual directories within src
    os.path.join(base_src_dir, "api"),
    os.path.join(base_src_dir, "neuro", "embeddings"),
    os.path.join(base_src_dir, "symbolic", "knowledge_base"),
    os.path.join(base_src_dir, "symbolic", "rules"),
    os.path.join(base_src_dir, "interface"),
    os.path.join(base_src_dir, "orchestration", "strategies"),

    # Refined compliance directory within src
    os.path.join(base_src_dir, "compliance", "frameworks"),
    os.path.join(base_src_dir, "compliance", "knowledge_base", "linkers"),
    os.path.join(base_src_dir, "compliance", "monitoring"),
    os.path.join(base_src_dir, "compliance", "proofs"),
    os.path.join(base_src_dir, "compliance", "reasoning"),
    os.path.join(base_src_dir, "compliance", "review"),
    os.path.join(base_src_dir, "compliance", "utils"),
    os.path.join(base_src_dir, "compliance", "verification", "handlers"),

    # NER directory within src
    os.path.join(base_src_dir, "ner"), # Ensure base NER exists
    os.path.join(base_src_dir, "ner", "classifiers"),
    os.path.join(base_src_dir, "ner", "language_model"),
    os.path.join(base_src_dir, "ner", "linkers"),

    # Refined utils directory within src
    os.path.join(base_src_dir, "utils"), # Ensure base utils exists
    os.path.join(base_src_dir, "utils", "cache"),
    os.path.join(base_src_dir, "utils", "config"),
    os.path.join(base_src_dir, "utils", "performance"),
    os.path.join(base_src_dir, "utils", "text"),
    os.path.join(base_src_dir, "utils", "legacy_core_utils"), # For moved core utils

    # Compliance monitoring legacy folder
    os.path.join(base_src_dir, "compliance", "monitoring", "legacy_core_monitoring"),

    # Tests directory at the top level of the new project
    os.path.join(new_project_root, "tests")
]

# --- Create New Project Root and Subdirectories ---
logging.info(f"Creating new project structure in '{new_project_root}'...")
created_dirs_count = 0

# Create the root directory first
try:
    if not os.path.exists(new_project_root):
        os.makedirs(new_project_root)
        logging.info(f"Created root directory: {new_project_root}")
        created_dirs_count += 1
    else:
        logging.info(f"Root directory '{new_project_root}' already exists.")
except OSError as e:
    logging.error(f"Error creating root directory {new_project_root}: {e}")
    exit() # Stop if root cannot be created

# Create all subdirectories
for directory in dirs_to_create:
    try:
        if not os.path.exists(directory):
             os.makedirs(directory, exist_ok=True)
             logging.info(f"Created directory: {directory}")
             created_dirs_count += 1
        else:
             logging.info(f"Verified directory exists: {directory}")
    except OSError as e:
        logging.error(f"Error creating directory {directory}: {e}")
logging.info(f"\nFinished creating/verifying {created_dirs_count} directories.")


# --- Create __init__.py files within the NEW structure ---
logging.info("\nCreating __init__.py files...")
all_dirs_for_init = set()
# Add explicitly created dirs and their parents within the new project structure
for d in dirs_to_create:
    all_dirs_for_init.add(d)
    parent = os.path.dirname(d)
    # Add parents up to the new root directory
    while parent and parent != new_project_root and parent != os.path.dirname(new_project_root) :
         # Check if parent is within the new project root before adding
        if parent.startswith(new_project_root):
            all_dirs_for_init.add(parent)
        parent = os.path.dirname(parent)
# Add the base src directory within the new project
all_dirs_for_init.add(base_src_dir)

# Add other necessary base directories within the new structure that need __init__.py
# (Adjust this list based on your actual structure if different)
new_subdirs_in_src = [
    os.path.join(base_src_dir, "compliance"),
    os.path.join(base_src_dir, "compliance", "verification"),
    os.path.join(base_src_dir, "ner"),
    os.path.join(base_src_dir, "neuro"),
    os.path.join(base_src_dir, "orchestration"),
    os.path.join(base_src_dir, "symbolic"),
    os.path.join(base_src_dir, "utils"),
]
all_dirs_for_init.update(new_subdirs_in_src)
# Add the top-level tests directory within the new project
all_dirs_for_init.add(os.path.join(new_project_root, "tests"))


# Add __init__.py to all relevant directories within the NEW project
init_created_count = 0
for directory in sorted(list(all_dirs_for_init)):
    # Only process directories that actually exist
    if os.path.isdir(directory):
        init_path = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_path):
            try:
                with open(init_path, "w") as f:
                    f.write("# Automatically generated __init__.py\n")
                init_created_count += 1
                # logging.info(f"Created __init__.py in: {directory}")
            except OSError as e:
                logging.error(f"Error creating __init__.py in {directory}: {e}")

logging.info(f"\nFinished creating {init_created_count} new __init__.py files in '{new_project_root}'.")
logging.info("New project structure script finished.")
logging.info(f"\nNext steps:")
logging.info(f"1. Manually copy or move the relevant .py files from your original 'SocratesNS' project into the corresponding folders within '{new_project_root}'.")
logging.info("2. Update the import statements in your copied/moved Python files.")

