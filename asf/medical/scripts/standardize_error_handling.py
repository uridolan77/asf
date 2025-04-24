"""
Standardize Error Handling in ASF Medical Codebase.

This script identifies and fixes inconsistent error handling patterns in the codebase,
implementing a standardized approach with proper logging and custom exceptions. It
detects issues such as bare except blocks, generic exception handling without custom
exceptions, missing logging, and missing custom exceptions.

The script applies fixes based on the context of the code, determining the appropriate
type of exception to raise (DatabaseError, APIError, ValidationError, MLError, or
OperationError) and ensuring proper logging is in place.

Usage:
    python -m asf.medical.scripts.standardize_error_handling <directory> [--fix]

Options:
    <directory>  Directory to process
    --fix        Fix the issues (otherwise just report them)

The script excludes certain directories and files from processing to avoid modifying
generated code, third-party libraries, and configuration files.
"""
import os
import re
import sys
import logging
from typing import List, Dict, Any
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("StandardizeErrorHandling")
# Directories to exclude from processing
EXCLUDE_DIRS = [
    "__pycache__",
    ".git",
    ".vscode",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build"
]
# Files to exclude from processing
EXCLUDE_FILES = [
    "__init__.py",
    "conftest.py",
    "standardize_error_handling.py",
    "deep_cleanup_phase2.py",
    "fix_unused_imports.py",
    "fix_docstrings.py"
]
# Patterns for finding inconsistent error handling
BARE_EXCEPT_PATTERN = r"except\s*:"
GENERIC_EXCEPT_PATTERN = r"except\s+Exception\s+as\s+e:"
MISSING_LOGGING_PATTERN = r"except.*:\s*\n\s*(?!.*logger\.(error|warning|exception))"
MISSING_CUSTOM_EXCEPTION_PATTERN = r"except.*:\s*\n\s*logger\.(error|warning|exception).*\n\s*(?!.*raise\s+\w+Error)"
# Standard error handling templates
STANDARD_ERROR_HANDLING = {
    "database": """try:
    # Database operation
except Exception as e:
    logger.error(f"Database error: {str(e)}")
    raise DatabaseError(f"Database operation failed: {str(e)}")""",
    "api": """try:
    # API call
except Exception as e:
    logger.error(f"API error: {str(e)}")
    raise APIError(f"API call failed: {str(e)}")""",
    "validation": """try:
    # Validation
except Exception as e:
    logger.error(f"Validation error: {str(e)}")
    raise ValidationError(f"Validation failed: {str(e)}")""",
    "ml": """try:
    # ML operation
except Exception as e:
    logger.error(f"ML error: {str(e)}")
    raise MLError(f"ML operation failed: {str(e)}")""",
    "general": """try:
    # Operation
except Exception as e:
    logger.error(f"Error: {str(e)}")
    raise OperationError(f"Operation failed: {str(e)}")"""
}
def find_python_files(directory: str) -> List[str]:
    """
    Find all Python files in the given directory and its subdirectories.

    This function recursively walks through the specified directory and collects
    paths to all Python files (.py extension) that are not in excluded directories
    and are not excluded files.

    Args:
        directory: The directory to search in.

    Returns:
        List of paths to Python files that should be processed.
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(".py") and file not in EXCLUDE_FILES:
                python_files.append(os.path.join(root, file))
    return python_files
def find_inconsistent_error_handling(content: str) -> List[Dict[str, Any]]:
    """
    Find inconsistent error handling in the content.

    This function analyzes the content of a Python file to identify inconsistent
    error handling patterns, including bare except blocks, generic exception handling
    without custom exceptions, missing logging, and missing custom exceptions.

    Args:
        content: The content of the Python file to analyze.

    Returns:
        List of dictionaries containing information about the identified issues,
        including the type of issue, line number, matching text, and context.
    """
    issues = []
    # Find bare except statements
    bare_excepts = []
    for match in re.finditer(BARE_EXCEPT_PATTERN, content):
        line_num = content[:match.start()].count("\n") + 1
        bare_excepts.append({
            "type": "bare_except",
            "line": line_num,
            "match": match.group(0),
            "context": get_context(content, match.start(), 3)
        })
    # Find generic except statements without custom exceptions
    generic_excepts = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if re.search(GENERIC_EXCEPT_PATTERN, line):
            # Check if there's a custom exception raised
            has_custom_exception = False
            for j in range(i+1, min(i+5, len(lines))):
                if re.search(r"raise\s+\w+Error", lines[j]):
                    has_custom_exception = True
                    break
            if not has_custom_exception:
                generic_excepts.append({
                    "type": "generic_except",
                    "line": i + 1,
                    "match": line,
                    "context": get_context(content, sum(len(l) + 1 for l in lines[:i]), 3)
                })
    # Find missing logging
    missing_logging = []
    for match in re.finditer(MISSING_LOGGING_PATTERN, content, re.MULTILINE):
        line_num = content[:match.start()].count("\n") + 1
        missing_logging.append({
            "type": "missing_logging",
            "line": line_num,
            "match": match.group(0),
            "context": get_context(content, match.start(), 3)
        })
    # Find missing custom exceptions
    missing_custom_exceptions = []
    for match in re.finditer(MISSING_CUSTOM_EXCEPTION_PATTERN, content, re.MULTILINE):
        line_num = content[:match.start()].count("\n") + 1
        missing_custom_exceptions.append({
            "type": "missing_custom_exception",
            "line": line_num,
            "match": match.group(0),
            "context": get_context(content, match.start(), 3)
        })
    issues.extend(bare_excepts)
    issues.extend(generic_excepts)
    issues.extend(missing_logging)
    issues.extend(missing_custom_exceptions)
    return issues
def get_context(content: str, pos: int, lines: int) -> str:
    """
    Get context around a position in the content.

    This function extracts a specified number of lines before and after a given
    position in the content, providing context for the identified issues.

    Args:
        content: The content of the Python file.
        pos: The position (character index) in the content.
        lines: The number of lines of context to include before and after.

    Returns:
        A string containing the specified number of lines before and after the position.
    """
    lines_list = content.split("\n")
    line_num = content[:pos].count("\n")
    start_line = max(0, line_num - lines)
    end_line = min(len(lines_list), line_num + lines + 1)
    return "\n".join(lines_list[start_line:end_line])
def fix_error_handling(content: str, issues: List[Dict[str, Any]]) -> str:
    """
    Fix inconsistent error handling in the content.

    This function applies fixes to the identified error handling issues in the content.
    It processes issues in reverse order of line number to avoid changing line numbers
    as it makes modifications. The fixes include:
    - Replacing bare except with except Exception as e
    - Adding appropriate logging statements
    - Adding custom exception raising based on the context

    Args:
        content: The content of the Python file to fix.
        issues: List of dictionaries containing information about the identified issues.

    Returns:
        The updated content with fixed error handling.
    """
    if not issues:
        return content
    lines = content.split("\n")
    # Process issues in reverse order to avoid changing line numbers
    for issue in sorted(issues, key=lambda x: x["line"], reverse=True):
        line_num = issue["line"] - 1  # Convert to 0-based index
        if issue["type"] == "bare_except":
            # Replace bare except with except Exception as e
            lines[line_num] = lines[line_num].replace("except:", "except Exception as e:")
            # Add logging if not present
            has_logging = False
            for i in range(line_num + 1, min(line_num + 5, len(lines))):
                if "logger." in lines[i]:
                    has_logging = True
                    break
            if not has_logging:
                indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                indent_str = " " * indent
                lines.insert(line_num + 1, f"{indent_str}    logger.error(f\"Error: {{str(e)}}\")")
        elif issue["type"] == "generic_except":
            # Find the try block
            try_line = line_num
            while try_line >= 0 and not lines[try_line].strip().startswith("try:"):
                try_line -= 1
            if try_line >= 0:
                # Determine the type of operation
                operation_type = "general"
                for i in range(try_line + 1, line_num):
                    line = lines[i].lower()
                    if "database" in line or "db" in line or "sql" in line:
                        operation_type = "database"
                        break
                    elif "api" in line or "request" in line or "http" in line:
                        operation_type = "api"
                        break
                    elif "valid" in line or "schema" in line or "check" in line:
                        operation_type = "validation"
                        break
                    elif "model" in line or "predict" in line or "ml" in line:
                        operation_type = "ml"
                        break
                # Add custom exception
                indent = len(lines[line_num]) - len(lines[line_num].lstrip())
                indent_str = " " * indent
                # Check if there's already logging
                has_logging = False
                has_exception = False
                for i in range(line_num + 1, min(line_num + 5, len(lines))):
                    if "logger." in lines[i]:
                        has_logging = True
                    if "raise" in lines[i]:
                        has_exception = True
                if not has_logging:
                    if operation_type == "database":
                        lines.insert(line_num + 1, f"{indent_str}    logger.error(f\"Database error: {{str(e)}}\")")
                    elif operation_type == "api":
                        lines.insert(line_num + 1, f"{indent_str}    logger.error(f\"API error: {{str(e)}}\")")
                    elif operation_type == "validation":
                        lines.insert(line_num + 1, f"{indent_str}    logger.error(f\"Validation error: {{str(e)}}\")")
                    elif operation_type == "ml":
                        lines.insert(line_num + 1, f"{indent_str}    logger.error(f\"ML error: {{str(e)}}\")")
                    else:
                        lines.insert(line_num + 1, f"{indent_str}    logger.error(f\"Error: {{str(e)}}\")")
                # Add custom exception if not present
                if not has_exception:
                    if operation_type == "database":
                        lines.insert(line_num + 2, f"{indent_str}    raise DatabaseError(f\"Database operation failed: {{str(e)}}\")")
                    elif operation_type == "api":
                        lines.insert(line_num + 2, f"{indent_str}    raise APIError(f\"API call failed: {{str(e)}}\")")
                    elif operation_type == "validation":
                        lines.insert(line_num + 2, f"{indent_str}    raise ValidationError(f\"Validation failed: {{str(e)}}\")")
                    elif operation_type == "ml":
                        lines.insert(line_num + 2, f"{indent_str}    raise MLError(f\"ML operation failed: {{str(e)}}\")")
                    else:
                        lines.insert(line_num + 2, f"{indent_str}    raise OperationError(f\"Operation failed: {{str(e)}}\")")
        elif issue["type"] == "missing_logging":
            # Add logging
            indent = len(lines[line_num]) - len(lines[line_num].lstrip())
            indent_str = " " * indent
            lines.insert(line_num + 1, f"{indent_str}    logger.error(f\"Error: {{str(e)}}\")")
        elif issue["type"] == "missing_custom_exception":
            # Add custom exception
            indent = len(lines[line_num]) - len(lines[line_num].lstrip())
            indent_str = " " * indent
            # Determine the type of operation
            operation_type = "general"
            for i in range(max(0, line_num - 5), line_num):
                line = lines[i].lower()
                if "database" in line or "db" in line or "sql" in line:
                    operation_type = "database"
                    break
                elif "api" in line or "request" in line or "http" in line:
                    operation_type = "api"
                    break
                elif "valid" in line or "schema" in line or "check" in line:
                    operation_type = "validation"
                    break
                elif "model" in line or "predict" in line or "ml" in line:
                    operation_type = "ml"
                    break
            # Check if there's already an exception
            has_exception = False
            for i in range(line_num + 1, min(line_num + 5, len(lines))):
                if "raise" in lines[i]:
                    has_exception = True
                    break
            # Add custom exception if not present
            if not has_exception:
                if operation_type == "database":
                    lines.insert(line_num + 2, f"{indent_str}    raise DatabaseError(f\"Database operation failed: {{str(e)}}\")")
                elif operation_type == "api":
                    lines.insert(line_num + 2, f"{indent_str}    raise APIError(f\"API call failed: {{str(e)}}\")")
                elif operation_type == "validation":
                    lines.insert(line_num + 2, f"{indent_str}    raise ValidationError(f\"Validation failed: {{str(e)}}\")")
                elif operation_type == "ml":
                    lines.insert(line_num + 2, f"{indent_str}    raise MLError(f\"ML operation failed: {{str(e)}}\")")
                else:
                    lines.insert(line_num + 2, f"{indent_str}    raise OperationError(f\"Operation failed: {{str(e)}}\")")
    return "\n".join(lines)
def ensure_custom_exceptions(content: str) -> str:
    """
    Ensure custom exceptions are defined in the file.

    This function checks if the custom exceptions used in the file are properly
    imported from the asf.medical.core.exceptions module. If not, it adds the
    necessary import statement at an appropriate location in the file.

    Args:
        content: The content of the Python file to check and update.

    Returns:
        The updated content with the necessary import statements added.
    """
    # Check if custom exceptions are already imported
    if "from asf.medical.core.exceptions import" in content:
        return content
    # Check which custom exceptions are used
    used_exceptions = set()
    if "DatabaseError" in content:
        used_exceptions.add("DatabaseError")
    if "APIError" in content:
        used_exceptions.add("APIError")
    if "ValidationError" in content:
        used_exceptions.add("ValidationError")
    if "MLError" in content:
        used_exceptions.add("MLError")
    if "OperationError" in content:
        used_exceptions.add("OperationError")
    if not used_exceptions:
        return content
    # Add import statement
    import_statement = f"from asf.medical.core.exceptions import {', '.join(sorted(used_exceptions))}\n"
    # Find the right place to add the import
    lines = content.split("\n")
    import_index = 0
    # Find the last import statement
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            import_index = i + 1
    # Add a blank line after imports if needed
    if import_index > 0 and import_index < len(lines) and lines[import_index].strip():
        lines.insert(import_index, "")
        import_index += 1
    # Add the import statement
    lines.insert(import_index, import_statement)
    return "\n".join(lines)
def process_file(file_path: str, fix: bool = False) -> Dict[str, Any]:
    """
    Process a single file to find and optionally fix inconsistent error handling.

    This function reads a Python file, analyzes it for inconsistent error handling,
    and optionally applies fixes. It handles exceptions gracefully to ensure the
    script continues running even if one file has issues.

    Args:
        file_path: Path to the Python file to process.
        fix: If True, modify the file to fix error handling issues; if False,
            just report the issues.

    Returns:
        A dictionary with the following keys:
        - 'issues': List of identified error handling issues
        - 'fixed': Boolean indicating whether the file was modified
    """
    results = {
        "issues": [],
        "fixed": False
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Find issues
        issues = find_inconsistent_error_handling(content)
        results["issues"] = issues
        # Fix issues if requested
        if fix and issues:
            updated_content = fix_error_handling(content, issues)
            # Ensure custom exceptions are defined
            updated_content = ensure_custom_exceptions(updated_content)
            # Write updated content if changes were made
            if updated_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                results["fixed"] = True
        return results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return results
def main():
    """
    Main function to run the error handling standardization script.

    This function parses command-line arguments, finds Python files in the specified
    directory, processes each file to identify and optionally fix error handling
    issues, and reports a summary of the results.

    Command-line arguments:
        <directory>: Directory to process
        --fix: Optional flag to actually fix the issues

    Returns:
        None, but exits with a non-zero code if there's an error
    """
    if len(sys.argv) < 2:
        print("Usage: python standardize_error_handling.py <directory> [--fix]")
        sys.exit(1)
    directory = sys.argv[1]
    fix = "--fix" in sys.argv
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory")
        sys.exit(1)
    logger.info(f"Starting standardize error handling for {directory}")
    logger.info(f"Fix mode: {fix}")
    python_files = find_python_files(directory)
    logger.info(f"Found {len(python_files)} Python files")
    total_issues = 0
    fixed_files = 0
    for file_path in python_files:
        logger.info(f"Processing {file_path}")
        results = process_file(file_path, fix)
        # Count issues
        total_issues += len(results["issues"])
        if results["fixed"]:
            fixed_files += 1
            logger.info(f"Fixed error handling in {file_path}")
        # Report issues
        for issue in results["issues"]:
            logger.info(f"{file_path}:{issue['line']} - {issue['type']}")
    # Report summary
    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"Total Python files processed: {len(python_files)}")
    logger.info(f"Total error handling issues: {total_issues}")
    if fix:
        logger.info(f"Fixed files: {fixed_files}")
    logger.info("=" * 50)
if __name__ == "__main__":
    main()