"""
Documentation Standards for ASF Medical Research Synthesizer.

This module defines the documentation standards for the ASF Medical Research Synthesizer
codebase. It includes templates and guidelines for documenting different components.

Functions:
    generate_class_template: Generate a class documentation template.
    generate_method_template: Generate a method documentation template.
    generate_module_template: Generate a module documentation template.
"""
# Class documentation template
CLASS_TEMPLATE = """
{class_name}

{description}

This class {functionality}.
{additional_info}
"""

# Method documentation template
METHOD_TEMPLATE = """
{method_name}

{description}

Args:
{args}

Returns:
{returns}

Raises:
{raises}
"""

# Module documentation template
MODULE_TEMPLATE = """
{module_name}

{description}

This module provides {functionality}.
"""

# Documentation guidelines
GUIDELINES = """
Documentation Guidelines:

1. Use Google-style docstrings for all code.
2. Document all public classes, methods, and functions.
3. Include Args, Returns, and Raises sections in all function/method docstrings.
4. Keep descriptions concise but informative.
5. Use proper grammar and punctuation.
6. Include examples for complex functionality.
7. Update documentation when code changes.
"""

def generate_class_template(class_name, description, functionality, additional_info=""):
    """
    Generate a class documentation template.

    Args:
        class_name (str): The name of the class.
        description (str): A brief description of the class.
        functionality (str): Description of what the class does.
        additional_info (str, optional): Any additional information about the class.

    Returns:
        str: Formatted class documentation template.
    """
    return CLASS_TEMPLATE.format(
        class_name=class_name,
        description=description,
        functionality=functionality,
        additional_info=additional_info
    )

def generate_method_template(method_name, description, args, returns, raises):
    """
    Generate a method documentation template.

    Args:
        method_name (str): The name of the method.
        description (str): A brief description of the method.
        args (str): Arguments of the method in formatted string.
        returns (str): Description of the return value.
        raises (str): Description of exceptions raised by the method.

    Returns:
        str: Formatted method documentation template.
    """
    return METHOD_TEMPLATE.format(
        method_name=method_name,
        description=description,
        args=args,
        returns=returns,
        raises=raises
    )

def generate_module_template(module_name, description, functionality):
    """
    Generate a module documentation template.

    Args:
        module_name (str): The name of the module.
        description (str): A brief description of the module.
        functionality (str): Description of what the module provides.

    Returns:
        str: Formatted module documentation template.
    """
    return MODULE_TEMPLATE.format(
        module_name=module_name,
        description=description,
        functionality=functionality
    )
