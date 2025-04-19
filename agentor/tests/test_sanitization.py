import pytest
from agentor.llm_gateway.utils.sanitization import sanitize_html, sanitize_input, sanitize_output


def test_sanitize_html():
    """Test the sanitize_html function."""
    # Test with HTML tags
    html = "<script>alert('XSS');</script><p>Hello, world!</p>"
    sanitized = sanitize_html(html)
    assert "<script>" not in sanitized
    assert "alert" not in sanitized
    assert "<p>Hello, world!</p>" in sanitized
    
    # Test with style tags
    html = "<style>body { color: red; }</style><p>Hello, world!</p>"
    sanitized = sanitize_html(html)
    assert "<style>" not in sanitized
    assert "color: red" not in sanitized
    assert "<p>Hello, world!</p>" in sanitized
    
    # Test with inline styles
    html = "<p style='color: red;'>Hello, world!</p>"
    sanitized = sanitize_html(html)
    assert "style=" not in sanitized
    assert "<p>Hello, world!</p>" in sanitized
    
    # Test with meta tags
    html = "<meta http-equiv='refresh' content='0;url=http://evil.com'><p>Hello, world!</p>"
    sanitized = sanitize_html(html)
    assert "<meta" not in sanitized
    assert "evil.com" not in sanitized
    assert "<p>Hello, world!</p>" in sanitized
    
    # Test with links
    html = "<a href='http://evil.com'>Click me</a><p>Hello, world!</p>"
    sanitized = sanitize_html(html)
    assert "href=" not in sanitized
    assert "evil.com" not in sanitized
    assert ">Click me</a>" in sanitized
    assert "<p>Hello, world!</p>" in sanitized


def test_sanitize_input():
    """Test the sanitize_input function."""
    # Test with HTML tags
    input_text = "<script>alert('XSS');</script><p>Hello, world!</p>"
    sanitized = sanitize_input(input_text)
    assert "<script>" not in sanitized
    assert "alert" not in sanitized
    assert "Hello, world!" in sanitized
    
    # Test with control characters
    input_text = "Hello,\x00 world!\x1F"
    sanitized = sanitize_input(input_text)
    assert sanitized == "Hello, world!"
    
    # Test with long input
    input_text = "a" * 20000
    sanitized = sanitize_input(input_text)
    assert len(sanitized) == 10000
    assert sanitized == "a" * 10000


def test_sanitize_output():
    """Test the sanitize_output function."""
    # Test with HTML tags
    output_text = "<script>alert('XSS');</script><p>Hello, world!</p>"
    sanitized = sanitize_output(output_text)
    assert "<script>" not in sanitized
    assert "alert" not in sanitized
    assert "Hello, world!" in sanitized
    
    # Test with control characters
    output_text = "Hello,\x00 world!\x1F"
    sanitized = sanitize_output(output_text)
    assert sanitized == "Hello, world!"
