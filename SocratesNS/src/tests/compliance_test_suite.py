import unittest
import json
import os
import pytest
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import hypothesis
from hypothesis import given, strategies as st


@dataclass
class ComplianceTestCase:
    """Test case for compliance verification"""
    id: str
    content: str
    content_type: str
    frameworks: List[str]
    expected_compliance: bool
    expected_violations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""


class ComplianceTestSuite:
    """
    Comprehensive test suite for compliance verification systems.
    """
    
    def __init__(self, compliance_system, test_cases_dir="test_cases"):
        """
        Initialize the compliance test suite
        
        Args:
            compliance_system: The compliance system to test
            test_cases_dir: Directory containing test case JSON files
        """
        self.compliance_system = compliance_system
        self.test_cases_dir = test_cases_dir
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[ComplianceTestCase]:
        """Load test cases from JSON files"""
        test_cases = []
        
        if not os.path.exists(self.test_cases_dir):
            print(f"Warning: Test cases directory {self.test_cases_dir} not found")
            return test_cases
            
        for filename in os.listdir(self.test_cases_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.test_cases_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        cases_data = json.load(f)
                        
                    for case_data in cases_data:
                        test_case = ComplianceTestCase(
                            id=case_data["id"],
                            content=case_data["content"],
                            content_type=case_data.get("content_type", "text"),
                            frameworks=case_data.get("frameworks", []),
                            expected_compliance=case_data["expected_compliance"],
                            expected_violations=case_data.get("expected_violations", []),
                            tags=case_data.get("tags", []),
                            description=case_data.get("description", "")
                        )
                        test_cases.append(test_case)
                except Exception as e:
                    print(f"Error loading test cases from {file_path}: {e}")
                    
        return test_cases
    
    def run_tests(self, framework_filter=None, tag_filter=None) -> Dict[str, Any]:
        """
        Run all compliance tests
        
        Args:
            framework_filter: Only run tests for specific frameworks
            tag_filter: Only run tests with specific tags
            
        Returns:
            Test results summary
        """
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "cases": []
        }
        
        filtered_cases = self._filter_test_cases(framework_filter, tag_filter)
        print(f"Running {len(filtered_cases)} compliance tests...")
        
        for case in filtered_cases:
            results["total_tests"] += 1
            case_result = {
                "id": case.id,
                "description": case.description,
                "status": "passed"
            }
            
            try:
                # Run the test case
                verification_result = self.compliance_system.verify_content(
                    case.content,
                    content_type=case.content_type,
                    frameworks=case.frameworks
                )
                
                # Check compliance status
                actual_compliance = verification_result.get("is_compliant", False)
                if actual_compliance != case.expected_compliance:
                    case_result["status"] = "failed"
                    case_result["reason"] = f"Expected compliance: {case.expected_compliance}, actual: {actual_compliance}"
                    results["failed"] += 1
                else:
                    # Check expected violations if compliant is False
                    if not case.expected_compliance and case.expected_violations:
                        actual_violations = []
                        for violation in verification_result.get("violations", []):
                            if "rule_id" in violation:
                                actual_violations.append(violation["rule_id"])
                                
                        # Check if all expected violations are present
                        missing_violations = [v for v in case.expected_violations if v not in actual_violations]
                        if missing_violations:
                            case_result["status"] = "failed"
                            case_result["reason"] = f"Missing expected violations: {missing_violations}"
                            results["failed"] += 1
                        else:
                            results["passed"] += 1
                    else:
                        results["passed"] += 1
                        
            except Exception as e:
                case_result["status"] = "error"
                case_result["reason"] = str(e)
                results["errors"] += 1
                
            results["cases"].append(case_result)
            
        return results
    
    def _filter_test_cases(self, framework_filter=None, tag_filter=None) -> List[ComplianceTestCase]:
        """Filter test cases by framework and/or tags"""
        if not framework_filter and not tag_filter:
            return self.test_cases
            
        filtered_cases = []
        
        for case in self.test_cases:
            # Filter by framework
            if framework_filter and not any(f in case.frameworks for f in framework_filter):
                continue
                
            # Filter by tag
            if tag_filter and not any(t in case.tags for t in tag_filter):
                continue
                
            filtered_cases.append(case)
            
        return filtered_cases
    
    def generate_report(self, results: Dict[str, Any], output_file="compliance_test_report.html") -> str:
        """
        Generate an HTML report of test results
        
        Args:
            results: Test results from run_tests
            output_file: Path to output HTML file
            
        Returns:
            Path to the generated report
        """
        # Create basic HTML report
        html = ["<!DOCTYPE html>", "<html>", "<head>"]
        html.append("<title>Compliance Test Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("table { border-collapse: collapse; width: 100%; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append(".passed { color: green; }")
        html.append(".failed { color: red; }")
        html.append(".error { color: orange; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Add summary
        html.append("<h1>Compliance Test Results</h1>")
        html.append("<div class='summary'>")
        html.append(f"<p>Total Tests: {results['total_tests']}</p>")
        html.append(f"<p>Passed: <span class='passed'>{results['passed']}</span></p>")
        html.append(f"<p>Failed: <span class='failed'>{results['failed']}</span></p>")
        html.append(f"<p>Errors: <span class='error'>{results['errors']}</span></p>")
        html.append("</div>")
        
        # Add detailed results
        html.append("<h2>Test Case Results</h2>")
        html.append("<table>")
        html.append("<tr><th>ID</th><th>Description</th><th>Status</th><th>Reason</th></tr>")
        
        for case in results["cases"]:
            status_class = "passed" if case["status"] == "passed" else "failed" if case["status"] == "failed" else "error"
            html.append("<tr>")
            html.append(f"<td>{case['id']}</td>")
            html.append(f"<td>{case['description']}</td>")
            html.append(f"<td class='{status_class}'>{case['status']}</td>")
            html.append(f"<td>{case.get('reason', '')}</td>")
            html.append("</tr>")
            
        html.append("</table>")
        html.append("</body>")
        html.append("</html>")
        
        # Save report
        with open(output_file, 'w') as f:
            f.write("\n".join(html))
            
        print(f"Test report saved to {output_file}")
        return output_file


class ComplianceUnitTests(unittest.TestCase):
    """Unit tests for compliance components"""
    
    def setUp(self):
        # Initialize components for testing
        pass
    
    def test_verify_text_returns_valid_result(self):
        """Test that verify_content returns a valid result structure"""
        verifier = self.create_test_verifier()
        result = verifier.verify_content("Test content")
        
        # Check result has expected fields
        self.assertIn("is_compliant", result)
        self.assertIn("compliance_score", result)
        self.assertIsInstance(result["is_compliant"], bool)
        self.assertIsInstance(result["compliance_score"], (int, float))
    
    def test_verify_text_identifies_violations(self):
        """Test that verify_content identifies violations correctly"""
        verifier = self.create_test_verifier()
        test_content = "This content contains PII like SSN 123-45-6789"
        result = verifier.verify_content(test_content)
        
        self.assertFalse(result["is_compliant"])
        self.assertIn("violations", result)
        self.assertGreater(len(result["violations"]), 0)
    
    def test_verify_text_handles_empty_content(self):
        """Test that verify_content handles empty content gracefully"""
        verifier = self.create_test_verifier()
        result = verifier.verify_content("")
        
        self.assertTrue(result["is_compliant"])
    
    def create_test_verifier(self):
        """Create a test instance of the verifier"""
        # This should be implemented to create a test instance
        # For this example, we'll just return a mock
        class MockVerifier:
            def verify_content(self, content, content_type="text", frameworks=None):
                has_pii = "SSN" in content
                return {
                    "is_compliant": not has_pii,
                    "compliance_score": 0.5 if has_pii else 1.0,
                    "violations": [{"rule_id": "PII_SSN", "description": "SSN found"}] if has_pii else []
                }
        return MockVerifier()


@pytest.fixture
def compliance_verifier():
    """Pytest fixture for compliance verifier"""
    # Create and return a test instance
    class TestVerifier:
        def verify_content(self, content, content_type="text", frameworks=None):
            has_pii = "SSN" in content or "passport" in content
            return {
                "is_compliant": not has_pii,
                "compliance_score": 0.5 if has_pii else 1.0,
                "violations": [{"rule_id": "PII_DETECTION", "description": "PII found"}] if has_pii else []
            }
    return TestVerifier()


def test_verify_content_with_pii(compliance_verifier):
    """Test verification with PII content"""
    result = compliance_verifier.verify_content("My SSN is 123-45-6789")
    assert not result["is_compliant"]
    assert len(result["violations"]) > 0


def test_verify_content_without_pii(compliance_verifier):
    """Test verification without PII content"""
    result = compliance_verifier.verify_content("This is safe content")
    assert result["is_compliant"]
    assert len(result.get("violations", [])) == 0


# Property-based testing examples
@given(st.text())
def test_verification_doesnt_crash(compliance_verifier, content):
    """Test that verification doesn't crash with arbitrary text"""
    try:
        result = compliance_verifier.verify_content(content)
        assert isinstance(result, dict)
        assert "is_compliant" in result
    except Exception as e:
        pytest.fail(f"Verification raised exception: {e}")


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz "))
def test_clean_text_is_compliant(compliance_verifier, content):
    """Test that clean text is always compliant"""
    # Property: Text without any PII markers should be compliant
    result = compliance_verifier.verify_content(content)
    assert result["is_compliant"]


if __name__ == "__main__":
    # Run unit tests
    unittest.main()