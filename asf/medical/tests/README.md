# ASF Medical Research Synthesizer Tests

This directory contains tests for the ASF Medical Research Synthesizer.

## Test Structure

The tests are organized into the following directories:

- `unit`: Unit tests for individual components
- `integration`: Integration tests for API endpoints and service interactions
- `performance`: Performance tests for measuring execution time and resource usage

## Running Tests

You can run the tests using the `scripts/run_tests.py` script:

```bash
# Run all tests
python -m asf.medical.scripts.run_tests

# Run unit tests only
python -m asf.medical.scripts.run_tests --unit

# Run integration tests only
python -m asf.medical.scripts.run_tests --integration

# Run tests with verbose output
python -m asf.medical.scripts.run_tests --verbose

# Run tests with coverage report
python -m asf.medical.scripts.run_tests --coverage
```

## Test Configuration

The test configuration is defined in the following files:

- `conftest.py`: Pytest fixtures and configuration
- `pytest.ini`: Pytest configuration

## Test Markers

The tests are marked with the following markers:

- `unit`: Unit tests
- `integration`: Integration tests
- `performance`: Performance tests
- `slow`: Slow tests
- `api`: API tests
- `service`: Service tests
- `model`: Model tests
- `repository`: Repository tests
- `cache`: Cache tests
- `database`: Database tests
- `async_test`: Asynchronous tests

## Test Coverage

The test coverage report is generated using pytest-cov. You can generate a coverage report by running:

```bash
python run_tests.py --coverage
```

This will generate a coverage report in the terminal and an HTML report in the `htmlcov` directory.

## Continuous Integration

The tests are run automatically on each pull request and push to the main branch using GitHub Actions.

## Writing Tests

### Unit Tests

Unit tests should test individual components in isolation. They should be fast and not depend on external services.

Example:

```python
@pytest.mark.unit
@pytest.mark.service
@pytest.mark.async_test
class TestBiasAssessmentService:
    """Test cases for BiasAssessmentService."""

    @pytest.mark.asyncio
    async def test_bias_assessment(self, bias_assessment_service: BiasAssessmentService, sample_study_text: str):
        """Test bias assessment with a sample text."""
        # Sample text with randomization but no blinding
        result = await bias_assessment_service.assess_study(sample_study_text)

        # Assertions
        assert result["randomization"]["risk"] == "low"
        assert result["blinding"]["risk"] == "high"
        assert "randomly assigned" in result["randomization"]["evidence"]
        assert "overall" in result
        assert result["overall"]["risk"] in ["low", "moderate", "high"]
```

### Integration Tests

Integration tests should test the interaction between components. They should test the API endpoints and service interactions.

Example:

```python
@pytest.mark.integration
@pytest.mark.api
class TestScreeningAPI:
    """Test cases for screening API."""

    def test_prisma_screening_endpoint(self):
        """Test PRISMA screening endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5,
            "stage": ScreeningStage.SCREENING,
            "criteria": {
                "include": ["randomized controlled trial", "cardiovascular outcomes"],
                "exclude": ["animal study", "in vitro"]
            }
        }

        # Make request
        response = client.post("/api/v1/screening/prisma", json=data)

        # Assertions
        assert response.status_code == 200
        assert "query" in response.json()
        assert "stage" in response.json()
        assert "total_articles" in response.json()
        assert "included" in response.json()
        assert "excluded" in response.json()
        assert "uncertain" in response.json()
        assert "results" in response.json()
        assert "flow_data" in response.json()
```

### Performance Tests

Performance tests should measure the execution time and resource usage of components.

Example:

```python
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.async_test
class TestServicePerformance:
    """Performance tests for services."""

    @pytest.mark.asyncio
    async def test_bias_assessment_performance(self, bias_assessment_service: BiasAssessmentService, sample_study_text: str):
        """Test performance of bias assessment."""
        # Warm-up
        await bias_assessment_service.assess_study(sample_study_text)

        # Measure performance
        start_time = time.time()
        iterations = 10

        for _ in range(iterations):
            await bias_assessment_service.assess_study(sample_study_text)

        elapsed_time = time.time() - start_time
        average_time = elapsed_time / iterations

        logger.info(f"Bias assessment performance: {average_time:.4f} seconds per assessment")

        # Assertion (adjust threshold as needed)
        assert average_time < 1.0, f"Bias assessment is too slow: {average_time:.4f} seconds per assessment"
```
