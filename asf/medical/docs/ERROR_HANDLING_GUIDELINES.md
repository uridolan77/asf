# Error Handling Guidelines for Medical Research Synthesizer

This document outlines the error handling standards for the Medical Research Synthesizer codebase. Following these guidelines ensures consistent, robust error handling throughout the application.

## Exception Hierarchy

The Medical Research Synthesizer uses a custom exception hierarchy defined in `asf/medical/core/exceptions.py`. All custom exceptions inherit from the base `MedicalResearchSynthesizerError` class.

### Key Exception Types

- **ValidationError**: For input validation failures
- **DatabaseError**: For database-related errors
- **AuthenticationError**: For authentication failures
- **AuthorizationError**: For permission-related errors
- **APIError**: For external API call failures
- **CacheError**: For caching-related errors
- **TaskError**: For background task failures
- **ServiceError**: For service-level errors
- **ResourceError**: For resource-related errors
- **ConfigurationError**: For configuration-related errors
- **MLError**: For machine learning-related errors
- **ExportError**: For export-related errors
- **SearchError**: For search-related errors
- **AnalysisError**: For analysis-related errors
- **KnowledgeBaseError**: For knowledge base-related errors
- **ContradictionError**: For contradiction detection errors
- **OperationError**: For general operation failures

## Error Handling Patterns

### Basic Pattern

All error handling should follow this basic pattern:

```python
try:
    # Operation that might fail
    result = perform_operation()
except SpecificError as e:
    # Log the error
    logger.error(f"Operation failed: {str(e)}")
    # Raise an appropriate custom exception
    raise CustomError(f"Operation failed: {str(e)}") from e
```

### Domain-Specific Patterns

#### Database Operations

```python
try:
    # Database operation
    result = await db.execute(query)
except Exception as e:
    logger.error(f"Database error: {str(e)}")
    raise DatabaseError(f"Database operation failed: {str(e)}", 
                       operation="execute_query", 
                       model="User") from e
```

#### External API Calls

```python
try:
    # API call
    response = await client.get(url)
    if response.status_code >= 400:
        raise ValueError(f"API returned error: {response.status_code}")
    return response.json()
except Exception as e:
    logger.error(f"API error: {str(e)}")
    raise APIError(f"API call failed: {str(e)}", 
                  api="ExternalService", 
                  status_code=getattr(response, "status_code", None),
                  endpoint=url) from e
```

#### Validation

```python
try:
    # Validation
    if not is_valid(data):
        raise ValueError("Invalid data")
    return process_data(data)
except Exception as e:
    logger.error(f"Validation error: {str(e)}")
    raise ValidationError(f"Validation failed: {str(e)}") from e
```

#### Machine Learning Operations

```python
try:
    # ML operation
    result = model.predict(input_data)
    return result
except Exception as e:
    logger.error(f"ML error: {str(e)}")
    raise MLError(f"ML operation failed: {str(e)}", 
                 model="BioMedLM", 
                 operation="predict") from e
```

## Logging Guidelines

### Log Levels

- **ERROR**: For errors that prevent an operation from completing successfully
- **WARNING**: For issues that don't prevent an operation but might indicate a problem
- **INFO**: For normal operational information
- **DEBUG**: For detailed debugging information

### What to Log

When logging errors, include:

1. A descriptive message about what failed
2. The exception message
3. Relevant context (e.g., operation, input parameters)
4. For sensitive operations, ensure no PII/PHI is logged

Example:
```python
logger.error(
    f"Failed to process search query",
    extra={
        "query_id": query_id,
        "error": str(e),
        "operation": "process_search"
    }
)
```

## Exception Propagation

### API Layer

In the API layer, use the exception middleware to convert custom exceptions to appropriate HTTP responses:

```python
@router.get("/items/{item_id}")
async def get_item(item_id: str):
    try:
        return await service.get_item(item_id)
    except NotFoundError as e:
        # Let the middleware handle this
        raise
    except ValidationError as e:
        # Let the middleware handle this
        raise
    except Exception as e:
        # Unexpected error, log and convert to a custom exception
        logger.error(f"Unexpected error in get_item: {str(e)}")
        raise ServiceError(f"Failed to get item: {str(e)}") from e
```

### Service Layer

In the service layer, catch specific exceptions and convert them to appropriate custom exceptions:

```python
async def get_item(self, item_id: str):
    try:
        return await self.repository.get_item(item_id)
    except DatabaseError as e:
        # Already a custom exception, just re-raise
        raise
    except Exception as e:
        # Convert to a service error
        logger.error(f"Service error in get_item: {str(e)}")
        raise ServiceError(f"Failed to get item: {str(e)}") from e
```

### Repository Layer

In the repository layer, catch database-specific exceptions and convert them to `DatabaseError`:

```python
async def get_item(self, item_id: str):
    try:
        query = select(Item).where(Item.id == item_id)
        result = await self.session.execute(query)
        item = result.scalar_one_or_none()
        if not item:
            raise NotFoundError(f"Item with ID {item_id} not found")
        return item
    except NoResultFound as e:
        raise NotFoundError(f"Item with ID {item_id} not found") from e
    except SQLAlchemyError as e:
        logger.error(f"Database error in get_item: {str(e)}")
        raise DatabaseError(f"Failed to get item: {str(e)}", 
                           operation="get_item", 
                           model="Item") from e
```

## Background Tasks

For background tasks, ensure errors are properly logged and tracked:

```python
@dramatiq.actor
def process_task(task_id: str, data: Dict[str, Any]):
    try:
        # Process the task
        result = process_data(data)
        # Update task status
        update_task_status(task_id, "completed", result=result)
    except Exception as e:
        logger.error(f"Task error: {str(e)}")
        # Update task status with error
        update_task_status(task_id, "failed", error=str(e))
        # Optionally re-raise to trigger retry
        raise
```

## Testing Error Handling

Always include tests for error scenarios:

```python
@pytest.mark.asyncio
async def test_get_item_not_found():
    # Arrange
    repository = MockRepository()
    repository.get_item.side_effect = NotFoundError("Item not found")
    service = ItemService(repository)
    
    # Act & Assert
    with pytest.raises(NotFoundError):
        await service.get_item("non-existent-id")
```

## Common Mistakes to Avoid

1. **Bare except clauses**: Always specify the exception type(s) to catch
   ```python
   # Bad
   try:
       result = operation()
   except:  # Catches everything, including KeyboardInterrupt!
       logger.error("Error")
   
   # Good
   try:
       result = operation()
   except Exception as e:
       logger.error(f"Error: {str(e)}")
   ```

2. **Catching exceptions without logging**: Always log exceptions
   ```python
   # Bad
   try:
       result = operation()
   except Exception as e:
       raise CustomError("Failed") from e
   
   # Good
   try:
       result = operation()
   except Exception as e:
       logger.error(f"Operation failed: {str(e)}")
       raise CustomError("Failed") from e
   ```

3. **Swallowing exceptions**: Don't catch exceptions without handling them properly
   ```python
   # Bad
   try:
       result = operation()
   except Exception:
       pass  # Silently ignores the error
   
   # Good
   try:
       result = operation()
   except Exception as e:
       logger.error(f"Operation failed: {str(e)}")
       # Either handle the error or re-raise an appropriate exception
   ```

4. **Not using custom exceptions**: Use the custom exception hierarchy
   ```python
   # Bad
   raise Exception("Database error")
   
   # Good
   raise DatabaseError("Failed to query database")
   ```

5. **Not preserving the original exception**: Use `from e` to preserve the original exception
   ```python
   # Bad
   try:
       result = operation()
   except Exception as e:
       raise CustomError("Failed")
   
   # Good
   try:
       result = operation()
   except Exception as e:
       raise CustomError("Failed") from e
   ```
