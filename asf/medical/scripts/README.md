# ASF Medical Codebase Cleanup Scripts

This directory contains scripts for cleaning up and standardizing the ASF Medical codebase.

## Available Scripts

### Master Cleanup

- **master_cleanup.py**: Runs all cleanup scripts in sequence.
  ```
  python -m asf.medical.scripts.master_cleanup [--dry-run] [--directory DIR] [--scripts SCRIPT1 SCRIPT2 ...]
  ```

### Service Naming Standardization

- **standardize_service_naming.py**: Standardizes service class names across the codebase.
  ```
  python asf\medical\scripts\standardize_service_naming.py <directory> [--fix]
  ```

### Import Standardization

- **standardize_imports.py**: Standardizes import statements for renamed services.
  ```
  python asf\medical\scripts\standardize_imports.py <directory> [--fix]
  ```

### Unused Imports Removal

- **fix_unused_imports.py**: Identifies and removes unused imports.
  ```
  python asf\medical\scripts\fix_unused_imports.py <directory> [--fix]
  ```

### Docstring Improvement

- **fix_docstrings.py**: Adds or improves docstrings in Python files.
  ```
  python asf\medical\scripts\fix_docstrings.py <directory> [--fix]
  ```

### Error Handling Standardization

- **standardize_error_handling.py**: Standardizes error handling patterns.
  ```
  python asf\medical\scripts\standardize_error_handling.py <directory> [--fix]
  ```

### Database Access Standardization

- **standardize_db_access.py**: Standardizes database access patterns.
  ```
  python asf\medical\scripts\standardize_db_access.py <directory> [--fix]
  ```

### Caching Standardization

- **standardize_caching.py**: Standardizes caching patterns.
  ```
  python asf\medical\scripts\standardize_caching.py <directory> [--fix]
  ```

### Deep Cleanup

- **deep_cleanup.py**: Performs deep cleanup of the codebase.
  ```
  python asf\medical\scripts\deep_cleanup.py <directory> [--fix]
  ```

## Standard Naming Conventions

### Service Classes

| Old Name | New Name |
|----------|----------|
| EnhancedUnifiedUnifiedContradictionService | ContradictionService |
| UnifiedUnifiedUnifiedContradictionService | ContradictionService |
| UnifiedUnifiedContradictionService | ContradictionService |
| UnifiedContradictionService | ContradictionService |
| EnhancedContradictionClassifier | ContradictionClassifierService |
| BiasAssessmentEngine | BiasAssessmentService |
| PRISMAScreeningEngine | PRISMAScreeningService |
| TemporalAnalysisEngine | TemporalService |
| ExplanationGeneratorEngine | ExplanationGeneratorService |
| ContradictionResolutionEngine | ContradictionResolutionService |
| MedicalContradictionResolutionEngine | MedicalContradictionResolutionService |

## Usage Examples

### Running All Cleanup Scripts

```bash
python -m asf.medical.scripts.master_cleanup
```

This will run all cleanup scripts on the asf/medical directory.

### Running Specific Cleanup Scripts

```bash
python -m asf.medical.scripts.master_cleanup --scripts asf.medical.scripts.fix_unused_imports asf.medical.scripts.fix_docstrings
```

### Dry Run Mode

```bash
python -m asf.medical.scripts.master_cleanup --dry-run
```

### Specifying a Directory

```bash
python -m asf.medical.scripts.master_cleanup --directory asf\medical\ml\services
```
