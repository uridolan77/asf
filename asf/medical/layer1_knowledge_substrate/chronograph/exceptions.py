"""Exceptions for the ChronoGraph middleware layer.

This module defines custom exceptions used throughout the ChronoGraph middleware
to provide clear error messages and enable proper error handling.
"""


class ChronoBaseError(Exception):
    """Base exception for all Chronograph errors.

    All other exceptions in the ChronoGraph middleware inherit from this class,
    allowing for catching all ChronoGraph-related errors with a single except clause.
    """
    pass


class ChronoSecurityError(ChronoBaseError):
    """Security-related errors in the ChronoGraph middleware.

    This exception is raised when there are issues with authentication, authorization,
    token validation, or other security-related operations.
    """
    pass


class ChronoQueryError(ChronoBaseError):
    """Query-related errors in the ChronoGraph middleware.

    This exception is raised when there are issues with database queries,
    including syntax errors, connection problems, or unexpected results.
    """
    pass


class ChronoIngestionError(ChronoBaseError):
    """Data ingestion errors in the ChronoGraph middleware.

    This exception is raised when there are issues with ingesting data into
    the ChronoGraph system, such as validation errors, duplicate entries,
    or storage failures.
    """
    pass


class ChronoConfigError(ChronoBaseError):
    """Configuration-related errors in the ChronoGraph middleware.

    This exception is raised when there are issues with the configuration of
    the ChronoGraph system, such as missing required parameters, invalid values,
    or incompatible settings.
    """
    pass


class ChronoConnectionError(ChronoBaseError):
    """Connection-related errors in the ChronoGraph middleware.

    This exception is raised when there are issues with connecting to external
    services or databases, such as network failures, authentication problems,
    or service unavailability.
    """
    pass
