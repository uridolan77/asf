# app/api/routes/reports.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.services.report_service import ReportService
from app.schemas.report_schemas import (
    ReportDefinitionCreate,
    ReportDefinitionResponse,
    ReportExecutionRequest,
    ReportExecutionResponse,
    DimensionResponse,
    FactResponse,
    DataSourceResponse
)
from app.core.auth import get_current_user
from app.models.user_model import User

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("/data-sources", response_model=List[DataSourceResponse])
def get_data_sources(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get all available data sources for reports"""
    service = ReportService(db)
    return service.get_data_sources()

@router.get("/dimensions", response_model=List[DimensionResponse])
def get_dimensions(
    data_source_id: Optional[int] = Query(None), 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Get all available dimensions, optionally filtered by data source"""
    service = ReportService(db)
    return service.get_dimensions(data_source_id)

@router.get("/facts", response_model=List[FactResponse])
def get_facts(
    data_source_id: Optional[int] = Query(None), 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    """Get all available facts, optionally filtered by data source"""
    service = ReportService(db)
    return service.get_facts(data_source_id)

@router.post("/definitions", response_model=ReportDefinitionResponse)
def create_report_definition(
    report_def: ReportDefinitionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new report definition"""
    service = ReportService(db)
    return service.create_report_definition(report_def, current_user.id)

@router.get("/definitions", response_model=List[ReportDefinitionResponse])
def get_report_definitions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all report definitions for the current user"""
    service = ReportService(db)
    return service.get_report_definitions(current_user.id)

@router.get("/definitions/{report_id}", response_model=ReportDefinitionResponse)
def get_report_definition(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a specific report definition"""
    service = ReportService(db)
    report_def = service.get_report_definition(report_id)
    if not report_def:
        raise HTTPException(status_code=404, detail="Report definition not found")
    if report_def.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this report")
    return report_def

@router.delete("/definitions/{report_id}", status_code=204)
def delete_report_definition(
    report_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a report definition"""
    service = ReportService(db)
    report_def = service.get_report_definition(report_id)
    if not report_def:
        raise HTTPException(status_code=404, detail="Report definition not found")
    if report_def.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this report")
    service.delete_report_definition(report_id)
    return None

@router.post("/execute", response_model=ReportExecutionResponse)
def execute_report(
    execution_request: ReportExecutionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Execute a report with optional filters"""
    service = ReportService(db)
    
    # If report_definition_id is provided, verify access
    if execution_request.report_definition_id:
        report_def = service.get_report_definition(execution_request.report_definition_id)
        if not report_def:
            raise HTTPException(status_code=404, detail="Report definition not found")
        if report_def.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to execute this report")
    
    return service.execute_report(execution_request)

# app/schemas/report_schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AggregationFunction(str, Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"

class FilterOperator(str, Enum):
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    GREATER_THAN_EQUALS = "gte"
    LESS_THAN = "lt"
    LESS_THAN_EQUALS = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    BETWEEN = "between"

class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"

class DataSourceResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

class DimensionResponse(BaseModel):
    id: int
    name: str
    display_name: str
    data_type: str
    data_source_id: int
    description: Optional[str] = None

class FactResponse(BaseModel):
    id: int
    name: str
    display_name: str
    data_type: str
    data_source_id: int
    description: Optional[str] = None
    supported_aggregations: List[AggregationFunction]

class DimensionSelect(BaseModel):
    dimension_id: int
    alias: Optional[str] = None

class FactSelect(BaseModel):
    fact_id: int
    aggregation: AggregationFunction
    alias: Optional[str] = None

class FilterValue(BaseModel):
    value: Any
    second_value: Optional[Any] = None  # For BETWEEN operator

class Filter(BaseModel):
    dimension_id: Optional[int] = None
    fact_id: Optional[int] = None
    operator: FilterOperator
    values: List[FilterValue]

class SortSpec(BaseModel):
    dimension_id: Optional[int] = None
    fact_id: Optional[int] = None
    direction: SortDirection = SortDirection.ASC

class ReportDefinitionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    data_source_id: int
    dimensions: List[DimensionSelect]
    facts: List[FactSelect]
    filters: Optional[List[Filter]] = None
    sort: Optional[List[SortSpec]] = None
    limit: Optional[int] = 1000

class ReportDefinitionResponse(ReportDefinitionCreate):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

class ReportExecutionRequest(BaseModel):
    report_definition_id: Optional[int] = None
    # Allow ad-hoc report execution without saving a definition
    ad_hoc_definition: Optional[ReportDefinitionCreate] = None
    # Additional runtime filters
    runtime_filters: Optional[List[Filter]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "report_definition_id": 1,
                "runtime_filters": [
                    {
                        "dimension_id": 3,
                        "operator": "between",
                        "values": [
                            {"value": "2023-01-01", "second_value": "2023-12-31"}
                        ]
                    }
                ]
            }
        }

class ReportExecutionResponse(BaseModel):
    report_id: Optional[int] = None
    executed_at: datetime
    columns: List[str]
    data: List[Dict[str, Any]]
    total_rows: int
    execution_time_ms: float

# app/models/report_models.py
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base
from app.schemas.report_schemas import AggregationFunction, FilterOperator, SortDirection

class DataSource(Base):
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    connection_details = Column(JSON, nullable=False)  # Connection string, credentials, etc.
    
    dimensions = relationship("Dimension", back_populates="data_source")
    facts = relationship("Fact", back_populates="data_source")

class Dimension(Base):
    __tablename__ = "dimensions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)  # Backend name (table.column)
    display_name = Column(String(255), nullable=False)  # User-friendly name
    data_type = Column(String(50), nullable=False)  # string, number, date, boolean, etc.
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    description = Column(Text, nullable=True)
    
    data_source = relationship("DataSource", back_populates="dimensions")

class Fact(Base):
    __tablename__ = "facts"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)  # Backend name (table.column)
    display_name = Column(String(255), nullable=False)  # User-friendly name
    data_type = Column(String(50), nullable=False)  # number, percentage, currency, etc.
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    description = Column(Text, nullable=True)
    supported_aggregations = Column(JSON, nullable=False)  # List of supported aggregation functions
    
    data_source = relationship("DataSource", back_populates="facts")

class ReportDefinition(Base):
    __tablename__ = "report_definitions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    data_source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    dimensions = Column(JSON, nullable=False)  # List of dimension_ids and aliases
    facts = Column(JSON, nullable=False)  # List of fact_ids, aggregations, and aliases
    filters = Column(JSON, nullable=True)  # Optional filters
    sort = Column(JSON, nullable=True)  # Optional sort specifications
    limit = Column(Integer, default=1000)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    user = relationship("User", back_populates="report_definitions")
    data_source = relationship("DataSource")
    executions = relationship("ReportExecution", back_populates="report_definition")

class ReportExecution(Base):
    __tablename__ = "report_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    report_definition_id = Column(Integer, ForeignKey("report_definitions.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    executed_at = Column(DateTime(timezone=True), server_default=func.now())
    runtime_filters = Column(JSON, nullable=True)  # Additional filters applied at runtime
    execution_time_ms = Column(Integer, nullable=False)  # Execution time in milliseconds
    result_row_count = Column(Integer, nullable=False)
    ad_hoc_definition = Column(JSON, nullable=True)  # For reports executed without saving a definition
    
    report_definition = relationship("ReportDefinition", back_populates="executions")
    user = relationship("User", back_populates="report_executions")

# app/services/report_service.py
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import json

from app.models.report_models import (
    DataSource, Dimension, Fact, ReportDefinition, ReportExecution
)
from app.schemas.report_schemas import (
    ReportDefinitionCreate, ReportDefinitionResponse, ReportExecutionRequest,
    ReportExecutionResponse, DimensionResponse, FactResponse, DataSourceResponse,
    FilterOperator
)
from app.core.exceptions import NotFoundException, ValidationException
from app.repositories.report_repository import ReportRepository

class ReportService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = ReportRepository(db)
    
    def get_data_sources(self) -> List[DataSourceResponse]:
        """Get all available data sources"""
        return self.repo.get_all_data_sources()
    
    def get_dimensions(self, data_source_id: Optional[int] = None) -> List[DimensionResponse]:
        """Get dimensions, optionally filtered by data source"""
        return self.repo.get_dimensions(data_source_id)
    
    def get_facts(self, data_source_id: Optional[int] = None) -> List[FactResponse]:
        """Get facts, optionally filtered by data source"""
        return self.repo.get_facts(data_source_id)
    
    def create_report_definition(self, report_def: ReportDefinitionCreate, user_id: int) -> ReportDefinitionResponse:
        """Create a new report definition"""
        # Validate that the data source exists
        data_source = self.repo.get_data_source_by_id(report_def.data_source_id)
        if not data_source:
            raise ValidationException(f"Data source with id {report_def.data_source_id} not found")
        
        # Validate that all dimensions exist and belong to the data source
        for dim in report_def.dimensions:
            dimension = self.repo.get_dimension_by_id(dim.dimension_id)
            if not dimension:
                raise ValidationException(f"Dimension with id {dim.dimension_id} not found")
            if dimension.data_source_id != report_def.data_source_id:
                raise ValidationException(f"Dimension with id {dim.dimension_id} does not belong to the specified data source")
        
        # Validate that all facts exist and belong to the data source
        for fact in report_def.facts:
            fact_obj = self.repo.get_fact_by_id(fact.fact_id)
            if not fact_obj:
                raise ValidationException(f"Fact with id {fact.fact_id} not found")
            if fact_obj.data_source_id != report_def.data_source_id:
                raise ValidationException(f"Fact with id {fact.fact_id} does not belong to the specified data source")
            
            # Validate that the aggregation is supported for this fact
            supported_aggs = json.loads(fact_obj.supported_aggregations)
            if fact.aggregation not in supported_aggs:
                raise ValidationException(f"Aggregation {fact.aggregation} is not supported for fact {fact.fact_id}")
        
        # Validate filters if provided
        if report_def.filters:
            self._validate_filters(report_def.filters, report_def.data_source_id)
        
        # Create the report definition
        return self.repo.create_report_definition(report_def, user_id)
    
    def get_report_definitions(self, user_id: int) -> List[ReportDefinitionResponse]:
        """Get all report definitions for a user"""
        return self.repo.get_report_definitions_by_user_id(user_id)
    
    def get_report_definition(self, report_id: int) -> Optional[ReportDefinitionResponse]:
        """Get a specific report definition"""
        return self.repo.get_report_definition_by_id(report_id)
    
    def delete_report_definition(self, report_id: int) -> None:
        """Delete a report definition"""
        self.repo.delete_report_definition(report_id)
    
    def execute_report(self, execution_request: ReportExecutionRequest) -> ReportExecutionResponse:
        """Execute a report with optional filters"""
        start_time = time.time()
        
        # Determine the report definition to use
        report_def = None
        if execution_request.report_definition_id:
            report_def = self.repo.get_report_definition_by_id(execution_request.report_definition_id)
            if not report_def:
                raise NotFoundException(f"Report definition with id {execution_request.report_definition_id} not found")
        elif execution_request.ad_hoc_definition:
            # For ad-hoc execution, validate the definition
            data_source = self.repo.get_data_source_by_id(execution_request.ad_hoc_definition.data_source_id)
            if not data_source:
                raise ValidationException(f"Data source with id {execution_request.ad_hoc_definition.data_source_id} not found")
            
            report_def = execution_request.ad_hoc_definition
        else:
            raise ValidationException("Either report_definition_id or ad_hoc_definition must be provided")
        
        # Validate runtime filters if provided
        if execution_request.runtime_filters:
            data_source_id = report_def.data_source_id
            self._validate_filters(execution_request.runtime_filters, data_source_id)
        
        # Build and execute the query
        results = self.repo.execute_report_query(report_def, execution_request.runtime_filters)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Format the response
        response = ReportExecutionResponse(
            report_id=execution_request.report_definition_id,
            executed_at=datetime.now(),
            columns=results["columns"],
            data=results["data"],
            total_rows=len(results["data"]),
            execution_time_ms=execution_time
        )
        
        # Record the execution
        self._record_execution(
            execution_request,
            len(results["data"]),
            execution_time
        )
        
        return response
    
    def _validate_filters(self, filters, data_source_id: int) -> None:
        """Validate that filters reference valid dimensions/facts and use appropriate operators"""
        for filter_item in filters:
            if filter_item.dimension_id:
                dimension = self.repo.get_dimension_by_id(filter_item.dimension_id)
                if not dimension:
                    raise ValidationException(f"Dimension with id {filter_item.dimension_id} not found")
                if dimension.data_source_id != data_source_id:
                    raise ValidationException(f"Dimension with id {filter_item.dimension_id} does not belong to the specified data source")
                
                # Validate operator compatibility with data type
                self._validate_operator_for_data_type(dimension.data_type, filter_item.operator)
                
            elif filter_item.fact_id:
                fact = self.repo.get_fact_by_id(filter_item.fact_id)
                if not fact:
                    raise ValidationException(f"Fact with id {filter_item.fact_id} not found")
                if fact.data_source_id != data_source_id:
                    raise ValidationException(f"Fact with id {filter_item.fact_id} does not belong to the specified data source")
                
                # Validate operator compatibility with data type
                self._validate_operator_for_data_type(fact.data_type, filter_item.operator)
            else:
                raise ValidationException("Filter must reference either a dimension_id or fact_id")
    
    def _validate_operator_for_data_type(self, data_type: str, operator: FilterOperator) -> None:
        """Validate that the operator is compatible with the data type"""
        # String-specific operators
        string_operators = [
            FilterOperator.CONTAINS, FilterOperator.NOT_CONTAINS,
            FilterOperator.STARTS_WITH, FilterOperator.ENDS_WITH
        ]
        
        # Numeric comparison operators
        numeric_operators = [
            FilterOperator.GREATER_THAN, FilterOperator.GREATER_THAN_EQUALS,
            FilterOperator.LESS_THAN, FilterOperator.LESS_THAN_EQUALS,
            FilterOperator.BETWEEN
        ]
        
        if data_type.lower() in ["string", "text"] and operator in numeric_operators:
            raise ValidationException(f"Operator {operator} is not valid for string data type")
        
        if data_type.lower() in ["number", "integer", "float"] and operator in string_operators:
            raise ValidationException(f"Operator {operator} is not valid for numeric data type")
        
        if data_type.lower() == "boolean" and operator not in [FilterOperator.EQUALS, FilterOperator.NOT_EQUALS]:
            raise ValidationException(f"Operator {operator} is not valid for boolean data type")
    
    def _record_execution(self, execution_request: ReportExecutionRequest, row_count: int, execution_time: float) -> None:
        """Record a report execution for analytics and history"""
        self.repo.create_report_execution(
            report_definition_id=execution_request.report_definition_id,
            user_id=1,  # This should be the current user ID from auth context
            runtime_filters=execution_request.runtime_filters,
            execution_time_ms=int(execution_time),
            result_row_count=row_count,
            ad_hoc_definition=execution_request.ad_hoc_definition
        )

# app/repositories/report_repository.py
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Any
import json

from app.models.report_models import (
    DataSource, Dimension, Fact, ReportDefinition, ReportExecution
)
from app.schemas.report_schemas import (
    ReportDefinitionCreate, ReportDefinitionResponse, DimensionResponse, 
    FactResponse, DataSourceResponse, Filter
)
from app.core.exceptions import DatabaseException
from app.repositories.sql_builder import ReportSQLBuilder

class ReportRepository:
    def __init__(self, db: Session):
        self.db = db
        self.sql_builder = ReportSQLBuilder()
    
    def get_all_data_sources(self) -> List[DataSourceResponse]:
        """Get all data sources"""
        sources = self.db.query(DataSource).all()
        return [DataSourceResponse(
            id=source.id,
            name=source.name,
            description=source.description
        ) for source in sources]
    
    def get_data_source_by_id(self, data_source_id: int) -> Optional[DataSource]:
        """Get a data source by ID"""
        return self.db.query(DataSource).filter(DataSource.id == data_source_id).first()
    
    def get_dimensions(self, data_source_id: Optional[int] = None) -> List[DimensionResponse]:
        """Get dimensions, optionally filtered by data source"""
        query = self.db.query(Dimension)
        if data_source_id:
            query = query.filter(Dimension.data_source_id == data_source_id)
        
        dimensions = query.all()
        return [DimensionResponse(
            id=dim.id,
            name=dim.name,
            display_name=dim.display_name,
            data_type=dim.data_type,
            data_source_id=dim.data_source_id,
            description=dim.description
        ) for dim in dimensions]
    
    def get_dimension_by_id(self, dimension_id: int) -> Optional[Dimension]:
        """Get a dimension by ID"""
        return self.db.query(Dimension).filter(Dimension.id == dimension_id).first()
    
    def get_facts(self, data_source_id: Optional[int] = None) -> List[FactResponse]:
        """Get facts, optionally filtered by data source"""
        query = self.db.query(Fact)
        if data_source_id:
            query = query.filter(Fact.data_source_id == data_source_id)
        
        facts = query.all()
        return [FactResponse(
            id=fact.id,
            name=fact.name,
            display_name=fact.display_name,
            data_type=fact.data_type,
            data_source_id=fact.data_source_id,
            description=fact.description,
            supported_aggregations=json.loads(fact.supported_aggregations)
        ) for fact in facts]
    
    def get_fact_by_id(self, fact_id: int) -> Optional[Fact]:
        """Get a fact by ID"""
        return self.db.query(Fact).filter(Fact.id == fact_id).first()
    
    def create_report_definition(self, report_def: ReportDefinitionCreate, user_id: int) -> ReportDefinitionResponse:
        """Create a new report definition"""
        try:
            new_definition = ReportDefinition(
                name=report_def.name,
                description=report_def.description,
                user_id=user_id,
                data_source_id=report_def.data_source_id,
                dimensions=json.dumps([dim.dict() for dim in report_def.dimensions]),
                facts=json.dumps([fact.dict() for fact in report_def.facts]),
                filters=json.dumps([filter_item.dict() for filter_item in report_def.filters]) if report_def.filters else None,
                sort=json.dumps([sort_item.dict() for sort_item in report_def.sort]) if report_def.sort else None,
                limit=report_def.limit or 1000
            )
            
            self.db.add(new_definition)
            self.db.commit()
            self.db.refresh(new_definition)
            
            # Convert to response model
            return self._report_definition_to_response(new_definition)
        except Exception as e:
            self.db.rollback()
            raise DatabaseException(f"Failed to create report definition: {str(e)}")
    
    def get_report_definitions_by_user_id(self, user_id: int) -> List[ReportDefinitionResponse]:
        """Get all report definitions for a user"""
        definitions = self.db.query(ReportDefinition).filter(ReportDefinition.user_id == user_id).all()
        return [self._report_definition_to_response(definition) for definition in definitions]
    
    def get_report_definition_by_id(self, report_id: int) -> Optional[ReportDefinitionResponse]:
        """Get a specific report definition"""
        definition = self.db.query(ReportDefinition).filter(ReportDefinition.id == report_id).first()
        if definition:
            return self._report_definition_to_response(definition)
        return None
    
    def delete_report_definition(self, report_id: int) -> None:
        """Delete a report definition"""
        try:
            definition = self.db.query(ReportDefinition).filter(ReportDefinition.id == report_id).first()
            if definition:
                self.db.delete(definition)
                self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise DatabaseException(f"Failed to delete report definition: {str(e)}")
    
    def execute_report_query(self, report_def, runtime_filters: Optional[List[Filter]] = None) -> Dict[str, Any]:
        """
        Execute a report query based on the definition and runtime filters
        Returns a dictionary with 'columns' and 'data' keys
        """
        try:
            # Get the actual dimensions and facts objects to build the query
            dimensions = []
            for dim in report_def.dimensions:
                dimension = self.get_dimension_by_id(dim.dimension_id)
                if dimension:
                    dimensions.append({
                        "object": dimension,
                        "alias": dim.alias
                    })
            
            facts = []
            for fact in report_def.facts:
                fact_obj = self.get_fact_by_id(fact.fact_id)
                if fact_obj:
                    facts.append({
                        "object": fact_obj,
                        "aggregation": fact.aggregation,
                        "alias": fact.alias
                    })
            
            # Build the SQL query using the SQL builder
            data_source = self.get_data_source_by_id(report_def.data_source_id)
            if not data_source:
                raise ValueError(f"Data source with id {report_def.data_source_id} not found")
            
            connection_details = json.loads(data_source.connection_details)
            
            # Combine report definition filters and runtime filters
            all_filters = []
            if hasattr(report_def, 'filters') and report_def.filters:
                # Handle dict or list format depending on source
                filters = report_def.filters
                if isinstance(filters, str):
                    # If it's from the database, it will be a JSON string
                    filters = json.loads(filters)
                
                # Add each filter to all_filters
                for filter_item in filters:
                    all_filters.append(filter_item)
            
            if runtime_filters:
                all_filters.extend(runtime_filters)
            
            # Build and execute the SQL query
            sql_query = self.sql_builder.build_query(
                data_source=data_source,
                dimensions=dimensions,
                facts=facts,
                filters=all_filters,
                sort=report_def.sort if hasattr(report_def, 'sort') and report_def.sort else None,
                limit=report_def.limit if hasattr(report_def, 'limit') else 1000
            )
            
            # Execute the query
            result = self.db.execute(text(sql_query))
            
            # Process the results
            columns = result.keys()
            data = [dict(row) for row in result]
            
            return {
                "columns": list(columns),
                "data": data
            }
        except Exception as e:
            self.db.rollback()
            raise DatabaseException(f"Failed to execute report query: {str(e)}")
    
    def create_report_execution(
        self, 
        report_definition_id: Optional[int],
        user_id: int,
        runtime_filters: Optional[List[Filter]],
        execution_time_ms: int,
        result_row_count: int,
        ad_hoc_definition: Optional[ReportDefinitionCreate]
    ) -> None:
        """Record a report execution"""
        try:
            new_execution = ReportExecution(
                report_definition_id=report_definition_id,
                user_id=user_id,
                runtime_filters=json.dumps([f.dict() for f in runtime_filters]) if runtime_filters else None,
                execution_time_ms=execution_time_ms,
                result_row_count=result_row_count,
                ad_hoc_definition=json.dumps(ad_hoc_definition.dict()) if ad_hoc_definition else None
            )
            
            self.db.add(new_execution)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            raise DatabaseException(f"Failed to record report execution: {str(e)}")
    
    def _report_definition_to_response(self, definition: ReportDefinition) -> ReportDefinitionResponse:
        """Convert a ReportDefinition DB model to a ReportDefinitionResponse schema"""
        return ReportDefinitionResponse(
            id=definition.id,
            name=definition.name,
            description=definition.description,
            user_id=definition.user_id,
            data_source_id=definition.data_source_id,
            dimensions=json.loads(definition.dimensions),
            facts=json.loads(definition.facts),
            filters=json.loads(definition.filters) if definition.filters else None,
            sort=json.loads(definition.sort) if definition.sort else None,
            limit=definition.limit,
            created_at=definition.created_at,
            updated_at=definition.updated_at
        )

# app/repositories/sql_builder.py
from typing import List, Dict, Any, Optional
from app.models.report_models import DataSource, Dimension, Fact
from app.schemas.report_schemas import Filter, SortSpec
import json

class ReportSQLBuilder:
    """
    Builds SQL queries for report execution based on the specified
    dimensions, facts, filters, and sorting.
    """
    
    def build_query(
        self,
        data_source: DataSource,
        dimensions: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        filters: Optional[List[Filter]] = None,
        sort: Optional[List[SortSpec]] = None,
        limit: int = 1000
    ) -> str:
        """
        Build a SQL query based on the report definition components
        
        Args:
            data_source: The data source to query
            dimensions: List of dimensions to include
            facts: List of facts with aggregations
            filters: Optional filters to apply
            sort: Optional sorting specifications
            limit: Maximum number of rows to return
            
        Returns:
            A SQL query string
        """
        # Extract table relationships from connection details
        connection_details = json.loads(data_source.connection_details)
        tables_info = connection_details.get("tables", {})
        relationships = connection_details.get("relationships", [])
        
        # Build SELECT clause
        select_parts = []
        group_by_parts = []
        
        # Add dimensions to SELECT and GROUP BY
        for i, dim in enumerate(dimensions):
            dimension_obj = dim["object"]
            alias = dim["alias"] or f"dim_{i}"
            
            # Extract table and column from dimension name (format: table.column)
            table_name, column_name = dimension_obj.name.split(".")
            
            select_parts.append(f"{dimension_obj.name} AS {alias}")
            group_by_parts.append(dimension_obj.name)
        
        # Add facts with aggregations to SELECT
        for i, fact in enumerate(facts):
            fact_obj = fact["object"]
            aggregation = fact["aggregation"]
            alias = fact["alias"] or f"fact_{i}"
            
            # Apply aggregation function
            select_parts.append(f"{aggregation}({fact_obj.name}) AS {alias}")
        
        # Determine the FROM clause with necessary joins
        from_clause = self._build_from_clause(dimensions, facts, relationships)
        
        # Build WHERE clause from filters
        where_clause = ""
        if filters and len(filters) > 0:
            where_conditions = self._build_filter_conditions(filters)
            if where_conditions:
                where_clause = f"WHERE {where_conditions}"
        
        # Build ORDER BY clause
        order_by_clause = ""
        if sort and len(sort) > 0:
            sort_parts = []
            for sort_spec in sort:
                if sort_spec.dimension_id:
                    dimension = next((d["object"] for d in dimensions if d["object"].id == sort_spec.dimension_id), None)
                    if dimension:
                        sort_parts.append(f"{dimension.name} {sort_spec.direction}")
                elif sort_spec.fact_id:
                    fact = next((f["object"] for f in facts if f["object"].id == sort_spec.fact_id), None)
                    if fact:
                        # Find the alias for this fact
                        fact_item = next((f for f in facts if f["object"].id == sort_spec.fact_id), None)
                        if fact_item:
                            alias = fact_item["alias"] or f"fact_{facts.index(fact_item)}"
                            sort_parts.append(f"{alias} {sort_spec.direction}")
            
            if sort_parts:
                order_by_clause = f"ORDER BY {', '.join(sort_parts)}"
        
        # Build the final query
        query = f"""
        SELECT 
            {', '.join(select_parts)}
        FROM 
            {from_clause}
        {where_clause}
        {f"GROUP BY {', '.join(group_by_parts)}" if group_by_parts else ""}
        {order_by_clause}
        LIMIT {limit}
        """
        
        return query
    
    def _build_from_clause(
        self, 
        dimensions: List[Dict[str, Any]], 
        facts: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> str:
        """
        Build the FROM clause with appropriate joins based on dimensions and facts
        """
        # Collect all tables needed for the query
        tables = set()
        
        # Add tables from dimensions
        for dim in dimensions:
            dimension_obj = dim["object"]
            table_name = dimension_obj.name.split(".")[0]
            tables.add(table_name)
        
        # Add tables from facts
        for fact in facts:
            fact_obj = fact["object"]
            table_name = fact_obj.name.split(".")[0]
            tables.add(table_name)
        
        # If there's only one table, return it directly
        if len(tables) == 1:
            return next(iter(tables))
        
        # Otherwise, build JOINs based on relationships
        tables_list = list(tables)
        main_table = tables_list[0]
        from_clause = main_table
        
        # Create a set to track which joins we've already added
        added_joins = set()
        
        # Find the minimal spanning tree of joins
        for i in range(1, len(tables_list)):
            target_table = tables_list[i]
            
            # Find a path from main_table to target_table
            path = self._find_join_path(main_table, target_table, relationships)
            
            if path:
                for j in range(len(path) - 1):
                    from_table = path[j]
                    to_table = path[j + 1]
                    
                    # Find the relationship between these tables
                    for rel in relationships:
                        if (rel["from_table"] == from_table and rel["to_table"] == to_table) or \
                           (rel["from_table"] == to_table and rel["to_table"] == from_table):
                            
                            join_key = f"{from_table}_{to_table}"
                            if join_key not in added_joins:
                                if rel["from_table"] == from_table:
                                    from_clause += f" JOIN {to_table} ON {from_table}.{rel['from_column']} = {to_table}.{rel['to_column']}"
                                else:
                                    from_clause += f" JOIN {to_table} ON {to_table}.{rel['to_column']} = {from_table}.{rel['from_column']}"
                                
                                added_joins.add(join_key)
                                added_joins.add(f"{to_table}_{from_table}")  # Add both directions
        
        return from_clause
    
    def _find_join_path(self, start_table: str, end_table: str, relationships: List[Dict[str, Any]]) -> List[str]:
        """
        Find a path of joins from start_table to end_table
        """
        # Build an adjacency list representation of the table relationships
        graph = {}
        for rel in relationships:
            from_table = rel["from_table"]
            to_table = rel["to_table"]
            
            if from_table not in graph:
                graph[from_table] = []
            if to_table not in graph:
                graph[to_table] = []
            
            graph[from_table].append(to_table)
            graph[to_table].append(from_table)  # Add bidirectional edges
        
        # BFS to find the shortest path
        visited = set()
        queue = [(start_table, [start_table])]
        
        while queue:
            (node, path) = queue.pop(0)
            
            if node == end_table:
                return path
            
            if node not in visited:
                visited.add(node)
                
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return None  # No path found
    
    def _build_filter_conditions(self, filters: List[Filter]) -> str:
        """
        Build the WHERE clause conditions from filters
        """
        conditions = []
        
        for filter_item in filters:
            field = None
            if filter_item.dimension_id:
                # Placeholder - in real implementation, would fetch dimension.name
                field = f"dimension_{filter_item.dimension_id}"
            elif filter_item.fact_id:
                # Placeholder - in real implementation, would fetch fact.name
                field = f"fact_{filter_item.fact_id}"
            
            if field:
                # Build condition based on operator
                if filter_item.operator == "eq":
                    conditions.append(f"{field} = '{filter_item.values[0].value}'")
                elif filter_item.operator == "neq":
                    conditions.append(f"{field} != '{filter_item.values[0].value}'")
                elif filter_item.operator == "gt":
                    conditions.append(f"{field} > {filter_item.values[0].value}")
                elif filter_item.operator == "gte":
                    conditions.append(f"{field} >= {filter_item.values[0].value}")
                elif filter_item.operator == "lt":
                    conditions.append(f"{field} < {filter_item.values[0].value}")
                elif filter_item.operator == "lte":
                    conditions.append(f"{field} <= {filter_item.values[0].value}")
                elif filter_item.operator == "in":
                    values = [f"'{v.value}'" for v in filter_item.values]
                    conditions.append(f"{field} IN ({', '.join(values)})")
                elif filter_item.operator == "not_in":
                    values = [f"'{v.value}'" for v in filter_item.values]
                    conditions.append(f"{field} NOT IN ({', '.join(values)})")
                elif filter_item.operator == "contains":
                    conditions.append(f"{field} LIKE '%{filter_item.values[0].value}%'")
                elif filter_item.operator == "not_contains":
                    conditions.append(f"{field} NOT LIKE '%{filter_item.values[0].value}%'")
                elif filter_item.operator == "starts_with":
                    conditions.append(f"{field} LIKE '{filter_item.values[0].value}%'")
                elif filter_item.operator == "ends_with":
                    conditions.append(f"{field} LIKE '%{filter_item.values[0].value}'")
                elif filter_item.operator == "between":
                    value = filter_item.values[0]
                    conditions.append(f"{field} BETWEEN '{value.value}' AND '{value.second_value}'")
        
        return " AND ".join(conditions) if conditions else ""

# app/core/exceptions.py
from fastapi import HTTPException
from typing import Any, Dict, Optional

class BaseAppException(Exception):
    """Base exception class for application errors"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ValidationException(BaseAppException):
    """Exception raised for validation errors"""
    def __init__(self, message: str):
        super().__init__(message)

class DatabaseException(BaseAppException):
    """Exception raised for database errors"""
    def __init__(self, message: str):
        super().__init__(message)

class NotFoundException(BaseAppException):
    """Exception raised when a resource is not found"""
    def __init__(self, message: str):
        super().__init__(message)

class AuthenticationException(BaseAppException):
    """Exception raised for authentication errors"""
    def __init__(self, message: str):
        super().__init__(message)

class AuthorizationException(BaseAppException):
    """Exception raised for authorization errors"""
    def __init__(self, message: str):
        super().__init__(message)

def handle_app_exception(exc: BaseAppException) -> HTTPException:
    """Convert application exceptions to FastAPI HTTPExceptions"""
    if isinstance(exc, ValidationException):
        return HTTPException(status_code=400, detail=exc.message)
    elif isinstance(exc, DatabaseException):
        return HTTPException(status_code=500, detail=exc.message)
    elif isinstance(exc, NotFoundException):
        return HTTPException(status_code=404, detail=exc.message)
    elif isinstance(exc, AuthenticationException):
        return HTTPException(status_code=401, detail=exc.message)
    elif isinstance(exc, AuthorizationException):
        return HTTPException(status_code=403, detail=exc.message)
    else:
        return HTTPException(status_code=500, detail=str(exc))

# app/api/exception_handlers.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.exceptions import BaseAppException, handle_app_exception

async def app_exception_handler(request: Request, exc: BaseAppException):
    """Handle application exceptions"""
    http_exc = handle_app_exception(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content={"detail": http_exc.detail}
    )

# Data migration scripts
# migrations/versions/001_create_reporting_tables.py
"""
Create reporting tables for the dynamic reporting system

Revision ID: 001
Revises: 
Create Date: 2023-05-01 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create data_sources table
    op.create_table(
        'data_sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('connection_details', JSON, nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create dimensions table
    op.create_table(
        'dimensions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('data_type', sa.String(50), nullable=False),
        sa.Column('data_source_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create facts table
    op.create_table(
        'facts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('display_name', sa.String(255), nullable=False),
        sa.Column('data_type', sa.String(50), nullable=False),
        sa.Column('data_source_id', sa.Integer(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('supported_aggregations', JSON, nullable=False),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create report_definitions table
    op.create_table(
        'report_definitions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('data_source_id', sa.Integer(), nullable=False),
        sa.Column('dimensions', JSON, nullable=False),
        sa.Column('facts', JSON, nullable=False),
        sa.Column('filters', JSON, nullable=True),
        sa.Column('sort', JSON, nullable=True),
        sa.Column('limit', sa.Integer(), nullable=False, server_default='1000'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create report_executions table
    op.create_table(
        'report_executions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('report_definition_id', sa.Integer(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('executed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('runtime_filters', JSON, nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=False),
        sa.Column('result_row_count', sa.Integer(), nullable=False),
        sa.Column('ad_hoc_definition', JSON, nullable=True),
        sa.ForeignKeyConstraint(['report_definition_id'], ['report_definitions.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_report_definitions_user_id', 'report_definitions', ['user_id'])
    op.create_index('ix_report_executions_report_definition_id', 'report_executions', ['report_definition_id'])
    op.create_index('ix_report_executions_user_id', 'report_executions', ['user_id'])
    op.create_index('ix_dimensions_data_source_id', 'dimensions', ['data_source_id'])
    op.create_index('ix_facts_data_source_id', 'facts', ['data_source_id'])

def downgrade():
    op.drop_table('report_executions')
    op.drop_table('report_definitions')
    op.drop_table('facts')
    op.drop_table('dimensions')
    op.drop_table('data_sources')