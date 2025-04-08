"""
RESTful API for Medical Research Synthesizer

This module provides a Flask-based RESTful API for the Enhanced Medical Research Synthesizer.
It exposes endpoints for querying medical literature, analyzing contradictions,
managing knowledge bases, and exporting results in various formats.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.security import generate_password_hash, check_password_hash

import os
import json
import logging
import datetime
import tempfile
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any

# Import our enhanced medical research synthesizer
from enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
from query_builder import (
    MedicalQueryBuilder, MedicalCondition, MedicalIntervention, 
    OutcomeMetric, StudyDesign, DateRange, JournalFilter,
    PublicationTypeFilter, LanguageFilter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('medical_research_api')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=1)
jwt = JWTManager(app)

# Configure rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configure Swagger UI
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Medical Research Synthesizer API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Initialize the medical research synthesizer
synthesizer = EnhancedMedicalResearchSynthesizer(
    api_key=os.environ.get('NCBI_API_KEY'),
    email=os.environ.get('NCBI_EMAIL'),
    impact_factor_source=os.environ.get('IMPACT_FACTOR_SOURCE')
)

# Mock user database (replace with a real database in production)
users_db = {
    'admin@example.com': {
        'password': generate_password_hash('admin_password'),
        'role': 'admin'
    },
    'user@example.com': {
        'password': generate_password_hash('user_password'),
        'role': 'user'
    }
}

# In-memory storage for queries and results (replace with a database in production)
query_storage = {}
result_storage = {}
kb_storage = {}

# Helper functions
def serialize_article(article):
    """Serialize article data to make it JSON-serializable."""
    serialized = {}
    for key, value in article.items():
        if isinstance(value, datetime.datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, set):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized

def generate_export_filename(prefix, format):
    """Generate a unique filename for exports."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{prefix}_{timestamp}.{format}"

# Authentication routes
@app.route('/api/auth/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """User login endpoint."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    email = request.json.get('email', None)
    password = request.json.get('password', None)
    
    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400
    
    if email not in users_db:
        return jsonify({"error": "Invalid email or password"}), 401
    
    if not check_password_hash(users_db[email]['password'], password):
        return jsonify({"error": "Invalid email or password"}), 401
    
    # Create access token
    access_token = create_access_token(identity=email)
    return jsonify({"access_token": access_token, "role": users_db[email]['role']}), 200

@app.route('/api/auth/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    """User registration endpoint (only available to admins in production)."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    email = request.json.get('email', None)
    password = request.json.get('password', None)
    
    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400
    
    if email in users_db:
        return jsonify({"error": "Email already registered"}), 409
    
    # In production, this should be more secure and include email verification
    users_db[email] = {
        'password': generate_password_hash(password),
        'role': 'user'
    }
    
    return jsonify({"message": "User registered successfully"}), 201

# Query Builder Routes
@app.route('/api/query/templates', methods=['GET'])
@jwt_required()
def get_query_templates():
    """Get available query templates."""
    templates = [
        {"id": "covid-19 treatment", "name": "COVID-19 Treatment", "description": "Query for COVID-19 treatment studies"},
        {"id": "diabetes management", "name": "Diabetes Management", "description": "Query for Type 2 diabetes management"},
        {"id": "hypertension in elderly", "name": "Hypertension in Elderly", "description": "Query for hypertension treatment in elderly"},
        {"id": "community-acquired pneumonia", "name": "Community-Acquired Pneumonia", "description": "Query for CAP studies"}
    ]
    return jsonify(templates), 200

@app.route('/api/query/create', methods=['POST'])
@jwt_required()
@limiter.limit("20 per hour")
def create_query():
    """Create a new query using the query builder."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        # Create a new query builder
        builder = synthesizer.create_query()
        
        # Add conditions
        if 'conditions' in data:
            for condition in data['conditions']:
                medical_condition = MedicalCondition(condition['name'])
                
                if 'synonyms' in condition:
                    for synonym in condition['synonyms']:
                        medical_condition.add_synonym(synonym)
                
                if 'subtypes' in condition:
                    for subtype in condition['subtypes']:
                        medical_condition.add_subtype(subtype)
                
                builder.add_condition(medical_condition)
        
        # Add interventions
        if 'interventions' in data:
            for intervention in data['interventions']:
                medical_intervention = MedicalIntervention(
                    intervention['name'],
                    intervention.get('type', 'treatment')
                )
                
                if 'alternatives' in intervention:
                    for alt in intervention['alternatives']:
                        medical_intervention.add_alternative(alt)
                
                if 'specific_forms' in intervention:
                    for form in intervention['specific_forms']:
                        medical_intervention.add_specific_form(form)
                
                builder.add_intervention(medical_intervention)
        
        # Add outcomes
        if 'outcomes' in data:
            for outcome in data['outcomes']:
                outcome_metric = OutcomeMetric(
                    outcome['name'],
                    outcome.get('type', 'efficacy')
                )
                
                if 'synonyms' in outcome:
                    for synonym in outcome['synonyms']:
                        outcome_metric.add_synonym(synonym)
                
                if 'related_metrics' in outcome:
                    for metric in outcome['related_metrics']:
                        outcome_metric.add_related_metric(metric)
                
                builder.add_outcome(outcome_metric)
        
        # Add study designs
        if 'study_designs' in data:
            for design in data['study_designs']:
                study_design = StudyDesign(design['type'])
                
                if 'related_designs' in design:
                    for related in design['related_designs']:
                        study_design.add_related_design(related)
                
                if 'characteristics' in design:
                    for char in design['characteristics']:
                        study_design.add_characteristic(char)
                
                builder.add_study_design(study_design)
        
        # Add date range
        if 'date_range' in data:
            date_range = DateRange(
                data['date_range'].get('start_date'),
                data['date_range'].get('end_date')
            )
            
            if 'date_type' in data['date_range']:
                date_range.set_date_type(data['date_range']['date_type'])
            
            builder.set_date_range(date_range)
        elif 'years' in data:
            builder.last_n_years(int(data['years']))
        
        # Add filters
        if 'filters' in data:
            filters = data['filters']
            
            if 'language' in filters and filters['language'] == 'english_only':
                builder.english_only()
            
            if 'quality' in filters and filters['quality'] == 'high_quality_only':
                builder.high_quality_only()
            
            if 'humans_only' in filters and filters['humans_only']:
                builder.humans_only()
        
        # Build the query
        query_type = data.get('query_type', 'pico')
        use_mesh = data.get('use_mesh', True)
        
        if query_type.lower() == 'pico':
            query = builder.build_pico_query(use_mesh=use_mesh)
        else:
            query = builder.build_simple_query(use_mesh=use_mesh)
        
        # Store the query and builder
        query_id = str(uuid.uuid4())
        query_storage[query_id] = {
            'query': query,
            'builder': builder,
            'created_at': datetime.datetime.now().isoformat(),
            'user': get_jwt_identity()
        }
        
        return jsonify({
            'query_id': query_id,
            'query': query
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating query: {str(e)}")
        return jsonify({"error": f"Error creating query: {str(e)}"}), 500

@app.route('/api/query/template/<template_id>', methods=['GET'])
@jwt_required()
def get_query_from_template(template_id):
    """Create a query from a template."""
    try:
        builder = synthesizer.create_query_from_template(template_id)
        
        # Build the query
        query = builder.build_pico_query(use_mesh=True)
        
        # Store the query and builder
        query_id = str(uuid.uuid4())
        query_storage[query_id] = {
            'query': query,
            'builder': builder,
            'created_at': datetime.datetime.now().isoformat(),
            'user': get_jwt_identity(),
            'template': template_id
        }
        
        # Get explanation for this query
        explanation = synthesizer.query_interface.explain_query(query_type='pico', use_mesh=True)
        
        return jsonify({
            'query_id': query_id,
            'query': query,
            'components': explanation['components']
        }), 200
        
    except Exception as e:
        logger.error(f"Error creating query from template: {str(e)}")
        return jsonify({"error": f"Error creating query from template: {str(e)}"}), 500

# Search and Analysis Routes
@app.route('/api/search/execute', methods=['POST'])
@jwt_required()
@limiter.limit("20 per hour")
def execute_search():
    """Execute a search using a stored query or raw query."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        max_results = int(data.get('max_results', 20))
        
        # Get the query
        if 'query_id' in data and data['query_id'] in query_storage:
            # Use stored query
            query_data = query_storage[data['query_id']]
            builder = query_data['builder']
            query = query_data['query']
            enriched_results = synthesizer.search_and_enrich(query_builder=builder, max_results=max_results)
        elif 'query' in data:
            # Use raw query
            query = data['query']
            enriched_results = synthesizer.search_and_enrich(query=query, max_results=max_results)
        else:
            return jsonify({"error": "No query provided"}), 400
        
        # Serialize results
        serialized_results = [serialize_article(article) for article in enriched_results]
        
        # Store results
        result_id = str(uuid.uuid4())
        result_storage[result_id] = {
            'query': query,
            'results': serialized_results,
            'timestamp': datetime.datetime.now().isoformat(),
            'user': get_jwt_identity()
        }
        
        return jsonify({
            'result_id': result_id,
            'query': query,
            'total_results': len(serialized_results),
            'results': serialized_results
        }), 200
        
    except Exception as e:
        logger.error(f"Error executing search: {str(e)}")
        return jsonify({"error": f"Error executing search: {str(e)}"}), 500

@app.route('/api/analysis/contradictions', methods=['POST'])
@jwt_required()
@limiter.limit("10 per hour")
def analyze_contradictions():
    """Analyze contradictions in search results."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        if 'result_id' in data and data['result_id'] in result_storage:
            # Use stored results
            result_data = result_storage[data['result_id']]
            articles = result_data['results']
        elif 'query' in data:
            # Execute a new search
            query = data['query']
            max_results = int(data.get('max_results', 50))
            
            # This directly uses the contradiction analysis function
            analysis = synthesizer.search_and_analyze_contradictions(query, max_results)
            
            # Store the analysis
            analysis_id = str(uuid.uuid4())
            result_storage[analysis_id] = {
                'query': query,
                'analysis': analysis,
                'timestamp': datetime.datetime.now().isoformat(),
                'user': get_jwt_identity()
            }
            
            return jsonify({
                'analysis_id': analysis_id,
                'query': query,
                'total_articles': analysis['total_articles'],
                'num_contradictions': analysis['num_contradictions'],
                'contradictions': analysis['contradictions'],
                'by_topic': analysis['by_topic'] if 'by_topic' in analysis else {}
            }), 200
        else:
            return jsonify({"error": "No result_id or query provided"}), 400
        
    except Exception as e:
        logger.error(f"Error analyzing contradictions: {str(e)}")
        return jsonify({"error": f"Error analyzing contradictions: {str(e)}"}), 500

@app.route('/api/analysis/cap', methods=['GET'])
@jwt_required()
@limiter.limit("5 per hour")
def analyze_cap_treatments():
    """Specialized analysis of CAP treatment contradictions."""
    try:
        analysis = synthesizer.search_cap_contradictory_treatments()
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'analysis': analysis,
            'timestamp': datetime.datetime.now().isoformat(),
            'user': get_jwt_identity()
        }
        
        return jsonify({
            'analysis_id': analysis_id,
            'total_articles': analysis['total_articles'],
            'num_contradictions': analysis['num_contradictions'],
            'contradictions_by_intervention': {k: len(v) for k, v in analysis['contradictions_by_intervention'].items()},
            'authority_analysis': analysis['authority_analysis']
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing CAP treatments: {str(e)}")
        return jsonify({"error": f"Error analyzing CAP treatments: {str(e)}"}), 500

@app.route('/api/analysis/cap/detailed', methods=['GET'])
@jwt_required()
@limiter.limit("5 per hour")
def analyze_cap_detailed():
    """Detailed analysis of CAP treatment duration vs agent."""
    try:
        analysis = synthesizer.cap_duration_vs_agent_analysis()
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'analysis': analysis,
            'timestamp': datetime.datetime.now().isoformat(),
            'user': get_jwt_identity()
        }
        
        return jsonify({
            'analysis_id': analysis_id,
            'duration_articles': analysis['duration_articles'],
            'duration_contradictions': analysis['duration_contradictions'],
            'duration_consensus': analysis['duration_consensus'],
            'agent_articles': analysis['agent_articles'],
            'agent_contradictions': analysis['agent_contradictions'],
            'agent_consensus': analysis['agent_consensus']
        }), 200
        
    except Exception as e:
        logger.error(f"Error analyzing CAP detailed: {str(e)}")
        return jsonify({"error": f"Error analyzing CAP detailed: {str(e)}"}), 500

# Knowledge Base Routes
@app.route('/api/kb/create', methods=['POST'])
@jwt_required()
@limiter.limit("10 per day")
def create_knowledge_base():
    """Create a new knowledge base."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        if not all(k in data for k in ['name', 'query']):
            return jsonify({"error": "Missing required fields"}), 400
        
        name = data['name']
        query = data['query']
        schedule = data.get('schedule', 'weekly')
        max_results = int(data.get('max_results', 100))
        
        kb_info = synthesizer.create_and_update_knowledge_base(
            name=name,
            query=query,
            schedule=schedule,
            max_results=max_results
        )
        
        # Store KB info
        kb_id = str(uuid.uuid4())
        kb_storage[kb_id] = {
            'kb_info': kb_info,
            'created_at': datetime.datetime.now().isoformat(),
            'user': get_jwt_identity()
        }
        
        return jsonify({
            'kb_id': kb_id,
            'name': kb_info['name'],
            'query': kb_info['query'],
            'initial_results': kb_info['initial_results'],
            'update_schedule': kb_info['update_schedule'],
            'created_date': kb_info['created_date']
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        return jsonify({"error": f"Error creating knowledge base: {str(e)}"}), 500

@app.route('/api/kb/list', methods=['GET'])
@jwt_required()
def list_knowledge_bases():
    """List all knowledge bases."""
    try:
        kb_list = []
        for kb_id, kb_data in kb_storage.items():
            kb_info = kb_data['kb_info']
            kb_list.append({
                'kb_id': kb_id,
                'name': kb_info['name'],
                'query': kb_info['query'],
                'initial_results': kb_info['initial_results'],
                'update_schedule': kb_info['update_schedule'],
                'created_date': kb_info['created_date']
            })
        
        return jsonify(kb_list), 200
        
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        return jsonify({"error": f"Error listing knowledge bases: {str(e)}"}), 500

@app.route('/api/kb/<kb_name>', methods=['GET'])
@jwt_required()
def get_knowledge_base(kb_name):
    """Get articles from a knowledge base."""
    try:
        articles = synthesizer.get_knowledge_base(kb_name)
        
        # Serialize articles
        serialized_articles = [serialize_article(article) for article in articles]
        
        return jsonify({
            'name': kb_name,
            'total_articles': len(serialized_articles),
            'articles': serialized_articles
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting knowledge base: {str(e)}")
        return jsonify({"error": f"Error getting knowledge base: {str(e)}"}), 500

@app.route('/api/kb/<kb_name>/update', methods=['POST'])
@jwt_required()
@limiter.limit("10 per day")
def update_knowledge_base(kb_name):
    """Manually update a knowledge base."""
    try:
        # Find the KB in storage
        kb_id = None
        kb_data = None
        kb_query = None
        
        for id, data in kb_storage.items():
            if data['kb_info']['name'] == kb_name:
                kb_id = id
                kb_data = data
                kb_query = data['kb_info']['query']
                break
        
        if not kb_query:
            return jsonify({"error": f"Knowledge base '{kb_name}' not found"}), 404
        
        # Update the KB
        result = synthesizer.incremental_client.search_and_update_knowledge_base(
            kb_query,
            kb_data['kb_info']['kb_file'],
            max_results=100
        )
        
        return jsonify({
            'kb_id': kb_id,
            'name': kb_name,
            'query': kb_query,
            'total_count': result['total_count'],
            'new_count': result['new_count'],
            'update_time': result['update_time']
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating knowledge base: {str(e)}")
        return jsonify({"error": f"Error updating knowledge base: {str(e)}"}), 500

# Export Routes
@app.route('/api/export/pdf', methods=['POST'])
@jwt_required()
@limiter.limit("20 per day")
def export_to_pdf():
    """Export results to PDF."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        if 'result_id' in data and data['result_id'] in result_storage:
            # Get stored results
            result_data = result_storage[data['result_id']]
            
            # Generate PDF using ReportLab or similar
            # This is a placeholder - in a real implementation, you would use a PDF library
            pdf_content = f"Query: {result_data['query']}\n\n"
            
            for article in result_data['results']:
                pdf_content += f"Title: {article.get('title', 'No title')}\n"
                pdf_content += f"Authors: {', '.join(article.get('authors', []))}\n"
                pdf_content += f"Journal: {article.get('journal', 'Unknown')}\n"
                pdf_content += f"Publication Date: {article.get('human_date', article.get('publication_date', 'Unknown'))}\n"
                pdf_content += f"Abstract: {article.get('abstract', 'No abstract available')[:200]}...\n\n"
            
            # For now, just return JSON
            return jsonify({
                "message": "PDF export not yet implemented",
                "content_preview": pdf_content[:500]
            }), 200
        else:
            return jsonify({"error": "Invalid result_id"}), 400
        
    except Exception as e:
        logger.error(f"Error exporting to PDF: {str(e)}")
        return jsonify({"error": f"Error exporting to PDF: {str(e)}"}), 500

@app.route('/api/export/excel', methods=['POST'])
@jwt_required()
@limiter.limit("20 per day")
def export_to_excel():
    """Export results to Excel."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        if 'result_id' in data and data['result_id'] in result_storage:
            # Get stored results
            result_data = result_storage[data['result_id']]
            
            # Create DataFrame
            articles_list = []
            for article in result_data['results']:
                articles_list.append({
                    'PMID': article.get('pmid', ''),
                    'Title': article.get('title', ''),
                    'Authors': ', '.join(article.get('authors', [])),
                    'Journal': article.get('journal', ''),
                    'Publication Date': article.get('human_date', article.get('publication_date', '')),
                    'Impact Factor': article.get('impact_factor', ''),
                    'Citation Count': article.get('citation_count', ''),
                    'Authority Score': article.get('authority_score', '')
                })
            
            df = pd.DataFrame(articles_list)
            
            # Create Excel file
            excel_filename = generate_export_filename('search_results', 'xlsx')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
                temp_path = temp.name
                df.to_excel(temp_path, index=False)
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=excel_filename,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            return jsonify({"error": "Invalid result_id"}), 400
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return jsonify({"error": f"Error exporting to Excel: {str(e)}"}), 500

@app.route('/api/export/csv', methods=['POST'])
@jwt_required()
@limiter.limit("20 per day")
def export_to_csv():
    """Export results to CSV."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        if 'result_id' in data and data['result_id'] in result_storage:
            # Get stored results
            result_data = result_storage[data['result_id']]
            
            # Create DataFrame
            articles_list = []
            for article in result_data['results']:
                articles_list.append({
                    'PMID': article.get('pmid', ''),
                    'Title': article.get('title', ''),
                    'Authors': ', '.join(article.get('authors', [])),
                    'Journal': article.get('journal', ''),
                    'Publication Date': article.get('human_date', article.get('publication_date', '')),
                    'Impact Factor': article.get('impact_factor', ''),
                    'Citation Count': article.get('citation_count', ''),
                    'Authority Score': article.get('authority_score', '')
                })
            
            df = pd.DataFrame(articles_list)
            
            # Create CSV file
            csv_filename = generate_export_filename('search_results', 'csv')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
                temp_path = temp.name
                df.to_csv(temp_path, index=False)
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=csv_filename,
                mimetype='text/csv'
            )
        else:
            return jsonify({"error": "Invalid result_id"}), 400
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        return jsonify({"error": f"Error exporting to CSV: {str(e)}"}), 500

@app.route('/api/export/json', methods=['POST'])
@jwt_required()
@limiter.limit("20 per day")
def export_to_json():
    """Export results to JSON."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        if 'result_id' in data and data['result_id'] in result_storage:
            # Get stored results
            result_data = result_storage[data['result_id']]
            
            # Create JSON file
            json_filename = generate_export_filename('search_results', 'json')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as temp:
                temp_path = temp.name
                with open(temp_path, 'w') as f:
                    json.dump(result_data['results'], f, indent=2)
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=json_filename,
                mimetype='application/json'
            )
        else:
            return jsonify({"error": "Invalid result_id"}), 400
        
    except Exception as e:
        logger.error(f"Error exporting to JSON: {str(e)}")
        return jsonify({"error": f"Error exporting to JSON: {str(e)}"}), 500

@app.route('/api/export/markdown', methods=['POST'])
@jwt_required()
@limiter.limit("20 per day")
def export_to_markdown():
    """Export results to Markdown."""
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400
    
    try:
        data = request.json
        
        if 'result_id' in data and data['result_id'] in result_storage:
            # Get stored results
            result_data = result_storage[data['result_id']]
            
            # Create Markdown content
            md_content = f"# Search Results for: {result_data['query']}\n\n"
            md_content += f"_Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
            md_content += f"## Found {len(result_data['results'])} articles\n\n"
            
            for i, article in enumerate(result_data['results'], 1):
                md_content += f"### {i}. {article.get('title', 'No title')}\n\n"
                md_content += f"**PMID**: {article.get('pmid', 'Unknown')}\n\n"
                md_content += f"**Authors**: {', '.join(article.get('authors', []))}\n\n"
                md_content += f"**Journal**: {article.get('journal', 'Unknown')}\n\n"
                md_content += f"**Publication Date**: {article.get('human_date', article.get('publication_date', 'Unknown'))}\n\n"
                md_content += f"**Impact Factor**: {article.get('impact_factor', 'Unknown')}\n\n"
                md_content += f"**Citation Count**: {article.get('citation_count', 'Unknown')}\n\n"
                md_content += f"**Authority Score**: {article.get('authority_score', 'Unknown')}\n\n"
                md_content += f"**Abstract**:\n\n{article.get('abstract', 'No abstract available')}\n\n"
                md_content += "---\n\n"
            
            # Create Markdown file
            md_filename = generate_export_filename('search_results', 'md')
            with tempfile.NamedTemporaryFile(delete=False, suffix='.md') as temp:
                temp_path = temp.name
                with open(temp_path, 'w') as f:
                    f.write(md_content)
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=md_filename,
                mimetype='text/markdown'
            )
        else:
            return jsonify({"error": "Invalid result_id"}), 400
        
    except Exception as e:
        logger.error(f"Error exporting to Markdown: {str(e)}")
        return jsonify({"error": f"Error exporting to Markdown: {str(e)}"}), 500

# Main entry point
if __name__ == '__main__':
    # Create Swagger JSON
    with open('static/swagger.json', 'w') as f:
        swagger_json = {
            "swagger": "2.0",
            "info": {
                "title": "Medical Research Synthesizer API",
                "description": "API for accessing medical research synthesis features",
                "version": "1.0.0"
            },
            "basePath": "/api",
            "tags": [
                {"name": "Authentication", "description": "User authentication endpoints"},
                {"name": "Query", "description": "Query building endpoints"},
                {"name": "Search", "description": "Search and retrieval endpoints"},
                {"name": "Analysis", "description": "Analysis endpoints"},
                {"name": "Knowledge Base", "description": "Knowledge base management endpoints"},
                {"name": "Export", "description": "Export endpoints"}
            ],
            "paths": {
                # Swagger paths would go here - in a real implementation, this would be more comprehensive
            }
        }
        json.dump(swagger_json, f, indent=2)
    
    # Run the server
    app.run(debug=True, host='0.0.0.0', port=5000)