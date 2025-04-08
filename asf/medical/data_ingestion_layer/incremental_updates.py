"""
Incremental Updates Module for Medical Research Synthesizer

This module extends the NCBIClient with functionality to track and retrieve only
new publications since the last query, facilitating regular knowledge base updates.
"""

import os
import json
import datetime
import logging
import sqlite3
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('incremental_updates')

@dataclass
class QueryHistory:
    """Class to store query history information."""
    query_id: str
    query_text: str
    last_run: datetime.datetime
    database: str
    result_count: int
    retrieved_ids: List[str]
    

class IncrementalUpdateManager:
    """
    Manages incremental updates for NCBI queries, allowing retrieval of only new
    publications since the last query execution.
    """
    
    def __init__(self, 
                 storage_dir: Optional[str] = None,
                 use_sqlite: bool = True):
        """
        Initialize the incremental update manager.
        
        Args:
            storage_dir: Directory to store query history
            use_sqlite: Whether to use SQLite for storage (more robust) or JSON
        """
        # Set up storage directory
        self.storage_dir = storage_dir or os.path.join(os.path.expanduser("~"), ".pubmed_updates")
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.use_sqlite = use_sqlite
        
        # File paths
        self.json_file = os.path.join(self.storage_dir, "query_history.json")
        self.db_file = os.path.join(self.storage_dir, "query_history.db")
        
        # Initialize storage
        if use_sqlite:
            self._init_sqlite_db()
        else:
            self._init_json_storage()
    
    def _init_sqlite_db(self):
        """Initialize SQLite database for storing query history."""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                last_run TEXT NOT NULL,
                database TEXT NOT NULL,
                result_count INTEGER NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS retrieved_ids (
                query_id TEXT,
                item_id TEXT,
                retrieved_date TEXT,
                PRIMARY KEY (query_id, item_id),
                FOREIGN KEY (query_id) REFERENCES queries(query_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Initialized SQLite database at {self.db_file}")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {str(e)}")
            raise
    
    def _init_json_storage(self):
        """Initialize JSON storage for query history."""
        if not os.path.exists(self.json_file):
            with open(self.json_file, 'w') as f:
                json.dump({}, f)
            logger.info(f"Initialized JSON storage at {self.json_file}")
    
    def _generate_query_id(self, query_text: str, database: str) -> str:
        """
        Generate a unique ID for a query.
        
        Args:
            query_text: The query text
            database: The database being queried
            
        Returns:
            Unique query ID
        """
        import hashlib
        query_string = f"{query_text}|{database}".encode('utf-8')
        return hashlib.md5(query_string).hexdigest()
    
    def register_query(self, 
                      query_text: str, 
                      database: str, 
                      retrieved_ids: List[str] = None,
                      result_count: int = 0) -> str:
        """
        Register a new query or update an existing one.
        
        Args:
            query_text: The query text
            database: The database being queried
            retrieved_ids: List of retrieved item IDs
            result_count: Total count of results
            
        Returns:
            Query ID
        """
        query_id = self._generate_query_id(query_text, database)
        now = datetime.datetime.now().isoformat()
        
        if self.use_sqlite:
            try:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                
                # Insert or update query
                cursor.execute('''
                INSERT OR REPLACE INTO queries 
                (query_id, query_text, last_run, database, result_count)
                VALUES (?, ?, ?, ?, ?)
                ''', (query_id, query_text, now, database, result_count))
                
                # Insert retrieved IDs
                if retrieved_ids:
                    for item_id in retrieved_ids:
                        cursor.execute('''
                        INSERT OR REPLACE INTO retrieved_ids
                        (query_id, item_id, retrieved_date)
                        VALUES (?, ?, ?)
                        ''', (query_id, item_id, now))
                
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Error registering query in SQLite: {str(e)}")
                raise
        else:
            try:
                # Load existing data
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                
                # Update query data
                if query_id not in data:
                    data[query_id] = {
                        'query_text': query_text,
                        'database': database,
                        'history': []
                    }
                
                # Add new history entry
                history_entry = {
                    'timestamp': now,
                    'result_count': result_count,
                    'retrieved_ids': retrieved_ids or []
                }
                
                data[query_id]['history'].append(history_entry)
                data[query_id]['last_run'] = now
                
                # Save updated data
                with open(self.json_file, 'w') as f:
                    json.dump(data, f)
            except Exception as e:
                logger.error(f"Error registering query in JSON storage: {str(e)}")
                raise
        
        logger.info(f"Registered query with ID {query_id}")
        return query_id
    
    def get_query_history(self, query_id: str) -> Optional[QueryHistory]:
        """
        Get history for a specific query.
        
        Args:
            query_id: Query ID
            
        Returns:
            QueryHistory object if found, None otherwise
        """
        if self.use_sqlite:
            try:
                conn = sqlite3.connect(self.db_file)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get query details
                cursor.execute('SELECT * FROM queries WHERE query_id = ?', (query_id,))
                query_row = cursor.fetchone()
                
                if not query_row:
                    conn.close()
                    return None
                
                # Get retrieved IDs
                cursor.execute('SELECT item_id FROM retrieved_ids WHERE query_id = ?', (query_id,))
                id_rows = cursor.fetchall()
                retrieved_ids = [row['item_id'] for row in id_rows]
                
                conn.close()
                
                return QueryHistory(
                    query_id=query_row['query_id'],
                    query_text=query_row['query_text'],
                    last_run=datetime.datetime.fromisoformat(query_row['last_run']),
                    database=query_row['database'],
                    result_count=query_row['result_count'],
                    retrieved_ids=retrieved_ids
                )
            except Exception as e:
                logger.error(f"Error getting query history from SQLite: {str(e)}")
                return None
        else:
            try:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                
                if query_id not in data:
                    return None
                
                query_data = data[query_id]
                
                # Get latest history entry
                if not query_data.get('history'):
                    return None
                    
                latest_entry = max(
                    query_data['history'], 
                    key=lambda x: x['timestamp']
                )
                
                # Collect all retrieved IDs across all history entries
                all_ids = set()
                for entry in query_data['history']:
                    all_ids.update(entry.get('retrieved_ids', []))
                
                return QueryHistory(
                    query_id=query_id,
                    query_text=query_data['query_text'],
                    last_run=datetime.datetime.fromisoformat(latest_entry['timestamp']),
                    database=query_data['database'],
                    result_count=latest_entry['result_count'],
                    retrieved_ids=list(all_ids)
                )
            except Exception as e:
                logger.error(f"Error getting query history from JSON storage: {str(e)}")
                return None
    
    def get_query_id_by_text(self, query_text: str, database: str) -> Optional[str]:
        """
        Get query ID for a specific query text and database.
        
        Args:
            query_text: The query text
            database: The database being queried
            
        Returns:
            Query ID if found, None otherwise
        """
        return self._generate_query_id(query_text, database)
    
    def get_last_run_date(self, query_id: str) -> Optional[datetime.datetime]:
        """
        Get the date when a query was last run.
        
        Args:
            query_id: Query ID
            
        Returns:
            Datetime of last run if found, None otherwise
        """
        history = self.get_query_history(query_id)
        return history.last_run if history else None
    
    def get_previously_retrieved_ids(self, query_id: str) -> Set[str]:
        """
        Get set of IDs that were previously retrieved for a query.
        
        Args:
            query_id: Query ID
            
        Returns:
            Set of previously retrieved IDs
        """
        history = self.get_query_history(query_id)
        return set(history.retrieved_ids if history else [])
    
    def build_incremental_query(self, 
                               base_query: str, 
                               database: str,
                               days_overlap: int = 1) -> Tuple[str, Optional[str]]:
        """
        Build an incremental query by adding date constraints.
        
        Args:
            base_query: The base query text
            database: The database to query
            days_overlap: Number of days to overlap with previous query (to avoid missing items)
            
        Returns:
            Tuple of (incremental query, query ID)
        """
        query_id = self.get_query_id_by_text(base_query, database)
        last_run_date = self.get_last_run_date(query_id)
        
        if not last_run_date:
            # No previous run, return original query
            return base_query, query_id
        
        # Calculate date from which to search (with overlap)
        from_date = last_run_date - datetime.timedelta(days=days_overlap)
        date_str = from_date.strftime("%Y/%m/%d")
        
        # Add date constraint to query
        if database.lower() == 'pubmed':
            # For PubMed, use EDAT (Entrez Date) filter
            incremental_query = f"({base_query}) AND {date_str}:3000[EDAT]"
        else:
            # For other databases, use a generic date filter
            # Note: This might need to be customized for specific databases
            incremental_query = f"({base_query}) AND {date_str}:3000[Date - Create]"
        
        return incremental_query, query_id
    
    def identify_new_results(self, 
                            query_id: str, 
                            current_ids: List[str]) -> List[str]:
        """
        Identify new results compared to previous runs.
        
        Args:
            query_id: Query ID
            current_ids: List of IDs from current query
            
        Returns:
            List of new IDs not seen in previous runs
        """
        previous_ids = self.get_previously_retrieved_ids(query_id)
        new_ids = [id for id in current_ids if id not in previous_ids]
        
        logger.info(f"Found {len(new_ids)} new results out of {len(current_ids)} total results for query {query_id}")
        return new_ids
    
    def update_knowledge_base(self, 
                             query_id: str, 
                             new_data: List[Dict[str, Any]],
                             kb_file: str) -> None:
        """
        Update a JSON knowledge base with new data.
        
        Args:
            query_id: Query ID
            new_data: List of new data items
            kb_file: Path to knowledge base file
        """
        if not new_data:
            logger.info(f"No new data to update knowledge base for query {query_id}")
            return
        
        try:
            # Load existing KB if it exists
            kb = []
            if os.path.exists(kb_file):
                with open(kb_file, 'r') as f:
                    kb = json.load(f)
            
            # Add new data
            kb.extend(new_data)
            
            # Save updated KB
            with open(kb_file, 'w') as f:
                json.dump(kb, f, indent=2)
            
            logger.info(f"Updated knowledge base at {kb_file} with {len(new_data)} new items")
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
    
    def list_tracked_queries(self) -> List[Dict[str, Any]]:
        """
        List all tracked queries.
        
        Returns:
            List of query information dictionaries
        """
        if self.use_sqlite:
            try:
                conn = sqlite3.connect(self.db_file)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT query_id, query_text, last_run, database, result_count 
                FROM queries
                ORDER BY last_run DESC
                ''')
                
                rows = cursor.fetchall()
                conn.close()
                
                queries = []
                for row in rows:
                    queries.append({
                        'query_id': row['query_id'],
                        'query_text': row['query_text'],
                        'last_run': row['last_run'],
                        'database': row['database'],
                        'result_count': row['result_count']
                    })
                
                return queries
            except Exception as e:
                logger.error(f"Error listing queries from SQLite: {str(e)}")
                return []
        else:
            try:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                
                queries = []
                for query_id, query_data in data.items():
                    if 'history' in query_data and query_data['history']:
                        latest_entry = max(
                            query_data['history'], 
                            key=lambda x: x['timestamp']
                        )
                        
                        queries.append({
                            'query_id': query_id,
                            'query_text': query_data['query_text'],
                            'last_run': latest_entry['timestamp'],
                            'database': query_data['database'],
                            'result_count': latest_entry['result_count']
                        })
                
                # Sort by last run date, most recent first
                queries.sort(key=lambda x: x['last_run'], reverse=True)
                return queries
            except Exception as e:
                logger.error(f"Error listing queries from JSON storage: {str(e)}")
                return []
    
    def delete_query_history(self, query_id: str) -> bool:
        """
        Delete history for a specific query.
        
        Args:
            query_id: Query ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if self.use_sqlite:
            try:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                
                # Delete query and its retrieved IDs
                cursor.execute('DELETE FROM retrieved_ids WHERE query_id = ?', (query_id,))
                cursor.execute('DELETE FROM queries WHERE query_id = ?', (query_id,))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Deleted query history for {query_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting query history from SQLite: {str(e)}")
                return False
        else:
            try:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                
                if query_id in data:
                    del data[query_id]
                    
                    with open(self.json_file, 'w') as f:
                        json.dump(data, f)
                    
                    logger.info(f"Deleted query history for {query_id}")
                    return True
                else:
                    logger.warning(f"Query {query_id} not found in history")
                    return False
            except Exception as e:
                logger.error(f"Error deleting query history from JSON storage: {str(e)}")
                return False
    
    def prune_old_history(self, days_to_keep: int = 365) -> int:
        """
        Prune history entries older than a specific number of days.
        
        Args:
            days_to_keep: Number of days of history to keep
            
        Returns:
            Number of pruned entries
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
        pruned_count = 0
        
        if self.use_sqlite:
            try:
                conn = sqlite3.connect(self.db_file)
                cursor = conn.cursor()
                
                # Find queries that haven't been run since the cutoff date
                cursor.execute('''
                SELECT query_id FROM queries 
                WHERE datetime(last_run) < datetime(?)
                ''', (cutoff_date.isoformat(),))
                
                old_query_ids = [row[0] for row in cursor.fetchall()]
                
                # Delete old queries and their retrieved IDs
                for query_id in old_query_ids:
                    cursor.execute('DELETE FROM retrieved_ids WHERE query_id = ?', (query_id,))
                    cursor.execute('DELETE FROM queries WHERE query_id = ?', (query_id,))
                    pruned_count += 1
                
                conn.commit()
                conn.close()
                
                logger.info(f"Pruned {pruned_count} old queries from history")
                return pruned_count
            except Exception as e:
                logger.error(f"Error pruning old history from SQLite: {str(e)}")
                return 0
        else:
            try:
                with open(self.json_file, 'r') as f:
                    data = json.load(f)
                
                # Find and delete old queries
                to_delete = []
                for query_id, query_data in data.items():
                    if 'last_run' in query_data:
                        last_run = datetime.datetime.fromisoformat(query_data['last_run'])
                        if last_run < cutoff_date:
                            to_delete.append(query_id)
                
                # Delete old queries
                for query_id in to_delete:
                    del data[query_id]
                    pruned_count += 1
                
                # Save updated data
                with open(self.json_file, 'w') as f:
                    json.dump(data, f)
                
                logger.info(f"Pruned {pruned_count} old queries from history")
                return pruned_count
            except Exception as e:
                logger.error(f"Error pruning old history from JSON storage: {str(e)}")
                return 0


class NCBIIncrementalClient:
    """
    Extension of NCBIClient that supports incremental updates.
    This is a facade class that wraps NCBIClient and adds incremental update functionality.
    """
    
    def __init__(self, ncbi_client, update_manager=None):
        """
        Initialize the incremental NCBI client.
        
        Args:
            ncbi_client: An instance of NCBIClient
            update_manager: Optional IncrementalUpdateManager instance
        """
        self.client = ncbi_client
        self.update_manager = update_manager or IncrementalUpdateManager()
    
    def search_pubmed_incremental(self, 
                                 query: str, 
                                 max_results: int = 100,
                                 days_overlap: int = 1,
                                 return_only_new: bool = True,
                                 **kwargs) -> Dict[str, Any]:
        """
        Search PubMed with incremental updates.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return
            days_overlap: Number of days to overlap with previous query
            return_only_new: Whether to return only new results
            **kwargs: Additional parameters for search_pubmed
            
        Returns:
            Dictionary with search results and metadata
        """
        # Build incremental query if there's previous history
        incremental_query, query_id = self.update_manager.build_incremental_query(
            query, 'pubmed', days_overlap
        )
        
        # Perform the search
        search_results = self.client.search_pubmed(
            incremental_query, 
            max_results=max_results,
            **kwargs
        )
        
        if not search_results or 'esearchresult' not in search_results:
            logger.error("Search failed or returned no results")
            return {
                'query': incremental_query,
                'query_id': query_id,
                'total_count': 0,
                'new_count': 0,
                'results': []
            }
        
        # Extract IDs from results
        result_ids = search_results['esearchresult'].get('idlist', [])
        total_count = int(search_results['esearchresult'].get('count', 0))
        
        # Identify new results
        new_ids = self.update_manager.identify_new_results(query_id, result_ids)
        
        # Register this query run
        self.update_manager.register_query(
            query, 'pubmed', result_ids, total_count
        )
        
        # Fetch abstracts for new results or all results
        ids_to_fetch = new_ids if return_only_new else result_ids
        
        if not ids_to_fetch:
            logger.info(f"No new results found for query: {query}")
            return {
                'query': incremental_query,
                'query_id': query_id,
                'total_count': total_count,
                'new_count': 0,
                'results': []
            }
        
        # Fetch abstracts (limit to max_results)
        abstracts = self.client.fetch_pubmed_abstracts(
            id_list=ids_to_fetch[:max_results]
        )
        
        return {
            'query': incremental_query,
            'query_id': query_id,
            'total_count': total_count,
            'new_count': len(new_ids),
            'results': abstracts or []
        }
    
    def search_and_update_knowledge_base(self, 
                                        query: str,
                                        kb_file: str,
                                        max_results: int = 100,
                                        days_overlap: int = 1,
                                        **kwargs) -> Dict[str, Any]:
        """
        Search PubMed and update a knowledge base file with new results.
        
        Args:
            query: The search query
            kb_file: Path to knowledge base file (JSON)
            max_results: Maximum number of results to return
            days_overlap: Number of days to overlap with previous query
            **kwargs: Additional parameters for search_pubmed
            
        Returns:
            Dictionary with update statistics
        """
        # Perform incremental search
        search_result = self.search_pubmed_incremental(
            query, 
            max_results=max_results,
            days_overlap=days_overlap,
            return_only_new=True,
            **kwargs
        )
        
        # Update knowledge base if we have new results
        if search_result['results']:
            self.update_manager.update_knowledge_base(
                search_result['query_id'],
                search_result['results'],
                kb_file
            )
        
        return {
            'query': search_result['query'],
            'query_id': search_result['query_id'],
            'total_count': search_result['total_count'],
            'new_count': search_result['new_count'],
            'updated_kb': kb_file,
            'update_time': datetime.datetime.now().isoformat()
        }
    
    def schedule_regular_updates(self,
                               query: str,
                               kb_file: str,
                               schedule: str = 'weekly',
                               max_results: int = 100,
                               **kwargs) -> Dict[str, Any]:
        """
        Schedule regular updates for a knowledge base.
        This method generates a crontab entry but doesn't actually install it.
        
        Args:
            query: The search query
            kb_file: Path to knowledge base file
            schedule: Update frequency ('daily', 'weekly', 'monthly')
            max_results: Maximum results per update
            **kwargs: Additional parameters for search
            
        Returns:
            Dictionary with scheduling information
        """
        # Generate a unique script name
        query_id = self.update_manager.get_query_id_by_text(query, 'pubmed')
        script_name = f"update_kb_{query_id[:8]}.py"
        script_path = os.path.join(self.update_manager.storage_dir, script_name)
        
        # Create update script
        with open(script_path, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
# Auto-generated update script for knowledge base
# Query: {query}
# KB File: {kb_file}
# Schedule: {schedule}

import os
import sys
import json
import logging
from datetime import datetime

# Add your import paths if needed
sys.path.append(os.path.expanduser("~"))

# Import required modules
from ncbi_api import NCBIClient
from incremental_updates import IncrementalUpdateManager, NCBIIncrementalClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename="{os.path.join(self.update_manager.storage_dir, f'update_log_{query_id[:8]}.log')}"
)

# Initialize clients
ncbi_client = NCBIClient()
update_manager = IncrementalUpdateManager()
incremental_client = NCBIIncrementalClient(ncbi_client, update_manager)

# Run update
result = incremental_client.search_and_update_knowledge_base(
    "{query}",
    "{kb_file}",
    max_results={max_results}
)

# Log results
logging.info(f"Update completed: {{result['new_count']}} new items added to {{result['updated_kb']}}")

# Save result to status file
status_file = "{os.path.join(self.update_manager.storage_dir, f'update_status_{query_id[:8]}.json')}"
with open(status_file, 'w') as f:
    json.dump({{
        "last_update": datetime.now().isoformat(),
        "query": "{query}",
        "kb_file": "{kb_file}",
        "result": result
    }}, f, indent=2)
''')
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Generate crontab entry
        cron_schedule = {
            'daily': '0 2 * * *',       # 2 AM every day
            'weekly': '0 2 * * 0',      # 2 AM every Sunday
            'monthly': '0 2 1 * *',     # 2 AM on the 1st of each month
        }.get(schedule.lower(), '0 2 * * 0')  # Default to weekly
        
        crontab_entry = f"{cron_schedule} {script_path}"
        
        return {
            'query_id': query_id,
            'script_path': script_path,
            'crontab_entry': crontab_entry,
            'schedule': schedule,
            'kb_file': kb_file
        }

# Example usage
if __name__ == "__main__":
    from ncbi_api import NCBIClient
    
    # Initialize clients
    ncbi_client = NCBIClient()
    update_manager = IncrementalUpdateManager()
    incremental_client = NCBIIncrementalClient(ncbi_client, update_manager)
    
    # Example incremental search
    result = incremental_client.search_pubmed_incremental(
        "community acquired pneumonia treatment",
        max_results=10
    )
    
    # Print results
    print(f"Query: {result['query']}")
    print(f"Total results: {result['total_count']}")
    print(f"New results: {result['new_count']}")
    
    if result['results']:
        for i, article in enumerate(result['results'], 1):
            print(f"\n{i}. {article['title']}")
            print(f"  PMID: {article['pmid']}")
            print(f"  Published: {article['publication_date']}")