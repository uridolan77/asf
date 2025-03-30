import datetime
import uuid

class ComplianceAuditLog:
    """Log of compliance checks for audit purposes"""
    
    def __init__(self, max_entries=1000):
        self.max_entries = max_entries
        self.entries = []
        
    def log_compliance_check(self, content, content_type, context, compliance_mode, result):
        """Log a compliance check"""
        # Create log entry
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'content_type': content_type,
            'compliance_mode': compliance_mode,
            'is_compliant': result['is_compliant'],
            'compliance_score': result.get('compliance_score', 1.0 if result['is_compliant'] else 0.0),
            'violation_count': len(result.get('violations', [])),
            'context_summary': self._create_context_summary(context),
            'result_id': str(uuid.uuid4())
        }
        
        # Limit content size for logging
        if isinstance(content, str):
            entry['content_preview'] = content[:200] + '...' if len(content) > 200 else content
        elif isinstance(content, dict):
            entry['content_preview'] = str(content)[:200] + '...' if len(str(content)) > 200 else str(content)
        else:
            entry['content_preview'] = str(type(content))
            
        # Add entry to log
        self.entries.append(entry)
        
        # Remove oldest entries if log is too large
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
            
        return entry['result_id']
    
    def _create_context_summary(self, context):
        """Create summary of context for logging"""
        if not context:
            return {}
            
        # Extract key information for summary
        summary = {}
        
        # Include key context fields
        key_fields = ['content_type', 'timestamp', 'regulatory_frameworks']
        for field in key_fields:
            if field in context:
                summary[field] = context[field]
                
        # Count entities if present
        if 'entities' in context:
            summary['entity_count'] = len(context['entities'])
            
        # Count relations if present
        if 'relations' in context:
            summary['relation_count'] = len(context['relations'])
            
        # Count concepts if present
        if 'concepts' in context:
            summary['concept_count'] = len(context['concepts'])
            
        return summary
    
    def get_entries(self, limit=None, filter_func=None):
        """Get log entries with optional filtering"""
        if filter_func:
            filtered_entries = [entry for entry in self.entries if filter_func(entry)]
        else:
            filtered_entries = self.entries
            
        if limit:
            return filtered_entries[-limit:]
        else:
            return filtered_entries
            
    def get_entry_by_id(self, result_id):
        """Get specific log entry by ID"""
        for entry in self.entries:
            if entry.get('result_id') == result_id:
                return entry
        return None
    
    def get_compliance_stats(self):
        """Get compliance statistics from log"""
        if not self.entries:
            return {
                'total_checks': 0,
                'compliant_count': 0,
                'compliance_rate': 0.0,
                'average_score': 0.0
            }
            
        total_checks = len(self.entries)
        compliant_count = sum(1 for entry in self.entries if entry['is_compliant'])
        compliance_rate = compliant_count / total_checks if total_checks > 0 else 0.0
        average_score = sum(entry.get('compliance_score', 0.0) for entry in self.entries) / total_checks
        
        # Get violation distribution
        violations = {}
        for entry in self.entries:
            violation_count = entry.get('violation_count', 0)
            if violation_count not in violations:
                violations[violation_count] = 0
            violations[violation_count] += 1
            
        return {
            'total_checks': total_checks,
            'compliant_count': compliant_count,
            'compliance_rate': compliance_rate,
            'average_score': average_score,
            'violation_distribution': violations
        }
    
    def clear(self):
        """Clear all log entries"""
        self.entries = []

