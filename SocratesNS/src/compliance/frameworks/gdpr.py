import re
from src.compliance.frameworks.compliance_processor_base import ComplianceProcessorBase

# Framework-specific compliance processors
class GDPRComplianceProcessor:
    """GDPR-specific compliance processor"""
    
    def __init__(self, config):
        self.config = config
        self.data_protection_principles = self._initialize_principles()
        self.data_subject_rights = self._initialize_rights()
        
    def _initialize_principles(self):
        """Initialize GDPR data protection principles"""
        return {
            'lawfulness': {
                'description': 'Processing must be lawful, fair, and transparent',
                'articles': ['5(1)(a)', '6'],
                'severity': 'high',
                'keywords': ['lawful', 'legal basis', 'consent', 'contract', 'legal obligation', 
                           'vital interests', 'public task', 'legitimate interests']
            },
            'purpose_limitation': {
                'description': 'Data must be collected for specified, explicit, and legitimate purposes',
                'articles': ['5(1)(b)'],
                'severity': 'high',
                'keywords': ['purpose', 'specified purpose', 'explicit purpose', 'legitimate purpose']
            },
            'data_minimization': {
                'description': 'Data must be adequate, relevant, and limited to what is necessary',
                'articles': ['5(1)(c)'],
                'severity': 'medium',
                'keywords': ['minimization', 'adequate', 'relevant', 'limited', 'necessary']
            },
            'accuracy': {
                'description': 'Data must be accurate and kept up to date',
                'articles': ['5(1)(d)'],
                'severity': 'medium',
                'keywords': ['accuracy', 'accurate', 'up to date', 'rectification']
            },
            'storage_limitation': {
                'description': 'Data must be kept in a form that permits identification for no longer than necessary',
                'articles': ['5(1)(e)'],
                'severity': 'medium',
                'keywords': ['storage', 'retention', 'time-limited', 'deletion', 'erasure']
            },
            'integrity_confidentiality': {
                'description': 'Data must be processed securely',
                'articles': ['5(1)(f)', '32'],
                'severity': 'high',
                'keywords': ['security', 'integrity', 'confidentiality', 'secure processing', 
                           'encryption', 'pseudonymization']
            },
            'accountability': {
                'description': 'Controller must be able to demonstrate compliance',
                'articles': ['5(2)'],
                'severity': 'medium',
                'keywords': ['accountability', 'demonstrate', 'compliance', 'documentation']
            }
        }
    
    def _initialize_rights(self):
        """Initialize GDPR data subject rights"""
        return {
            'information': {
                'description': 'Right to be informed about collection and use of personal data',
                'articles': ['12', '13', '14'],
                'severity': 'medium',
                'keywords': ['information', 'informed', 'privacy notice', 'privacy policy']
            },
            'access': {
                'description': 'Right to access personal data and supplementary information',
                'articles': ['15'],
                'severity': 'high',
                'keywords': ['access', 'subject access request', 'SAR', 'obtain copy']
            },
            'rectification': {
                'description': 'Right to have inaccurate personal data rectified',
                'articles': ['16'],
                'severity': 'medium',
                'keywords': ['rectification', 'correction', 'accurate', 'update']
            },
            'erasure': {
                'description': 'Right to have personal data erased',
                'articles': ['17'],
                'severity': 'high',
                'keywords': ['erasure', 'deletion', 'right to be forgotten', 'remove']
            },
            'restriction': {
                'description': 'Right to restrict processing of personal data',
                'articles': ['18'],
                'severity': 'medium',
                'keywords': ['restriction', 'restrict processing', 'limit processing']
            },
            'portability': {
                'description': 'Right to data portability',
                'articles': ['20'],
                'severity': 'medium',
                'keywords': ['portability', 'obtain and reuse', 'transfer', 'machine readable']
            },
            'object': {
                'description': 'Right to object to processing of personal data',
                'articles': ['21'],
                'severity': 'high',
                'keywords': ['object', 'objection', 'stop processing', 'direct marketing']
            },
            'automated_decision': {
                'description': 'Rights related to automated decision making and profiling',
                'articles': ['22'],
                'severity': 'high',
                'keywords': ['automated decision', 'automated processing', 'profiling', 'human intervention']
            }
        }
    
    def verify_compliance(self, context, compliance_mode):
        """Verify GDPR compliance"""
        # Extract text from context
        text = None
        if context['content_type'] == 'text':
            text = context['content']
        elif isinstance(context['content'], dict) and 'text' in context['content']:
            text = context['content']['text']
        elif isinstance(context['content'], str):
            text = context['content']
            
        if not text:
            # No text to verify
            return {
                'is_compliant': True,
                'compliance_score': 1.0,
                'framework': 'GDPR',
                'metadata': {'reason': 'no_text_to_verify'}
            }
            
        # Check for PII presence
        has_pii = self._has_pii(context)
        
        if not has_pii:
            # No PII, GDPR doesn't apply
            return {
                'is_compliant': True,
                'compliance_score': 1.0,
                'framework': 'GDPR',
                'metadata': {'reason': 'no_pii_detected'}
            }
            
        # Check principles
        principle_results = self._check_principles(text, context)
        
        # Check rights
        rights_results = self._check_rights(text, context)
        
        # Check for specific violations
        violations = []
        
        # Add principle violations
        for principle, result in principle_results.items():
            if not result['is_compliant']:
                violations.append({
                    'type': 'gdpr_principle_violation',
                    'principle': principle,
                    'description': f"GDPR principle violation: {self.data_protection_principles[principle]['description']}",
                    'articles': self.data_protection_principles[principle]['articles'],
                    'severity': self.data_protection_principles[principle]['severity'],
                    'recommended_action': 'review'
                })
                
        # Add rights violations
        for right, result in rights_results.items():
            if not result['is_compliant']:
                violations.append({
                    'type': 'gdpr_right_violation',
                    'right': right,
                    'description': f"GDPR right violation: {self.data_subject_rights[right]['description']}",
                    'articles': self.data_subject_rights[right]['articles'],
                    'severity': self.data_subject_rights[right]['severity'],
                    'recommended_action': 'review'
                })
                
        # Calculate compliance score
        principle_scores = [result['compliance_score'] for result in principle_results.values()]
        rights_scores = [result['compliance_score'] for result in rights_results.values()]
        
        if principle_scores and rights_scores:
            # Weight principles more than rights (principles are more fundamental)
            overall_score = (0.7 * sum(principle_scores) / len(principle_scores) + 
                           0.3 * sum(rights_scores) / len(rights_scores))
        elif principle_scores:
            overall_score = sum(principle_scores) / len(principle_scores)
        elif rights_scores:
            overall_score = sum(rights_scores) / len(rights_scores)
        else:
            overall_score = 1.0
            
        # Determine compliance based on mode and score
        is_compliant = len(violations) == 0 or (
            compliance_mode == 'soft' and 
            not any(v['severity'] == 'high' for v in violations) and
            overall_score >= 0.7
        )
        
        return {
            'is_compliant': is_compliant,
            'compliance_score': overall_score,
            'framework': 'GDPR',
            'violations': violations if not is_compliant else [],
            'principle_results': principle_results,
            'right_results': rights_results,
            'has_pii': has_pii,
            'metadata': {
                'principle_count': len(principle_results),
                'right_count': len(rights_results),
                'violation_count': len(violations)
            }
        }
    
    def _has_pii(self, context):
        """Check if context contains PII"""
        # Check entities
        if 'entities' in context:
            for entity in context['entities']:
                if entity.get('type') == 'PII':
                    return True
                    
        # Check content directly if it's a short text
        if context['content_type'] == 'text' and isinstance(context['content'], str) and len(context['content']) < 1000:
            text = context['content']
            # Check for common PII patterns
            pii_patterns = [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
                r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',  # SSN
                r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # phone
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # date
                r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'  # credit card
            ]
            for pattern in pii_patterns:
                if re.search(pattern, text):
                    return True
                    
        return False
    
    def _check_principles(self, text, context):
        """Check GDPR principles"""
        results = {}
        
        for principle, data in self.data_protection_principles.items():
            # Check if principle is mentioned in text
            principle_keywords = data['keywords']
            keyword_matches = self._check_keywords(text, principle_keywords)
            
            if keyword_matches > 0:
                # Principle is mentioned, check if it's respected
                is_compliant, compliance_score = self._evaluate_principle_compliance(
                    principle, text, context, keyword_matches
                )
                
                results[principle] = {
                    'is_compliant': is_compliant,
                    'compliance_score': compliance_score,
                    'keyword_matches': keyword_matches,
                    'description': data['description'],
                    'articles': data['articles']
                }
                
        return results
    
    def _check_rights(self, text, context):
        """Check GDPR rights"""
        results = {}
        
        for right, data in self.data_subject_rights.items():
            # Check if right is mentioned in text
            right_keywords = data['keywords']
            keyword_matches = self._check_keywords(text, right_keywords)
            
            if keyword_matches > 0:
                # Right is mentioned, check if it's respected
                is_compliant, compliance_score = self._evaluate_right_compliance(
                    right, text, context, keyword_matches
                )
                
                results[right] = {
                    'is_compliant': is_compliant,
                    'compliance_score': compliance_score,
                    'keyword_matches': keyword_matches,
                    'description': data['description'],
                    'articles': data['articles']
                }
                
        return results
    
    def _check_keywords(self, text, keywords):
        """Check how many keywords are present in text"""
        count = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        return count
    
    def _evaluate_principle_compliance(self, principle, text, context, keyword_matches):
        """Evaluate compliance with a GDPR principle"""
        # This is a simplified implementation
        # A real implementation would have more sophisticated checks
        
        # Simple heuristic: more keyword matches suggest better compliance
        # but also depends on text length
        text_length = len(text.split())
        keyword_density = keyword_matches / max(1, text_length) * 1000  # keywords per 1000 words
        
        # Simple threshold-based evaluation
        if principle == 'lawfulness':
            # Check for legal basis
            legal_basis_keywords = ['consent', 'contract', 'legal obligation', 
                                   'vital interests', 'public task', 'legitimate interests']
            legal_basis_matches = self._check_keywords(text, legal_basis_keywords)
            if legal_basis_matches > 0:
                return True, min(1.0, 0.7 + 0.1 * min(3, legal_basis_matches))
            else:
                return False, 0.5
        elif principle == 'purpose_limitation':
            if keyword_density > 1.0:
                return True, min(1.0, 0.7 + 0.1 * keyword_density)
            else:
                return False, 0.6
        elif principle == 'data_minimization':
            if keyword_density > 0.8:
                return True, min(1.0, 0.7 + 0.15 * keyword_density)
            else:
                return False, 0.6
        elif principle == 'accuracy':
            if keyword_density > 0.5:
                return True, min(1.0, 0.8 + 0.1 * keyword_density)
            else:
                return False, 0.7
        elif principle == 'storage_limitation':
            if keyword_density > 0.8:
                return True, min(1.0, 0.7 + 0.1 * keyword_density)
            else:
                return False, 0.6
        elif principle == 'integrity_confidentiality':
            security_keywords = ['security', 'encryption', 'pseudonymization', 'measures']
            security_matches = self._check_keywords(text, security_keywords)
            if security_matches > 0:
                return True, min(1.0, 0.7 + 0.1 * min(3, security_matches))
            else:
                return False, 0.5
        elif principle == 'accountability':
            if keyword_density > 0.5:
                return True, min(1.0, 0.8 + 0.1 * keyword_density)
            else:
                return False, 0.7
                
        # Default
        return True, 0.8
    
    def _evaluate_right_compliance(self, right, text, context, keyword_matches):
        """Evaluate compliance with a GDPR right"""
        # This is a simplified implementation
        # A real implementation would have more sophisticated checks
        
        # Simple heuristic: more keyword matches suggest better compliance
        # but also depends on text length
        text_length = len(text.split())
        keyword_density = keyword_matches / max(1, text_length) * 1000  # keywords per 1000 words
        
        # Check if there's a sufficient explanation of the right
        explanation_threshold = {
            'information': 0.8,
            'access': 1.0,
            'rectification': 0.6,
            'erasure': 1.0,
            'restriction': 0.7,
            'portability': 0.8,
            'object': 1.0,
            'automated_decision': 1.2
        }
        
        threshold = explanation_threshold.get(right, 0.8)
        
        if keyword_density >= threshold:
            compliance_score = min(1.0, 0.7 + 0.15 * (keyword_density / threshold))
            return True, compliance_score
        else:
            compliance_score = max(0.4, 0.7 * (keyword_density / threshold))
            return False, compliance_score
