from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
import datetime
import logging
import uuid
import json

class ReviewPriority(Enum):
    """Priority levels for human review"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class ReviewStatus(Enum):
    """Status states for review tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    

class ReviewTriggerType(Enum):
    """Types of triggers that can initiate human review"""
    LOW_CONFIDENCE = "low_confidence"          # Model confidence below threshold
    HIGH_RISK_DOMAIN = "high_risk_domain"      # Sensitive domain (healthcare, finance)
    REGULATORY_CONFLICT = "regulatory_conflict" # Conflict between regulatory frameworks
    NOVEL_PATTERN = "novel_pattern"            # Unfamiliar compliance pattern
    FLAG_FOR_REVIEW = "flag_for_review"        # Explicitly flagged for review
    RANDOM_AUDIT = "random_audit"              # Random sampling for quality assurance
    THRESHOLD_VIOLATION = "threshold_violation" # Compliance score below threshold
    EXECUTIVE_ACTION = "executive_action"       # Action requiring executive approval


@dataclass
class ComplianceReviewTask:
    """Task for human review of compliance decisions"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    generated_text: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    compliance_result: Dict[str, Any] = field(default_factory=dict)
    regulatory_frameworks: List[str] = field(default_factory=list)
    priority: ReviewPriority = ReviewPriority.MEDIUM
    status: ReviewStatus = ReviewStatus.PENDING
    trigger_type: ReviewTriggerType = ReviewTriggerType.FLAG_FOR_REVIEW
    trigger_details: Dict[str, Any] = field(default_factory=dict)
    reviewer_id: Optional[str] = None
    review_notes: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    action_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_action(self, action: str, user_id: str, details: Dict[str, Any] = None):
        """Add an action to the task history"""
        self.action_history.append({
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details or {}
        })
        self.updated_at = datetime.datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "generated_text": self.generated_text,
            "context": self.context,
            "compliance_result": self.compliance_result,
            "regulatory_frameworks": self.regulatory_frameworks,
            "priority": self.priority.name,
            "status": self.status.value,
            "trigger_type": self.trigger_type.value,
            "trigger_details": self.trigger_details,
            "reviewer_id": self.reviewer_id,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "action_history": self.action_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceReviewTask':
        """Create task from dictionary representation"""
        task = cls(
            id=data.get("id", str(uuid.uuid4())),
            prompt=data.get("prompt", ""),
            generated_text=data.get("generated_text", ""),
            context=data.get("context", {}),
            compliance_result=data.get("compliance_result", {}),
            regulatory_frameworks=data.get("regulatory_frameworks", []),
            priority=ReviewPriority[data.get("priority", "MEDIUM")],
            status=ReviewStatus(data.get("status", "pending")),
            trigger_type=ReviewTriggerType(data.get("trigger_type", "flag_for_review")),
            trigger_details=data.get("trigger_details", {}),
            reviewer_id=data.get("reviewer_id"),
            review_notes=data.get("review_notes", ""),
            action_history=data.get("action_history", [])
        )
        
        # Parse dates if they exist
        if "created_at" in data:
            task.created_at = datetime.datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            task.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            
        return task


class ComplianceReviewSystem:
    """
    Human-in-the-loop system for reviewing compliance decisions,
    managing workflows, and integrating human oversight into the
    compliance filtering process.
    """
    
    def __init__(self, 
                 compliance_system,
                 review_queue_backend=None,
                 notification_backend=None,
                 audit_logger=None):
        """
        Initialize the compliance review system.
        
        Args:
            compliance_system: The underlying compliance system (e.g., CompliantLanguageModelProcessor)
            review_queue_backend: Backend for storing and retrieving review tasks
            notification_backend: Backend for sending notifications to reviewers
            audit_logger: Logger for audit trail
        """
        self.compliance_system = compliance_system
        self.review_queue = review_queue_backend or InMemoryReviewQueue()
        self.notifier = notification_backend or ConsoleNotifier()
        self.audit_logger = audit_logger or logging.getLogger("compliance_audit")
        
        # Default trigger configuration
        self.trigger_config = {
            # Trigger human review if compliance score is below this threshold
            "compliance_score_threshold": 0.7,
            
            # High-risk domains requiring additional scrutiny
            "high_risk_domains": ["healthcare", "finance", "legal", "children", "military"],
            
            # Randomly sample this percentage of requests for quality assurance
            "random_audit_percentage": 5.0,
            
            # Frameworks with special review requirements
            "sensitive_frameworks": ["HIPAA", "GDPR", "FINRA", "PCI-DSS"],
            
            # Auto-escalation rules for high-risk situations
            "auto_escalation_rules": [
                {"condition": "compliance_score < 0.5 AND framework IN ['HIPAA', 'GDPR']", 
                 "priority": ReviewPriority.HIGH},
                {"condition": "entity_type IN ['PHI', 'PII'] AND action == 'disclosure'", 
                 "priority": ReviewPriority.CRITICAL}
            ]
        }
        
        # Register callbacks for notification events
        self.event_handlers = {
            "task_created": [],
            "task_claimed": [],
            "task_completed": [],
            "task_escalated": []
        }
    
    def process_with_review(self, 
                           prompt: str, 
                           context: Dict[str, Any] = None,
                           compliance_mode: str = "strict",
                           force_review: bool = False) -> Dict[str, Any]:
        """
        Process text generation with potential human review.
        
        Args:
            prompt: The input prompt
            context: Additional context information
            compliance_mode: 'strict' or 'soft' enforcement
            force_review: Force human review regardless of triggers
            
        Returns:
            Result dictionary with generation and review status
        """
        # First, process with the compliance system
        result = self.compliance_system.generate_compliant_text(
            prompt, 
            context=context,
            compliance_mode=compliance_mode
        )
        
        # Determine if human review is needed
        needs_review, trigger_info = self._check_review_triggers(
            prompt, result, context, force_review
        )
        
        if needs_review:
            # Create a review task
            task = self._create_review_task(
                prompt, result, context, trigger_info
            )
            
            # Add to review queue
            self.review_queue.add_task(task)
            
            # Notify about the new task
            self._notify_event("task_created", task)
            
            # Add review status to result
            result["review_status"] = {
                "needs_review": True,
                "review_id": task.id,
                "review_priority": task.priority.name,
                "review_trigger": task.trigger_type.value
            }
            
            # If high priority, consider blocking until review
            if task.priority in [ReviewPriority.HIGH, ReviewPriority.CRITICAL]:
                result["awaiting_review"] = True
                if trigger_info.get("potentially_harmful", False):
                    # Replace generated text with a holding message
                    result["original_text"] = result["text"]
                    result["text"] = "This content requires review before it can be shown."
        else:
            # No review needed
            result["review_status"] = {
                "needs_review": False
            }
            
        return result
    
    def get_pending_reviews(self, 
                            reviewer_id: Optional[str] = None, 
                            priority: Optional[ReviewPriority] = None,
                            limit: int = 100) -> List[ComplianceReviewTask]:
        """
        Get pending review tasks, optionally filtered by reviewer or priority.
        
        Args:
            reviewer_id: Filter by reviewer ID
            priority: Filter by review priority
            limit: Maximum number of tasks to return
            
        Returns:
            List of pending review tasks
        """
        filters = {"status": ReviewStatus.PENDING}
        if reviewer_id:
            filters["reviewer_id"] = reviewer_id
        if priority:
            filters["priority"] = priority
            
        return self.review_queue.get_tasks(filters, limit)
    
    def claim_review_task(self, task_id: str, reviewer_id: str) -> ComplianceReviewTask:
        """
        Claim a review task for a specific reviewer.
        
        Args:
            task_id: ID of the task to claim
            reviewer_id: ID of the reviewer claiming the task
            
        Returns:
            Updated review task
            
        Raises:
            ValueError: If the task cannot be claimed
        """
        # Get the task
        task = self.review_queue.get_task(task_id)
        if not task:
            raise ValueError(f"Review task {task_id} not found")
            
        # Check if task can be claimed
        if task.status != ReviewStatus.PENDING:
            raise ValueError(f"Task {task_id} cannot be claimed (status: {task.status.value})")
            
        # Update task
        task.status = ReviewStatus.IN_PROGRESS
        task.reviewer_id = reviewer_id
        task.add_action("claimed", reviewer_id)
        
        # Save updated task
        self.review_queue.update_task(task)
        
        # Notify about claim
        self._notify_event("task_claimed", task)
        
        return task
    
    def complete_review(self, 
                       task_id: str, 
                       reviewer_id: str,
                       approved: bool,
                       notes: str = "",
                       modified_text: Optional[str] = None) -> ComplianceReviewTask:
        """
        Complete a review task.
        
        Args:
            task_id: ID of the task to complete
            reviewer_id: ID of the reviewer
            approved: Whether the content is approved
            notes: Reviewer notes
            modified_text: Modified text if changes were made
            
        Returns:
            Updated review task
            
        Raises:
            ValueError: If the task cannot be completed
        """
        # Get the task
        task = self.review_queue.get_task(task_id)
        if not task:
            raise ValueError(f"Review task {task_id} not found")
            
        # Check if task can be completed
        if task.status != ReviewStatus.IN_PROGRESS:
            raise ValueError(f"Task {task_id} cannot be completed (status: {task.status.value})")
            
        if task.reviewer_id != reviewer_id:
            raise ValueError(f"Task {task_id} is assigned to {task.reviewer_id}, not {reviewer_id}")
            
        # Update task
        task.status = ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED
        task.review_notes = notes
        task.add_action(
            "approved" if approved else "rejected", 
            reviewer_id,
            {"notes": notes, "modified_text": modified_text is not None}
        )
        
        # If text was modified, record the change
        if modified_text is not None:
            task.generated_text = modified_text
        
        # Save updated task
        self.review_queue.update_task(task)
        
        # Notify about completion
        self._notify_event("task_completed", task)
        
        return task
    
    def escalate_review(self,
                        task_id: str,
                        reviewer_id: str,
                        escalation_reason: str,
                        escalation_level: str = "manager") -> ComplianceReviewTask:
        """
        Escalate a review task to a higher level.
        
        Args:
            task_id: ID of the task to escalate
            reviewer_id: ID of the reviewer escalating the task
            escalation_reason: Reason for escalation
            escalation_level: Level to escalate to
            
        Returns:
            Updated review task
            
        Raises:
            ValueError: If the task cannot be escalated
        """
        # Get the task
        task = self.review_queue.get_task(task_id)
        if not task:
            raise ValueError(f"Review task {task_id} not found")
            
        # Escalate task
        task.status = ReviewStatus.ESCALATED
        task.add_action(
            "escalated", 
            reviewer_id, 
            {"reason": escalation_reason, "level": escalation_level}
        )
        
        # Increase priority
        if task.priority != ReviewPriority.CRITICAL:
            task.priority = ReviewPriority(task.priority.value + 1)
        
        # Save updated task
        self.review_queue.update_task(task)
        
        # Notify about escalation
        self._notify_event("task_escalated", task)
        
        return task
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about review tasks.
        
        Returns:
            Dictionary with review statistics
        """
        # Get tasks
        all_tasks = self.review_queue.get_all_tasks()
        
        # Calculate basic statistics
        stats = {
            "total_tasks": len(all_tasks),
            "by_status": {},
            "by_priority": {},
            "by_trigger_type": {},
            "by_framework": {},
            "review_time_avg_seconds": 0,
            "approval_rate": 0
        }
        
        # Count by status
        for status in ReviewStatus:
            stats["by_status"][status.value] = len([
                t for t in all_tasks if t.status == status
            ])
            
        # Count by priority
        for priority in ReviewPriority:
            stats["by_priority"][priority.name] = len([
                t for t in all_tasks if t.priority == priority
            ])
            
        # Count by trigger type
        for trigger_type in ReviewTriggerType:
            stats["by_trigger_type"][trigger_type.value] = len([
                t for t in all_tasks if t.trigger_type == trigger_type
            ])
            
        # Count by framework
        framework_counts = {}
        for task in all_tasks:
            for framework in task.regulatory_frameworks:
                if framework not in framework_counts:
                    framework_counts[framework] = 0
                framework_counts[framework] += 1
        stats["by_framework"] = framework_counts
        
        # Calculate review time for completed tasks
        completed_tasks = [
            t for t in all_tasks 
            if t.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED]
        ]
        
        if completed_tasks:
            total_time = 0
            for task in completed_tasks:
                # Find claim and completion timestamps
                claim_time = None
                completion_time = None
                
                for action in task.action_history:
                    if action["action"] == "claimed":
                        claim_time = datetime.datetime.fromisoformat(action["timestamp"])
                    elif action["action"] in ["approved", "rejected"]:
                        completion_time = datetime.datetime.fromisoformat(action["timestamp"])
                
                if claim_time and completion_time:
                    review_time = (completion_time - claim_time).total_seconds()
                    total_time += review_time
            
            stats["review_time_avg_seconds"] = total_time / len(completed_tasks)
            
            # Calculate approval rate
            approved_count = stats["by_status"].get(ReviewStatus.APPROVED.value, 0)
            stats["approval_rate"] = approved_count / len(completed_tasks)
            
        return stats
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """
        Register a handler for review system events.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event occurs
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def update_trigger_config(self, config_updates: Dict[str, Any]):
        """
        Update the trigger configuration.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        self.trigger_config.update(config_updates)
        
    def _check_review_triggers(self, 
                             prompt: str, 
                             result: Dict[str, Any], 
                             context: Dict[str, Any],
                             force_review: bool) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if human review is needed based on triggers.
        
        Returns:
            Tuple of (needs_review, trigger_info)
        """
        trigger_info = {}
        
        # Forced review
        if force_review:
            trigger_info["trigger"] = ReviewTriggerType.FLAG_FOR_REVIEW
            trigger_info["reason"] = "Explicitly flagged for review"
            return True, trigger_info
        
        # Check compliance score
        compliance_score = result.get("compliance_score", 1.0)
        threshold = self.trigger_config["compliance_score_threshold"]
        if compliance_score < threshold:
            trigger_info["trigger"] = ReviewTriggerType.THRESHOLD_VIOLATION
            trigger_info["reason"] = f"Compliance score {compliance_score} below threshold {threshold}"
            trigger_info["compliance_score"] = compliance_score
            return True, trigger_info
            
        # Check high-risk domains
        domain = context.get("domain") if context else None
        if domain and domain in self.trigger_config["high_risk_domains"]:
            trigger_info["trigger"] = ReviewTriggerType.HIGH_RISK_DOMAIN
            trigger_info["reason"] = f"Content in high-risk domain: {domain}"
            trigger_info["domain"] = domain
            return True, trigger_info
            
        # Check sensitive frameworks
        frameworks = result.get("compliance_metadata", {}).get("frameworks", [])
        sensitive_frameworks = [
            f for f in frameworks 
            if f in self.trigger_config["sensitive_frameworks"]
        ]
        if sensitive_frameworks:
            trigger_info["trigger"] = ReviewTriggerType.HIGH_RISK_DOMAIN
            trigger_info["reason"] = f"Content involves sensitive frameworks: {', '.join(sensitive_frameworks)}"
            trigger_info["frameworks"] = sensitive_frameworks
            return True, trigger_info
            
        # Check for regulatory conflicts
        violations = result.get("violations", [])
        if len(violations) > 0:
            # Check for conflicts between frameworks
            framework_violations = {}
            for violation in violations:
                framework = violation.get("framework", "unknown")
                if framework not in framework_violations:
                    framework_violations[framework] = []
                framework_violations[framework].append(violation)
                
            if len(framework_violations) > 1:
                trigger_info["trigger"] = ReviewTriggerType.REGULATORY_CONFLICT
                trigger_info["reason"] = f"Conflicts detected between frameworks: {', '.join(framework_violations.keys())}"
                trigger_info["conflicts"] = framework_violations
                return True, trigger_info
            
        # Random audit
        import random
        if random.random() * 100 < self.trigger_config["random_audit_percentage"]:
            trigger_info["trigger"] = ReviewTriggerType.RANDOM_AUDIT
            trigger_info["reason"] = "Randomly selected for quality assurance review"
            return True, trigger_info
            
        # No triggers matched
        return False, {}
    
    def _create_review_task(self, 
                         prompt: str, 
                         result: Dict[str, Any], 
                         context: Dict[str, Any],
                         trigger_info: Dict[str, Any]) -> ComplianceReviewTask:
        """Create a review task based on generation result and trigger info"""
        # Extract relevant information
        frameworks = result.get("compliance_metadata", {}).get("frameworks", [])
        
        # Determine priority based on trigger and auto-escalation rules
        priority = self._determine_review_priority(trigger_info, result, context)
        
        # Create the task
        task = ComplianceReviewTask(
            prompt=prompt,
            generated_text=result.get("text", ""),
            context=context or {},
            compliance_result=result,
            regulatory_frameworks=frameworks,
            priority=priority,
            trigger_type=ReviewTriggerType(trigger_info.get("trigger", 
                                            ReviewTriggerType.FLAG_FOR_REVIEW).value 
                                      if isinstance(trigger_info.get("trigger"), ReviewTriggerType) 
                                      else trigger_info.get("trigger", "flag_for_review")),
            trigger_details=trigger_info
        )
        
        # Log task creation
        self.audit_logger.info(
            f"Created review task {task.id} with priority {priority.name}, "
            f"trigger: {task.trigger_type.value}"
        )
        
        return task
    
    def _determine_review_priority(self, 
                                trigger_info: Dict[str, Any], 
                                result: Dict[str, Any],
                                context: Dict[str, Any]) -> ReviewPriority:
        """Determine review priority based on triggers and escalation rules"""
        # Default based on trigger type
        trigger = trigger_info.get("trigger")
        
        # Map trigger types to default priorities
        if trigger == ReviewTriggerType.THRESHOLD_VIOLATION:
            # Lower score = higher priority
            score = trigger_info.get("compliance_score", 0.5)
            if score < 0.3:
                return ReviewPriority.CRITICAL
            elif score < 0.5:
                return ReviewPriority.HIGH
            else:
                return ReviewPriority.MEDIUM
        elif trigger == ReviewTriggerType.HIGH_RISK_DOMAIN:
            return ReviewPriority.HIGH
        elif trigger == ReviewTriggerType.REGULATORY_CONFLICT:
            return ReviewPriority.HIGH
        elif trigger == ReviewTriggerType.FLAG_FOR_REVIEW:
            return ReviewPriority.MEDIUM
        elif trigger == ReviewTriggerType.RANDOM_AUDIT:
            return ReviewPriority.LOW
        
        # Check auto-escalation rules
        for rule in self.trigger_config.get("auto_escalation_rules", []):
            # Simple rule evaluator
            # In a real system, would use a proper expression evaluator
            # This is a placeholder implementation
            condition = rule.get("condition", "")
            if "compliance_score < " in condition:
                threshold = float(condition.split("compliance_score < ")[1].split(" ")[0])
                score = result.get("compliance_score", 1.0)
                if score < threshold:
                    return rule.get("priority", ReviewPriority.MEDIUM)
                    
        # Default priority
        return ReviewPriority.MEDIUM
    
    def _notify_event(self, event_type: str, task: ComplianceReviewTask):
        """Notify handlers about an event"""
        # Send to notification backend
        self.notifier.send_notification(event_type, task)
        
        # Call registered handlers
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(task)
            except Exception as e:
                self.audit_logger.error(f"Error in event handler for {event_type}: {str(e)}")


class InMemoryReviewQueue:
    """Simple in-memory implementation of review queue"""
    
    def __init__(self):
        self.tasks = {}
        
    def add_task(self, task: ComplianceReviewTask):
        """Add a task to the queue"""
        self.tasks[task.id] = task
        
    def get_task(self, task_id: str) -> Optional[ComplianceReviewTask]:
        """Get a task by ID"""
        return self.tasks.get(task_id)
        
    def update_task(self, task: ComplianceReviewTask):
        """Update a task in the queue"""
        self.tasks[task.id] = task
        
    def get_tasks(self, filters: Dict[str, Any], limit: int = 100) -> List[ComplianceReviewTask]:
        """Get tasks matching filters"""
        results = []
        for task in self.tasks.values():
            match = True
            for key, value in filters.items():
                task_value = getattr(task, key, None)
                if task_value != value:
                    match = False
                    break
            if match:
                results.append(task)
                if len(results) >= limit:
                    break
        return results
    
    def get_all_tasks(self) -> List[ComplianceReviewTask]:
        """Get all tasks"""
        return list(self.tasks.values())


class ConsoleNotifier:
    """Simple console-based notifier for testing"""
    
    def send_notification(self, event_type: str, task: ComplianceReviewTask):
        """Send a notification to the console"""
        print(f"[NOTIFICATION] {event_type}: Task {task.id} ({task.priority.name})")
        if event_type == "task_created":
            print(f"  Trigger: {task.trigger_type.value}")
            print(f"  Reason: {task.trigger_details.get('reason', 'Not specified')}")
        elif event_type == "task_completed":
            status = "approved" if task.status == ReviewStatus.APPROVED else "rejected"
            print(f"  Status: {status}")
            print(f"  Reviewer: {task.reviewer_id}")
            if task.review_notes:
                print(f"  Notes: {task.review_notes}")


# Example usage:
# compliance_system = CompliantLanguageModelProcessor(...)
# review_system = ComplianceReviewSystem(compliance_system)

# # Process with potential review
# result = review_system.process_with_review(
#     "Please generate a document about patient data handling...",
#     context={"domain": "healthcare"}
# )

# # Check if review was triggered
# if result.get("review_status", {}).get("needs_review", False):
#     print(f"Review requested: {result['review_status']['review_id']}")

# # Get pending reviews
# pending_reviews = review_system.get_pending_reviews()
# for task in pending_reviews:
#     print(f"Pending review: {task.id} ({task.priority.name})")

# # Claim and complete a review
# if pending_reviews:
#     task = review_system.claim_review_task(pending_reviews[0].id, "reviewer1")
#     review_system.complete_review(
#         task.id, 
#         "reviewer1", 
#         approved=True, 
#         notes="Content is compliant with healthcare regulations."
#     )

# # Get review statistics
# stats = review_system.get_review_statistics()
# print(f"Total reviews: {stats['total_tasks']}")
# print(f"Approval rate: {stats['approval_rate'] * 100:.1f}%")
