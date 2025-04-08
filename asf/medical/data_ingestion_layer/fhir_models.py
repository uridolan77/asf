import json
from typing import Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class FHIRPatient:
    """
    Simplified FHIR Patient Resource representation
    Following core FHIR Patient resource structure
    """
    resourceType: str = "Patient"
    id: str = None
    identifier: list = None
    active: bool = True
    name: list = None
    telecom: list = None
    gender: str = None
    birthDate: str = None
    address: list = None
    maritalStatus: Dict[str, str] = None
    communication: list = None

    @classmethod
    def create(
        cls, 
        patient_id: str, 
        first_name: str, 
        last_name: str, 
        gender: str, 
        birth_date: str
    ):
        """
        Factory method to create a patient resource
        
        :param patient_id: Unique patient identifier
        :param first_name: Patient's first name
        :param last_name: Patient's last name
        :param gender: Patient's gender
        :param birth_date: Patient's birth date (YYYY-MM-DD)
        :return: FHIRPatient instance
        """
        return cls(
            id=patient_id,
            name=[{
                "use": "official",
                "family": last_name,
                "given": [first_name]
            }],
            gender=gender,
            birthDate=birth_date
        )
    
    def to_json(self) -> str:
        """
        Convert patient resource to JSON
        
        :return: JSON representation of patient resource
        """
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FHIRPatient':
        """
        Create patient resource from JSON
        
        :param json_str: JSON string representing patient resource
        :return: FHIRPatient instance
        """
        data = json.loads(json_str)
        return cls(**data)

class FHIRClient:
    """
    Simplified FHIR Client for resource interactions
    """
    def __init__(self, base_url: str):
        """
        Initialize FHIR client
        
        :param base_url: Base URL of FHIR server
        """
        self.base_url = base_url
    
    def create_resource(self, resource: Any) -> Dict[str, Any]:
        """
        Create a new FHIR resource
        
        :param resource: FHIR resource to create
        :return: Server response
        """
        # Simulated create method
        return {
            "status": "created",
            "id": resource.id,
            "resourceType": resource.resourceType,
            "timestamp": datetime.now().isoformat()
        }
    
    def read_resource(self, resource_type: str, resource_id: str) -> Dict[str, Any]:
        """
        Read a specific FHIR resource
        
        :param resource_type: Type of resource
        :param resource_id: Unique resource identifier
        :return: Resource details
        """
        # Simulated read method
        return {
            "resourceType": resource_type,
            "id": resource_id
        }

def main():
    # Create a FHIR patient
    patient = FHIRPatient.create(
        patient_id="example-123",
        first_name="John",
        last_name="Doe",
        gender="male", 
        birth_date="1980-01-15"
    )
    
    # Convert to JSON
    patient_json = patient.to_json()
    print("Patient JSON:")
    print(patient_json)
    
    # Recreate from JSON
    reconstructed_patient = FHIRPatient.from_json(patient_json)
    
    # Simulate FHIR client interaction
    client = FHIRClient("https://example.fhirserver.org")
    
    # Create resource
    create_response = client.create_resource(patient)
    print("\nCreate Response:")
    print(json.dumps(create_response, indent=2))
    
    # Read resource
    read_response = client.read_resource("Patient", patient.id)
    print("\nRead Response:")
    print(json.dumps(read_response, indent=2))

if __name__ == "__main__":
    main()