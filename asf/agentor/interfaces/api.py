class APIInterface:
    """Interface for connecting agents with RESTful APIs."""
    
    def __init__(self, base_url, headers=None, auth=None):
        """
        Initialize an API interface.
        
        Args:
            base_url: Base URL for the API
            headers: Optional headers to include in requests
            auth: Optional authentication credentials
        """
        self.base_url = base_url
        self.headers = headers or {}
        self.auth = auth
        self.session = None
    
    def connect(self):
        """Establish a connection to the API."""
        import requests
        self.session = requests.Session()
        if self.auth:
            self.session.auth = self.auth
        self.session.headers.update(self.headers)
        return True
    
    def disconnect(self):
        """Close the connection to the API."""
        if self.session:
            self.session.close()
            self.session = None
    
    def get(self, endpoint, params=None):
        """Send a GET request to the API."""
        if not self.session:
            self.connect()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint, data=None, json=None):
        """Send a POST request to the API."""
        if not self.session:
            self.connect()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.post(url, data=data, json=json)
        response.raise_for_status()
        return response.json()
    
    # Add other HTTP methods as needed
