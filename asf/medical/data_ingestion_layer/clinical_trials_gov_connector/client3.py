import requests
import json
import time
import logging
from urllib.parse import urljoin
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClinicalTrialsGovConnector:
    """
    A Python connector for the ClinicalTrials.gov API v2.

    Provides methods to search for studies and retrieve details for specific studies.
    Handles pagination, field selection, and basic error handling.
    Optionally converts search results to a Pandas DataFrame.

    API Documentation: https://clinicaltrials.gov/data-api/api
    """
    BASE_URL = "https://clinicaltrials.gov/api/v2/"
    DEFAULT_PAGE_SIZE = 100 # API default is 100, max is 1000
    MAX_PAGE_SIZE = 1000
    DEFAULT_TIMEOUT = 30 # seconds

    def __init__(self, timeout=DEFAULT_TIMEOUT, user_agent="Python ClinicalTrialsGov Connector"):
        """
        Initializes the connector.

        Args:
            timeout (int): Request timeout in seconds.
            user_agent (str): User-Agent string for requests.
        """
        self.base_url = self.BASE_URL
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        logging.info("ClinicalTrialsGovConnector initialized.")

    def _make_request(self, endpoint, params=None):
        """
        Internal method to make GET requests to the API.

        Args:
            endpoint (str): API endpoint path (e.g., 'studies').
            params (dict, optional): Dictionary of query parameters.

        Returns:
            dict: Parsed JSON response from the API.

        Raises:
            requests.exceptions.RequestException: For network-related errors.
            ValueError: For non-200 status codes or JSON decoding errors.
        """
        url = urljoin(self.base_url, endpoint)
        params = params or {}

        # Filter out None values from params
        params = {k: v for k, v in params.items() if v is not None}

        logging.debug(f"Making request to {url} with params: {params}")
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            # Handle potential empty response body for certain successful status codes
            if response.status_code == 204: # No Content
                 logging.warning(f"Received 204 No Content for {url} with params: {params}")
                 return {} # Return empty dict for no content

            # Check if content type is JSON before attempting to decode
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                 logging.error(f"Unexpected Content-Type: {content_type} for URL: {url}")
                 raise ValueError(f"Expected JSON response, but got Content-Type: {content_type}. Response text: {response.text[:500]}...") # Show beginning of text

            data = response.json()
            logging.debug(f"Successfully received response from {url}")
            return data

        except requests.exceptions.Timeout as e:
            logging.error(f"Request timed out for {url}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error occurred for {url}: {e. Status code: {e.response.status_code}}. Response: {e.response.text[:500]}")
            raise ValueError(f"API request failed with status {e.response.status_code}: {e.response.text[:500]}") from e
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred for {url}: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON response from {url}. Response text: {response.text[:500]}... Error: {e}")
            raise ValueError("Failed to decode JSON response from API.") from e

    def get_study_fields(self, fields=None):
        """
        Retrieves a list of available study data fields.

        Args:
            fields (list, optional): Specific fields to get metadata for.
                                     Defaults to None (all fields).

        Returns:
            dict: API response containing field metadata.
        """
        params = {}
        if fields:
            params['fields'] = ",".join(fields)
        endpoint = "studies/metadata"
        logging.info(f"Fetching study field metadata...")
        return self._make_request(endpoint, params=params)

    def get_study_details(self, nct_id, fields=None):
        """
        Retrieves detailed information for a single study by its NCT ID.

        Args:
            nct_id (str): The NCT ID of the study (e.g., "NCT04368728").
            fields (list, optional): A list of specific fields to retrieve.
                                     If None, retrieves a default set defined by the API.
                                     See API docs for available fields.

        Returns:
            dict: A dictionary containing the detailed study data, or None if not found/error.
                  The structure matches the API's JSON response for a single study.
        """
        if not nct_id or not isinstance(nct_id, str):
            logging.error("Invalid NCT ID provided.")
            raise ValueError("A valid NCT ID (string) is required.")

        endpoint = f"studies/{nct_id}"
        params = {}
        if fields:
            params['fields'] = ",".join(fields)

        logging.info(f"Fetching details for study {nct_id} with fields: {fields or 'default'}")
        try:
            return self._make_request(endpoint, params=params)
        except ValueError as e:
             # Handle 404 Not Found specifically
            if "404" in str(e):
                 logging.warning(f"Study {nct_id} not found.")
                 return None
            else:
                 # Re-raise other ValueErrors (like bad requests, JSON errors)
                 raise

    def search_studies(self,
                       query_term=None,
                       filter_ids=None,
                       filter_overall_status=None,
                       filter_geo=None,
                       filter_sponsor=None,
                       # Add more filter parameters here based on API docs (e.g., filter.studyType, filter.phase)
                       sort=None,
                       fields=None,
                       page_size=DEFAULT_PAGE_SIZE,
                       max_results=None,
                       get_all=False):
        """
        Searches for clinical studies based on specified criteria.

        Args:
            query_term (str, optional): General search query (keywords, conditions, etc.).
                                        Maps to 'query.term'.
            filter_ids (list, optional): List of NCT IDs to filter by. Maps to 'filter.ids'.
            filter_overall_status (list, optional): List of study statuses to filter by
                                                    (e.g., ["RECRUITING", "COMPLETED"]).
                                                    Maps to 'filter.overallStatus'.
            filter_geo (str, optional): Geographic filter (e.g., "distance(40.71,-74.01,50 miles)").
                                        Maps to 'filter.geo'. See API docs for format.
            filter_sponsor (str, optional): Filter by sponsor name. Maps to 'filter.sponsor'.
            sort (str, optional): Field to sort results by (e.g., "lastUpdatePostDate:desc").
                                  Maps to 'sort'.
            fields (list, optional): List of specific fields to retrieve for each study.
                                     Reduces payload size. If None, API returns default fields.
            page_size (int, optional): Number of results per page (max 1000).
            max_results (int, optional): Maximum total number of results to return.
                                         Overrides get_all if set.
            get_all (bool, optional): If True, fetches all pages of results until no
                                      more pages are available or max_results is hit.
                                      Defaults to False (fetches only the first page).

        Returns:
            list: A list of dictionaries, where each dictionary represents a study
                  matching the search criteria. Returns only the 'study' objects.
                  Returns an empty list if no studies are found or an error occurs during
                  the *initial* request. Subsequent page errors might result in partial data.
        """
        endpoint = "studies"
        params = {
            'format': 'json',
            'pageSize': min(page_size, self.MAX_PAGE_SIZE),
            'query.term': query_term,
            'filter.ids': ",".join(filter_ids) if filter_ids else None,
            'filter.overallStatus': ",".join(filter_overall_status) if filter_overall_status else None,
            'filter.geo': filter_geo,
            'filter.sponsor': filter_sponsor,
            'sort': sort,
            'fields': ",".join(fields) if fields else None,
            'pageToken': None # Start with no page token
        }

        all_studies = []
        total_fetched = 0
        page_count = 0

        logging.info(f"Starting study search with parameters: { {k:v for k,v in params.items() if v is not None and k != 'pageToken'} }")

        while True:
            page_count += 1
            logging.info(f"Fetching page {page_count}...")
            try:
                response_data = self._make_request(endpoint, params=params)
            except ValueError as e:
                 # Log error but allow returning potentially partial data if get_all=True and previous pages worked
                 logging.error(f"Error fetching page {page_count}: {e}. Returning fetched data so far.")
                 break # Exit loop on error

            # Check if response_data is valid and contains 'studies'
            if not isinstance(response_data, dict) or 'studies' not in response_data:
                logging.warning(f"Invalid or empty response structure on page {page_count}. Response: {response_data}")
                break # Stop if the response format is unexpected

            studies_on_page = response_data.get('studies', [])
            num_on_page = len(studies_on_page)
            logging.info(f"Fetched {num_on_page} studies on page {page_count}.")

            all_studies.extend(studies_on_page)
            total_fetched += num_on_page

            # --- Check stopping conditions ---
            # 1. Max results limit reached
            if max_results is not None and total_fetched >= max_results:
                logging.info(f"Reached max_results limit ({max_results}).")
                # Trim excess results if needed
                all_studies = all_studies[:max_results]
                break

            # 2. Check for next page token
            next_page_token = response_data.get('nextPageToken')
            if next_page_token and get_all:
                params['pageToken'] = next_page_token
                # Optional: add a small delay to be polite to the API
                # time.sleep(0.1)
            else:
                # No next token or get_all is False
                if not next_page_token and get_all:
                    logging.info("No more pages available.")
                elif not get_all:
                    logging.info("Fetching only the first page as requested.")
                break # Exit loop

        logging.info(f"Search complete. Total studies fetched: {len(all_studies)}")
        return all_studies

    def search_studies_to_dataframe(self, *args, **kwargs):
        """
        Performs a study search and returns the results as a Pandas DataFrame.

        Requires Pandas to be installed.

        Args:
            *args, **kwargs: Arguments passed directly to the `search_studies` method.

        Returns:
            pandas.DataFrame: A DataFrame containing the search results,
                              or an empty DataFrame if no results or Pandas is not available.
        """
        if not PANDAS_AVAILABLE:
            logging.error("Pandas library is not installed. Cannot convert to DataFrame.")
            print("Please install pandas: pip install pandas")
            return pd.DataFrame() # Return empty DataFrame

        search_results = self.search_studies(*args, **kwargs)

        if not search_results:
            logging.info("No search results found or an error occurred during search.")
            return pd.DataFrame() # Return empty DataFrame if list is empty

        # The 'studies' list contains dicts. Each dict usually has a nested structure.
        # Often, key info is under 'protocolSection'. We extract relevant parts.
        # This might need adjustment based on the specific 'fields' requested
        # and the desired structure of the DataFrame.
        processed_data = []
        for study_wrapper in search_results:
            # The main study data is usually nested inside the first key (e.g. 'study')
            # Let's try to handle if the structure is flat or nested
            if 'study' in study_wrapper:
                 study = study_wrapper['study']
            else:
                 # Assume flat structure if 'study' key isn't present (less common for search results)
                 study = study_wrapper

            # Extract common useful fields (adjust based on your needs and requested 'fields')
            # Use .get() to avoid KeyError if a field is missing
            protocol = study.get('protocolSection', {})
            results = study.get('resultsSection', {})
            derived = study.get('derivedSection', {})
            annotation = study.get('annotationSection', {})
            doc = study.get('documentSection', {})

            record = {
                'NCTID': protocol.get('identificationModule', {}).get('nctId'),
                'BriefTitle': protocol.get('identificationModule', {}).get('briefTitle'),
                'OfficialTitle': protocol.get('identificationModule', {}).get('officialTitle'),
                'OverallStatus': protocol.get('statusModule', {}).get('overallStatus'),
                'StudyFirstPostDate': protocol.get('statusModule', {}).get('studyFirstPostDateStruct',{}).get('date'),
                'LastUpdatePostDate': protocol.get('statusModule', {}).get('lastUpdatePostDateStruct',{}).get('date'),
                'Phase': "|".join(protocol.get('designModule', {}).get('phases', [])) if protocol.get('designModule', {}).get('phases') else None,
                'StudyType': protocol.get('designModule', {}).get('studyType'),
                'LeadSponsor': protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name'),
                'Conditions': "|".join(protocol.get('conditionsModule', {}).get('conditions', [])) if protocol.get('conditionsModule', {}).get('conditions') else None,
                'Interventions': "|".join([f"{i.get('type', '')}: {i.get('name', '')}" for i in protocol.get('armsInterventionsModule', {}).get('interventions', [])]) if protocol.get('armsInterventionsModule', {}).get('interventions') else None,
                'Locations': "|".join([f"{loc.get('city')}, {loc.get('state')}, {loc.get('country')}" for loc in derived.get('locationsModule', {}).get('locations', []) if loc.get('country')]) if derived.get('locationsModule', {}).get('locations') else None,
                'EnrollmentCount': protocol.get('designModule', {}).get('enrollmentInfo', {}).get('count'),
                 # Add more fields as needed by inspecting the API response or your 'fields' list
                 # Example: 'StartDate': protocol.get('statusModule', {}).get('startDateStruct',{}).get('date'),
                 # Example: 'PrimaryCompletionDate': protocol.get('statusModule', {}).get('primaryCompletionDateStruct',{}).get('date'),
                 # Example: 'EligibilityCriteria': protocol.get('eligibilityModule', {}).get('eligibilityCriteria'),
            }
            processed_data.append(record)

        df = pd.DataFrame(processed_data)
        logging.info(f"Converted {len(df)} studies to Pandas DataFrame.")
        return df

# --- Example Usage ---
if __name__ == "__main__":
    connector = ClinicalTrialsGovConnector()

    # --- Example 1: Get details for a specific study ---
    print("\n--- Example 1: Get Study Details ---")
    nct_id_to_fetch = "NCT04368728" # Example COVID-19 Vaccine Trial
    # Select specific fields to retrieve
    specific_fields = [
        "NCTId", "BriefTitle", "OverallStatus", "StudyFirstPostDate", "LastUpdatePostDate",
        "Phase", "LeadSponsorName", "Condition", "InterventionName", "LocationCountry",
        "LocationState", "LocationCity", "EnrollmentCount"
    ]
    study_details = connector.get_study_details(nct_id_to_fetch, fields=specific_fields)
    if study_details:
        # Print selected details (adjust keys based on actual response structure or requested fields)
        study_protocol = study_details.get('study', {}).get('protocolSection', {})
        study_derived = study_details.get('study', {}).get('derivedSection', {})
        print(f"Details for {study_protocol.get('identificationModule', {}).get('nctId')}:")
        print(f"  Title: {study_protocol.get('identificationModule', {}).get('briefTitle')}")
        print(f"  Status: {study_protocol.get('statusModule', {}).get('overallStatus')}")
        print(f"  Sponsor: {study_protocol.get('sponsorCollaboratorsModule', {}).get('leadSponsor', {}).get('name')}")
        print(f"  Phases: {study_protocol.get('designModule', {}).get('phases')}")
        print(f"  Conditions: {protocol.get('conditionsModule', {}).get('conditions', [])[:5]}...") # Print first 5 conditions
        # Location info is often in derivedSection
        locations = study_derived.get('locationsModule', {}).get('locations', [])
        print(f"  Locations ({len(locations)}): {locations[:3]}...") # Print first 3 locations
        print("-" * 20)
    else:
        print(f"Could not retrieve details for {nct_id_to_fetch}.")


    # --- Example 2: Search for studies (first page only) ---
    print("\n--- Example 2: Search Studies (First Page) ---")
    # Search for recruiting breast cancer studies in the USA
    search_results_page1 = connector.search_studies(
        query_term="breast cancer",
        filter_overall_status=["RECRUITING"],
        filter_geo="country(United States)", # Filter by country
        fields=["NCTId", "BriefTitle", "OverallStatus"], # Request fewer fields
        page_size=5 # Small page size for demo
    )
    print(f"Found {len(search_results_page1)} studies on the first page:")
    for study_wrapper in search_results_page1:
        study = study_wrapper.get('study', {}) # Adjust if structure differs
        protocol = study.get('protocolSection', {})
        print(f"  - {protocol.get('identificationModule', {}).get('nctId')}: {protocol.get('identificationModule', {}).get('briefTitle')} ({protocol.get('statusModule', {}).get('overallStatus')})")
    print("-" * 20)


    # --- Example 3: Search for studies and get all results (up to a limit) ---
    print("\n--- Example 3: Search Studies (All Pages, Max 50 Results) ---")
    # Search for completed studies related to 'ozempic'
    all_ozempic_studies = connector.search_studies(
        query_term="ozempic OR semaglutide",
        filter_overall_status=["COMPLETED"],
        fields=["NCTId", "BriefTitle", "CompletionDate", "OverallStatus"],
        max_results=50, # Set a max limit
        page_size=20,   # Use a smaller page size for demo pagination
        get_all=True    # Fetch all pages up to max_results
    )
    print(f"Found a total of {len(all_ozempic_studies)} completed studies (up to 50):")
    # Print first 5 results
    for study_wrapper in all_ozempic_studies[:5]:
        study = study_wrapper.get('study', {})
        protocol = study.get('protocolSection', {})
        status_module = protocol.get('statusModule', {})
        completion_date = status_module.get('primaryCompletionDateStruct', {}).get('date') if status_module.get('primaryCompletionDateStruct') else status_module.get('completionDateStruct',{}).get('date')

        print(f"  - {protocol.get('identificationModule', {}).get('nctId')}: {protocol.get('identificationModule', {}).get('briefTitle')} (Completed: {completion_date})")
    print("  ...")
    print("-" * 20)

    # --- Example 4: Search and convert to Pandas DataFrame ---
    if PANDAS_AVAILABLE:
        print("\n--- Example 4: Search and Convert to DataFrame ---")
        # Search for Interventional studies for Diabetes Type 2, Phase 3
        df_diabetes = connector.search_studies_to_dataframe(
            query_term="Diabetes Mellitus, Type 2",
            filter_overall_status=["RECRUITING", "ACTIVE_NOT_RECRUITING"],
            # Example adding another filter: studyType
            # Note: Check API docs for exact filter parameter names like 'filter.studyType'
            # We would need to add 'filter_study_type' arg to search_studies and map it
            # For now, let's assume it's combined in query_term or handled manually
            # For this example, let's filter 'Phase' post-hoc if not directly supported via simple arg
            fields=[ # Request fields needed for the DataFrame
                "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus",
                "StudyFirstPostDate", "LastUpdatePostDate", "Phase", "StudyType",
                "LeadSponsorName", "Condition", "InterventionName", "LocationCity",
                "LocationState", "LocationCountry", "EnrollmentCount", "StartDate",
                 "PrimaryCompletionDate"
                ],
            max_results=150,
            get_all=True
        )

        if not df_diabetes.empty:
            print(f"Created DataFrame with {df_diabetes.shape[0]} rows and {df_diabetes.shape[1]} columns.")
             # Filter DataFrame further if needed (e.g., for Phase 3)
            df_diabetes_phase3 = df_diabetes[df_diabetes['Phase'].str.contains("PHASE3", na=False)]
            print(f"Filtered down to {len(df_diabetes_phase3)} Phase 3 studies.")
            print("DataFrame head:")
            print(df_diabetes_phase3.head())

            # Example: Save to CSV
            # try:
            #     csv_filename = "diabetes_phase3_studies.csv"
            #     df_diabetes_phase3.to_csv(csv_filename, index=False)
            #     print(f"\nDataFrame saved to {csv_filename}")
            # except Exception as e:
            #     print(f"\nError saving DataFrame to CSV: {e}")

        else:
            print("Could not create DataFrame or no results found.")
        print("-" * 20)
    else:
        print("\n--- Example 4: Search to DataFrame (Skipped - Pandas not installed) ---")


    # --- Example 5: Get Available Fields ---
    # print("\n--- Example 5: Get Available Study Fields ---")
    # try:
    #     fields_metadata = connector.get_study_fields()
    #     print(f"Retrieved metadata for available fields. Example: {fields_metadata.get('studyFields', [])[:5]}") # Show first 5
    # except Exception as e:
    #     print(f"Could not retrieve field metadata: {e}")
    # print("-" * 20)