from typing import Protocol, TypeVar, Generic, List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, select, update, delete
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from typing import Dict, Any, List, Optional, TypeVar, Generic

T = TypeVar('T')  # Generic type variable for the data entity

class DataRepository(Protocol[T]):
    """
    Defines the interface for a generic data repository.
    """

    def create(self, entity: T) -> T:
        """Adds a new entity to the repository.  Returns the entity,
        potentially with an updated ID or other generated fields.
        """
        ...

    def read(self, id: Any) -> Optional[T]:
        """Retrieves an entity by its ID. Returns None if not found."""
        ...

    def read_all(self) -> List[T]:
        """Retrieves all entities from the repository."""
        ...
    
    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """
        Retrieves entities that match the given criteria.
        For example:  criteria = {"name": "John", "age": 30}
        """
        ...

    def update(self, entity: T) -> T:
        """Updates an existing entity. Returns the updated entity."""
        ...

    def delete(self, id: Any) -> bool:
        """Deletes an entity by its ID. Returns True if successful, 
        False if the entity wasn't found.
        ...
class Cache:
    """A simple in-memory cache."""
    def __init__(self, ttl: int = 60):
        """
        Initializes the cache.

        Args:
            ttl: Time-to-live (in seconds) for cached items.
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:
                del self._cache[key]  # Expire the entry
        return None

    def set(self, key: str, value: Any) -> None:
        """Stores an item in the cache."""
        self._cache[key] = {'data': value, 'timestamp': time.time()}

    def clear(self) -> None:
      self. _cache = {}

class CachedDataRepository(Generic[T]):
    """
    A base class for DataRepositories that adds caching.
        Initializes the cached repository.

        Args:
            repository: The underlying DataRepository to wrap.
            cache: The Cache instance to use.
        return f"{method_name}:{args}:{kwargs}"

    def create(self, entity: T) -> T:
        self.cache.clear()
        return self.repository.create(entity)

    def read(self, id: Any) -> Optional[T]:
        cache_key = self._get_cache_key("read", id)
        cached_value = self.cache.get(cache_key)
        if cached_value:
            return cached_value
        else:
            entity = self.repository.read(id)
            if entity:
                self.cache.set(cache_key, entity)
            return entity

    def read_all(self) -> List[T]:
      cache_key = self._get_cache_key("read_all")
      cached_value = self.cache.get(cache_key)
      if cached_value:
          return cached_value
      else:
          entities = self.repository.read_all()
          self.cache.set(cache_key, entities)
          return entities

    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        cache_key = self._get_cache_key("read_by_criteria", criteria)
        cached_value = self.cache.get(cache_key)
        if cached_value:
            return cached_value
        else:
            entities = self.repository.read_by_criteria(criteria)
            self.cache.set(cache_key, entities)
            return entities


    def update(self, entity: T) -> T:
        self.cache.clear()
        return self.repository.update(entity)

    def delete(self, id: Any) -> bool:
        self.cache.clear()
        return self.repository.delete(id)

    def count(self) -> int:
        cache_key = self._get_cache_key("count")
        cached_value = self.cache.get(cache_key)
        if cached_value:
            return cached_value
        else:
            result = self.repository.count()
            self.cache.set(cache_key, result)
            return result



    
class InMemoryDataRepository(Generic[T]):
    """
    An in-memory implementation of the DataRepository interface.
    """

    def __init__(self, id_field: str = "id") -> None:
        """
        Initializes the repository.

        Args:
            id_field: The name of the field to use as the primary key.
        """
        self._data: Dict[Any, T] = {}
        self._next_id: int = 1  # Simple auto-incrementing ID
        self.id_field = id_field

    def create(self, entity: T) -> T:
        """Adds a new entity, assigning a unique ID."""
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__

        if self.id_field in entity_dict and entity_dict[self.id_field] is not None:
          entity_id = entity_dict[self.id_field]
          if entity_id in self._data:
            raise ValueError(f"Entity with id {entity_id} already exists.")
        else:
          entity_id = self._next_id
          entity_dict[self.id_field] = entity_id
          self._next_id += 1

        self._data[entity_id] = entity
        return entity

    def read(self, id: Any) -> Optional[T]:
        """Retrieves an entity by ID."""
        return self._data.get(id)

    def read_all(self) -> List[T]:
        """Retrieves all entities."""
        return list(self._data.values())

    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """Retrieves entities matching the given criteria."""
        results = []
        for entity in self._data.values():
            entity_dict = entity if isinstance(entity, dict) else entity.__dict__
            match = True
            for key, value in criteria.items():
                if key not in entity_dict or entity_dict[key] != value:
                    match = False
                    break
            if match:
                results.append(entity)
        return results

    def update(self, entity: T) -> T:
        """Updates an existing entity."""
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__
        entity_id = entity_dict.get(self.id_field)
        if entity_id is None or entity_id not in self._data:
            raise ValueError(f"Entity with id {entity_id} not found for update.")
        self._data[entity_id] = entity
        return entity

    def delete(self, id: Any) -> bool:
        """Deletes an entity by ID."""
        if id in self._data:
            del self._data[id]
            return True
        return False

    def count(self) -> int:
        """Returns the total number of entities."""
        return len(self._data)

import json
import os
from typing import Dict, Any, List, Optional, TypeVar, Generic

T = TypeVar('T')

class FileDataRepository(Generic[T]):
    """
    A file-based DataRepository implementation using JSON.
        Initializes the repository.

        Args:
            filepath: The path to the JSON file.
            id_field: The name of the field to use as the primary key.
        if not os.path.exists(self.filepath):
            return {}
        with open(self.filepath, 'r') as f:
            try:
                data = json.load(f)
                return {int(k) if k.isdigit() else k: v for k, v in data.items()}
            except json.JSONDecodeError:
                return {}  # Handle empty or invalid file

    def _save_data(self) -> None:
        """Saves the data to the JSON file."""
        with open(self.filepath, 'w') as f:
            json.dump(self._data, f, indent=4)

    def _get_next_id(self) -> int:
      """Determines next id to use"""
      if not self._data:
        return 1

      return max([int(id) for id in self._data.keys() if isinstance(id, (int, str)) and str(id).isdigit()]) + 1


    def create(self, entity: T) -> T:
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__

        if self.id_field in entity_dict and entity_dict[self.id_field] is not None:
          entity_id = entity_dict[self.id_field]
          if entity_id in self._data:
            raise ValueError(f"Entity with id {entity_id} already exists.")
        else:
          entity_id = self._next_id
          entity_dict[self.id_field] = entity_id
          self._next_id += 1

        self._data[entity_id] = entity
        self._save_data()
        return entity

    def read(self, id: Any) -> Optional[T]:
        return self._data.get(id)

    def read_all(self) -> List[T]:
        return list(self._data.values())
    
    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """Retrieves entities matching the given criteria."""
        results = []
        for entity in self._data.values():
            entity_dict = entity if isinstance(entity, dict) else entity.__dict__
            match = True
            for key, value in criteria.items():
                if key not in entity_dict or entity_dict[key] != value:
                    match = False
                    break
            if match:
                results.append(entity)
        return results


    def update(self, entity: T) -> T:
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__
        entity_id = entity_dict.get(self.id_field)
        if entity_id is None or entity_id not in self._data:
            raise ValueError(f"Entity with id {entity_id} not found for update.")
        self._data[entity_id] = entity
        self._save_data()
        return entity

    def delete(self, id: Any) -> bool:
        if id in self._data:
            del self._data[id]
            self._save_data()
            return True
        return False
    
    def count(self) -> int:
        """Returns the total number of entities."""
        return len(self._data)
    

import sqlite3
from typing import Dict, Any, List, Optional, TypeVar, Generic

T = TypeVar('T')

class SQLiteDataRepository(Generic[T]):
    """
    A DataRepository implementation using SQLite.
    """

    def __init__(self, db_path: str, table_name: str, id_field: str = "id") -> None:
        """
        Initializes the repository.

        Args:
            db_path: Path to the SQLite database file.
            table_name: The name of the table to use.
            id_field: The name of the primary key field.
        """
        self.db_path = db_path
        self.table_name = table_name
        self.id_field = id_field
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_table_if_not_exists()


    def _create_table_if_not_exists(self) -> None:
        """Creates the table if it doesn't exist."""
        cursor = self._conn.cursor()
        # We'll use TEXT for all columns for simplicity.
        # In a real application, you'd define specific data types.
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                {self.id_field} TEXT PRIMARY KEY,
                data TEXT  -- Store the entire entity as JSON
            )
        if row is None:
            return None
        return json.loads(row['data'])


    def create(self, entity: T) -> T:
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__
        entity_id = entity_dict.get(self.id_field)

        if entity_id is None:
            entity_id = str(uuid.uuid4()) # Use UUIDs for SQLite
            entity_dict[self.id_field] = entity_id

        cursor = self._conn.cursor()
        try:
          cursor.execute(
              f"INSERT INTO {self.table_name} ({self.id_field}, data) VALUES (?, ?)",
              (str(entity_id), json.dumps(entity_dict))
          )
          self._conn.commit()
        except sqlite3.IntegrityError:
           raise ValueError(f"Entity with id {entity_id} already exists.")

        return entity

    def read(self, id: Any) -> Optional[T]:
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT data FROM {self.table_name} WHERE {self.id_field} = ?", (str(id),))
        row = cursor.fetchone()
        return self._dict_to_entity(row)


    def read_all(self) -> List[T]:
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT data FROM {self.table_name}")
        rows = cursor.fetchall()
        return [self._dict_to_entity(row) for row in rows]

    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        conditions = []
        values = []
        for key, value in criteria.items():
            conditions.append(f"json_extract(data, '$.{key}') = ?")
            values.append(str(value))  # Convert to string for JSON comparison

        where_clause = " AND ".join(conditions)
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT data FROM {self.table_name} WHERE {where_clause}", values)
        rows = cursor.fetchall()
        return [self._dict_to_entity(row) for row in rows]

    def update(self, entity: T) -> T:
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__
        entity_id = entity_dict.get(self.id_field)
        if entity_id is None:
             raise ValueError(f"Entity with id {entity_id} not found for update.")

        cursor = self._conn.cursor()
        cursor.execute(
            f"UPDATE {self.table_name} SET data = ? WHERE {self.id_field} = ?",
            (json.dumps(entity_dict), str(entity_id))
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Entity with ID '{entity_id}' not found.")
        self._conn.commit()
        return entity

    def delete(self, id: Any) -> bool:
        cursor = self._conn.cursor()
        cursor.execute(f"DELETE FROM {self.table_name} WHERE {self.id_field} = ?", (str(id),))
        rows_affected = cursor.rowcount
        self._conn.commit()
        return rows_affected > 0

    def count(self) -> int:
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        return count

    def close(self):
      self._conn.close()


import requests
from typing import Dict, Any, List, Optional, TypeVar, Generic

T = TypeVar('T')
class RestApiDataRepository(Generic[T]):
    """
    A DataRepository implementation that fetches data from a REST API.
    """

    def __init__(self, base_url: str, id_field: str = "id") -> None:
        """
        Initializes the repository.

        Args:
            base_url: The base URL of the API endpoint (e.g., "https://api.example.com/users").
            id_field: The name of the primary key field.
        """
        self.base_url = base_url
        self.id_field = id_field

    def _handle_response(self, response: requests.Response) -> Any:
        """Handles the API response, raising exceptions for errors."""
        if response.status_code >= 400:
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()

    def create(self, entity: T) -> T:
        response = requests.post(self.base_url, json=entity)
        return self._handle_response(response)

    def read(self, id: Any) -> Optional[T]:
        response = requests.get(f"{self.base_url}/{id}")
        if response.status_code == 404:
          return None
        return self._handle_response(response)

    def read_all(self) -> List[T]:
        response = requests.get(self.base_url)
        return self._handle_response(response)

    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        #How this is implemented depends greatly on the API.
        #  This is just a basic example using query parameters.
        response = requests.get(self.base_url, params=criteria)
        return self._handle_response(response)

    def update(self, entity: T) -> T:
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__
        entity_id = entity_dict.get(self.id_field)
        if entity_id is None:
            raise ValueError("Entity must have an ID to be updated.")
        response = requests.put(f"{self.base_url}/{entity_id}", json=entity)
        return self._handle_response(response)

    def delete(self, id: Any) -> bool:
        response = requests.delete(f"{self.base_url}/{id}")
        if response.status_code == 404:
            return False;
        self._handle_response(response)  # Will raise for other error codes
        return response.status_code == 204  # Typically, DELETE returns 204 No Content

    def count(self) -> int:
        # This might not be directly supported by all APIs.  A common approach
        # is to fetch all items and count them, but that's inefficient.
        # We'll provide a simple (but potentially slow) implementation.
        return len(self.read_all())


from sqlalchemy import create_engine, Column, Integer, String, Text, select, update, delete
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from typing import Dict, Any, List, Optional, TypeVar, Generic

# --- Define your data model (using SQLAlchemy's declarative base) ---

Base = declarative_base()
T = TypeVar('T')

class User(Base):  # Example entity
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    # Add other fields as needed

    def to_dict(self):
      """Converts the User object to a dictionary."""
      return {
          "user_id": self.user_id,
          "name": self.name,
          "age": self.age,
      }

class SQLAlchemyDataRepository:
    def __init__(self, db_url: str, model_class: type, s3_bucket_name: str) -> None:
        self.engine = create_async_engine(db_url)
        self.async_session = sessionmaker(
            self.engine, expire_on_commit=False, class_=AsyncSession
        )
        self.model_class = model_class
        self.s3_bucket_name = s3_bucket_name
        self.s3_client = boto3.client('s3') #sync client for generating urls
        self.s3 = aiobotocore.session.get_session().create_client(
            "s3",
        )

    async def _upload_to_s3(self, file_content: bytes, filename: str) -> str:
      try:
          url = self.s3_client.generate_presigned_url(
              ClientMethod='get_object',
              Params={
                  'Bucket': self.s3_bucket_name,
                  'Key': s3_key
              },
              ExpiresIn=3600  # URL expires in 1 hour (adjust as needed)
          )
          return url
      except ClientError as e:
          raise Exception(f"Failed to generate presigned URL: {e}") from e

    async def create(self, entity: ImageUpload, file_content: bytes) -> dict:
        async with self.async_session() as session:
            result = await session.get(self.model_class, id)
            if result:
                image_data = result.to_dict()
                image_data['image_url'] = self._generate_presigned_url(image_data['s3_key'])
                return image_data
            return None

    async def read_all(self) -> List[dict]:
        async with self.async_session() as session:
            async with session.begin():
                stmt = (
                    update(self.model_class)
                    .where(self.model_class.id == entity["id"])
                    .values(**entity)
                    .returning(self.model_class)
                )
                result = await session.execute(stmt)
                updated_entity = result.scalar_one_or_none()
                if updated_entity is None:
                    raise ValueError(f"Image with ID '{entity['id']}' not found.")
                result = updated_entity.to_dict()
                result['image_url'] = self._generate_presigned_url(result['s3_key'])
                return result


    async def delete(self, id: int) -> bool:
      async with self.async_session() as session:
        stmt = select(self.model_class)
        for key, value in criteria.items():
            stmt = stmt.where(getattr(self.model_class, key) == value)
        result = await session.execute(stmt)
        results = []
        for entity in result.scalars():
          entity_dict = entity.to_dict()
          entity_dict['image_url'] = self._generate_presigned_url(entity_dict['s3_key'])
          results.append(entity_dict)
        return results
    
    async def close(self):
    A DataRepository implementation using the GPT-4V (Vision) API.
    This implementation is highly conceptual and depends heavily on the
    specific capabilities of the GPT-4V API and the nature of your data.
    It assumes you're storing and querying based on visual data and descriptions.
        Initializes the repository.

        Args:
            api_key: Your OpenAI API key.
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _gpt4v_request(self, messages: list[dict[str, any]]) -> dict:
        """Sends a request to the GPT-4V API and handles the response."""

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2000  # Adjust as needed
        }

        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    
    def _add_image_message_part(self, image_path:str|None = None, base64_image:str|None = None):
        """Creates a message part for an image, handling file paths or base64 strings"""
        if image_path:
            base64_image = self._encode_image(image_path)
        if base64_image is None:
            raise ValueError("Must provide either image_path or base64_image")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }


    def create(self, entity: T) -> T:
        """
        "Creates" a new entity. In the GPT-4V context, this might mean
        sending an image and description to the API for analysis/storage
        (though the API itself doesn't persistently store data in a traditional sense).

        This implementation adds to the in memory data store.
        """
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__

        entity_id = self._next_id
        entity_dict["id"] = entity_id
        self._next_id+=1;
        self._data[str(entity_id)] = entity

        return entity


    def read(self, id: Any) -> Optional[T]:
        """
        "Reads" an entity.  This would likely involve sending an identifier
        to GPT-4V, and GPT-4V would need some way to reference previously
        "stored" information (e.g., using metadata or descriptions).

        This implementation reads from the in memory store.
        """
        return self._data.get(str(id))


    def read_all(self) -> List[T]:
        """
        "Reads" all entities.  Highly conceptual for GPT-4V.

        This implementation reads from the in memory store.
        """
        return list(self._data.values())

    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]:
        """
        "Reads" entities based on criteria.  This is where you'd leverage
        GPT-4V's ability to understand natural language descriptions.

        Args:
            criteria: A dictionary where keys are attributes (e.g., "description", "color")
                      and values are the desired values or descriptions.

        Returns:
            A list of entities (likely represented as dictionaries) that match
            the criteria, according to GPT-4V's interpretation.
        """
        image_path = criteria.pop("image_path", None)
        base64_image = criteria.pop("base64_image", None)
        criteria_str = ", ".join(f"{key} is {value}" for key, value in criteria.items())

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Find entities where {criteria_str}.  Return the result as a JSON array of objects."
                    },
                    self._add_image_message_part(image_path, base64_image)
                ]
            }
        ]

        response_json = self._gpt4v_request(messages)
        try:
            return json.loads(response_json["choices"][0]["message"]["content"])
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"Unexpected response format from GPT-4V: {response_json}") from e
    
    def describe_image(self, image_path:str|None = None, base64_image:str|None = None) -> str:
        """
        Gets a textual description of an image from gpt-4v

        Args:
            image_path: path to image
            base64_image: base64 encoded image.  If both image_path and base64_image, base64_image takes precedence.

        Returns:
            String containing description.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this image."
                    },
                    self._add_image_message_part(image_path, base64_image)
                ]
            }
        ]

        response_json = self._gpt4v_request(messages)
        return response_json["choices"][0]["message"]["content"]

    def update(self, entity: T) -> T:
        """
        Simulates "updating" an entity.  This is difficult to define
        meaningfully without a persistent storage mechanism associated with
        GPT-4V.
        This implementation updates the in memory data.
        """
        entity_dict = entity if isinstance(entity, dict) else entity.__dict__
        entity_id = entity_dict.get("id")
        if entity_id is None or str(entity_id) not in self._data:
            raise ValueError(f"Entity with id {entity_id} not found for update.")

        self._data[str(entity_id)] = entity
        return entity

    def delete(self, id: Any) -> bool:
        """
        "Deletes" an entity.  Again, highly conceptual without persistent storage.
        This implementation deletes from the in memory data.
        """
        if str(id) in self._data:
            del self._data[str(id)]
            return True
        return False
    
    def count(self) -> int:
        """
        Returns count of "entities".
        This implementation reads from the in-memory data.
        """
        return len(self._data)


# IMPORTANT: Replace "YOUR_API_KEY" with your actual OpenAI API key.
api_key = "YOUR_API_KEY"
repo = GPT4VDataRepository[dict](api_key=api_key)







import json
import uuid #for creating unique IDs.



import unittest

class TestInMemoryDataRepository(unittest.TestCase):

    def setUp(self):
        self.repository = InMemoryDataRepository[dict](id_field="user_id")
        self.user1 = {"user_id": None, "name": "Alice", "age": 30}
        self.user2 = {"user_id": None, "name": "Bob", "age": 25}

    def test_create_and_read(self):
        created_user = self.repository.create(self.user1)
        self.assertEqual(created_user["user_id"], 1)  # Check ID assignment
        read_user = self.repository.read(1)
        self.assertEqual(read_user, self.user1)

    def test_create_existing_id(self):
        self.repository.create(self.user1)
        self.user2["user_id"] = 1
        with self.assertRaises(ValueError):
          self.repository.create(self.user2)

    def test_read_not_found(self):
        read_user = self.repository.read(999)
        self.assertIsNone(read_user)

    def test_read_all(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        all_users = self.repository.read_all()
        self.assertEqual(len(all_users), 2)
        self.assertIn(self.user1, all_users)
        self.assertIn(self.user2, all_users)

    def test_update(self):
        created_user = self.repository.create(self.user1)
        created_user["age"] = 35
        updated_user = self.repository.update(created_user)
        read_user = self.repository.read(1)
        self.assertEqual(read_user["age"], 35)

    def test_update_not_found(self):
        self.user1["user_id"] = 999  # Non-existent ID
        with self.assertRaises(ValueError):
            self.repository.update(self.user1)

    def test_delete(self):
        created_user = self.repository.create(self.user1)
        result = self.repository.delete(1)
        self.assertTrue(result)
        read_user = self.repository.read(1)
        self.assertIsNone(read_user)

    def test_delete_not_found(self):
        result = self.repository.delete(999)
        self.assertFalse(result)

    def test_count(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        count = self.repository.count()
        self.assertEqual(count,2)

    def test_read_by_criteria(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        users = self.repository.read_by_criteria({"age": 30})
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["name"], "Alice")

        users = self.repository.read_by_criteria({"age": 30, "name": "Bob"})
        self.assertEqual(len(users), 0)
        
        users = self.repository.read_by_criteria({"name": "Bob"})
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["age"], 25)


class TestFileDataRepository(unittest.TestCase):

    def setUp(self):
        self.filepath = "test_data.json"
        self.repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        self.user1 = {"user_id": None, "name": "Alice", "age": 30}
        self.user2 = {"user_id": None, "name": "Bob", "age": 25}
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_create_and_read(self):
      created_user = self.repository.create(self.user1)
      self.assertEqual(created_user["user_id"], 1)
      read_user = self.repository.read(1)
      self.assertEqual(read_user, self.user1)

      with open(self.filepath, 'r') as f:
          data = json.load(f)
          self.assertEqual(data, {"1": self.user1})
    
    def test_create_existing_id(self):
        self.repository.create(self.user1)
        self.user2["user_id"] = 1
        with self.assertRaises(ValueError):
          self.repository.create(self.user2)

    def test_read_from_existing_file(self):
        initial_data = {"1": self.user1, "2": self.user2}
        with open(self.filepath, 'w') as f:
            json.dump(initial_data, f)
        
        repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        read_user = repository.read(1)
        self.assertEqual(read_user, self.user1)
        self.assertEqual(repository.count(), 2)

    def test_read_not_found(self):
        read_user = self.repository.read(999)
        self.assertIsNone(read_user)

    def test_read_all(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        all_users = self.repository.read_all()
        self.assertEqual(len(all_users), 2)
        self.assertIn(self.user1, all_users)
        self.assertIn(self.user2, all_users)

    def test_update(self):
        created_user = self.repository.create(self.user1)
        created_user["age"] = 35
        updated_user = self.repository.update(created_user)
        read_user = self.repository.read(1)
        self.assertEqual(read_user["age"], 35)
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            self.assertEqual(data, {"1": read_user})

    def test_update_not_found(self):
        self.user1["user_id"] = 999
        with self.assertRaises(ValueError):
            self.repository.update(self.user1)

    def test_delete(self):
        created_user = self.repository.create(self.user1)
        result = self.repository.delete(1)
        self.assertTrue(result)
        read_user = self.repository.read(1)
        self.assertIsNone(read_user)

        with open(self.filepath, 'r') as f:
            data = json.load(f)
            self.assertEqual(data, {})

    def test_delete_not_found(self):
        result = self.repository.delete(999)
        self.assertFalse(result)

    def test_count(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        count = self.repository.count()
        self.assertEqual(count,2)

    def test_read_by_criteria(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        users = self.repository.read_by_criteria({"age": 30})
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["name"], "Alice")

        users = self.repository.read_by_criteria({"age": 30, "name": "Bob"})
        self.assertEqual(len(users), 0)
        
        users = self.repository.read_by_criteria({"name": "Bob"})
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["age"], 25)

    def test_empty_file(self):
        with open(self.filepath, 'w') as f:
            pass
        repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        self.assertEqual(repository.count(), 0)
        created_user = repository.create(self.user1)
        self.assertEqual(created_user["user_id"], 1)
        self.assertEqual(repository.count(), 1)

    def test_invalid_json(self):
        with open(self.filepath, 'w') as f:
            f.write("{invalid json")

        repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        self.assertEqual(repository.count(), 0)
        created_user = repository.create(self.user1)
        self.assertEqual(created_user["user_id"], 1)
        self.assertEqual(repository.count(), 1)

    def test_mixed_id_types(self):
        initial_data = {"1": self.user1, "abc": {"user_id": "abc", "name": "Charlie", "age": 40}}
        with open(self.filepath, 'w') as f:
            json.dump(initial_data, f)

        repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        self.assertEqual(repository.count(), 2)

        user3 = {"user_id": None, "name": "Dave", "age":50}
        created = repository.create(user3)
        self.assertEqual(created["user_id"], 2)

        self.assertEqual(repository.read("abc")["name"], "Charlie")
        self.assertEqual(repository.read(2)["name"], "Dave")

class TestSQLiteDataRepository(unittest.TestCase):
    def setUp(self):
        self.db_path = "test.db"
        self.table_name = "users"
        self.repository = SQLiteDataRepository[dict](db_path=self.db_path, table_name=self.table_name, id_field="user_id")
        self.user1 = {"user_id": None, "name": "Alice", "age": 30}
        self.user2 = {"user_id": None, "name": "Bob", "age": 25}


    def tearDown(self):
        self.repository.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_create_and_read(self):
        created_user = self.repository.create(self.user1)
        self.assertIsNotNone(created_user["user_id"]) # Check UUID assignment
        read_user = self.repository.read(created_user["user_id"])
        self.assertEqual(read_user, created_user)


    def test_create_existing_id(self):
        created_user = self.repository.create(self.user1)
        with self.assertRaises(ValueError):
          self.repository.create(created_user) #try to create again.


    def test_read_not_found(self):
        read_user = self.repository.read("nonexistent_id")
        self.assertIsNone(read_user)

    def test_read_all(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        all_users = self.repository.read_all()
        self.assertEqual(len(all_users), 2)
        self.assertEqual(all_users[0]["name"], "Alice")
        self.assertEqual(all_users[1]["name"], "Bob")

    def test_update(self):
        created_user = self.repository.create(self.user1)
        created_user["age"] = 35
        updated_user = self.repository.update(created_user)
        read_user = self.repository.read(created_user["user_id"])
        self.assertEqual(read_user["age"], 35)

    def test_update_not_found(self):
         with self.assertRaises(ValueError):
            self.repository.update({"user_id":"non_existant", "name":"Test"})

    def test_delete(self):
        created_user = self.repository.create(self.user1)
        result = self.repository.delete(created_user["user_id"])
        self.assertTrue(result)
        read_user = self.repository.read(created_user["user_id"])
        self.assertIsNone(read_user)

    def test_delete_not_found(self):
        result = self.repository.delete("nonexistent_id")
        self.assertFalse(result)

    def test_count(self):
        self.repository.create(self.user1)
        self.repository.create(self.user2)
        count = self.repository.count()
        self.assertEqual(count,2)

    def test_read_by_criteria(self):
      self.repository.create(self.user1)
      self.repository.create(self.user2)

      users = self.repository.read_by_criteria({"age": 30})
      self.assertEqual(len(users), 1)
      self.assertEqual(users[0]["name"], "Alice")

      users = self.repository.read_by_criteria({"age": 30, "name": "Bob"})
      self.assertEqual(len(users), 0)

      users = self.repository.read_by_criteria({"name": "Bob"})
      self.assertEqual(len(users), 1)
      self.assertEqual(users[0]["age"], 25)

if __name__ == '__main__':
    unittest.main()


import unittest
from unittest.mock import patch, Mock
class TestRestApiDataRepository(unittest.TestCase):

    def setUp(self):
        self.base_url = "https://api.example.com/users"
        self.repository = RestApiDataRepository[dict](base_url=self.base_url, id_field="id")
        self.user1 = {"id": 1, "name": "Alice", "age": 30}
        self.user2 = {"id": 2, "name": "Bob", "age": 25}

    @patch('requests.post')
    def test_create(self, mock_post):
        mock_post.return_value.status_code = 201  # Simulate successful creation
        mock_post.return_value.json.return_value = self.user1
        created_user = self.repository.create(self.user1)
        mock_post.assert_called_once_with(self.base_url, json=self.user1)
        self.assertEqual(created_user, self.user1)

    @patch('requests.get')
    def test_read(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = self.user1

        user = self.repository.read(1)
        mock_get.assert_called_once_with(f"{self.base_url}/1")
        self.assertEqual(user, self.user1)

    @patch('requests.get')
    def test_read_not_found(self, mock_get):
        mock_get.return_value.status_code = 404
        user = self.repository.read(999)
        mock_get.assert_called_once_with(f"{self.base_url}/999")
        self.assertIsNone(user)

    @patch('requests.get')
    def test_read_all(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [self.user1, self.user2]

        users = self.repository.read_all()
        mock_get.assert_called_once_with(self.base_url)
        self.assertEqual(users, [self.user1, self.user2])

    @patch('requests.get')
    def test_read_by_criteria(self, mock_get):
      mock_get.return_value.status_code = 200
      mock_get.return_value.json.return_value = [self.user1]
      users = self.repository.read_by_criteria({"age": 30})
      mock_get.assert_called_once_with(self.base_url, params={"age": 30})
      self.assertEqual(users, [self.user1])

    @patch('requests.put')
    def test_update(self, mock_put):
        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = self.user1
        updated_user = self.repository.update(self.user1)
        mock_put.assert_called_once_with(f"{self.base_url}/1", json=self.user1)
        self.assertEqual(updated_user, self.user1)
    
    @patch('requests.put')
    def test_update_no_id(self, mock_put):
        with self.assertRaises(ValueError):
            self.repository.update({"name":"No ID"})

    @patch('requests.delete')
    def test_delete(self, mock_delete):
        mock_delete.return_value.status_code = 204  # No Content
        result = self.repository.delete(1)
        mock_delete.assert_called_once_with(f"{self.base_url}/1")
        self.assertTrue(result)

    @patch('requests.delete')
    def test_delete_not_found(self, mock_delete):
        mock_delete.return_value.status_code = 404
        result = self.repository.delete(999)
        mock_delete.assert_called_once_with(f"{self.base_url}/999")
        self.assertFalse(result)

    @patch('requests.get')
    def test_count(self, mock_get):
      mock_get.return_value.status_code = 200
      mock_get.return_value.json.return_value = [self.user1, self.user2]

      count = self.repository.count()
      mock_get.assert_called_once_with(self.base_url) #read_all is called.
      self.assertEqual(count, 2)


    @patch('requests.get')
    def test_handle_response_error(self, mock_get):
        mock_get.return_value.status_code = 500  # Simulate server error
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("Server Error")

        with self.assertRaises(requests.exceptions.HTTPError):
            self.repository.read(1)
        mock_get.assert_called_once_with(f"{self.base_url}/1")


async def upload_image(
    image_file: bytes,
    filename: str,
    image_data: ImageUpload,
    db_repo: SQLAlchemyDataRepository,
    gpt4v_repo: GPT4VDataRepository,
):