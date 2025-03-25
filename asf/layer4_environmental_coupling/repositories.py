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
        """
        ...

    def count(self) -> int:
        """Returns the total number of entities in the repository."""
        ...
class Cache:
    """A simple in-memory cache."""
    def __init__(self, ttl: int = 60):
        """
        Initializes the cache.

        Args:
            ttl: Time-to-live (in seconds) for cached items.
        """
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache."""
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
    """

    def __init__(self, repository: DataRepository[T], cache: Cache) -> None:
        """
        Initializes the cached repository.

        Args:
            repository: The underlying DataRepository to wrap.
            cache: The Cache instance to use.
        """
        self.repository = repository
        self.cache = cache

    def _get_cache_key(self, method_name: str, *args: Any, **kwargs: Any) -> str:
        """Generates a unique cache key based on the method and arguments."""
        # A simple key generation; you might need a more robust method
        # for complex arguments.
        return f"{method_name}:{args}:{kwargs}"

    def create(self, entity: T) -> T:
        # Invalidate cache on create (since data has changed).
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
        # Invalidate cache on update.
        self.cache.clear()
        return self.repository.update(entity)

    def delete(self, id: Any) -> bool:
        # Invalidate cache on delete.
        self.cache.clear()
        return self.repository.delete(id)

    def count(self) -> int:
        # count can also be cached, or you may decide to always get a fresh count.
        cache_key = self._get_cache_key("count")
        cached_value = self.cache.get(cache_key)
        if cached_value:
            return cached_value
        else:
            result = self.repository.count()
            self.cache.set(cache_key, result)
            return result

# Example usage (with InMemoryDataRepository)
# repo = InMemoryDataRepository[dict]()
# cache = Cache(ttl=300)  # 5-minute TTL
# cached_repo = CachedDataRepository(repo, cache)

# Example usage (with SQLiteDataRepository)
# repo = SQLiteDataRepository[dict](db_path="test.db", table_name="users")
# cache = Cache(ttl=300)  # 5-minute TTL
# cached_repo = CachedDataRepository(repo, cache)

#Example usage (with RestApiDataRepository)
# repo = RestApiDataRepository[dict](base_url = "...")
# cache = Cache(ttl=300)
# cached_repo = CachedDataRepository(repo, cache)
    
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
    """

    def __init__(self, filepath: str, id_field: str = "id") -> None:
        """
        Initializes the repository.

        Args:
            filepath: The path to the JSON file.
            id_field: The name of the field to use as the primary key.
        """
        self.filepath = filepath
        self.id_field = id_field
        self._data: Dict[Any, T] = self._load_data()
        self._next_id: int = self._get_next_id()

    def _load_data(self) -> Dict[Any, T]:
        """Loads data from the JSON file."""
        if not os.path.exists(self.filepath):
            return {}
        with open(self.filepath, 'r') as f:
            try:
                data = json.load(f)
                # Convert keys to integers if they represent IDs
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

# Assuming DataRepository and T are defined as in the previous examples
# ... (Include the DataRepository Protocol definition from earlier)
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
        """)
        self._conn.commit()

    def _dict_to_entity(self, row: sqlite3.Row) -> T:
        """Converts a database row (sqlite3.Row) back into an entity."""
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
        # Simple criteria matching using json_extract (SQLite 3.38+).
        #  More complex criteria require more sophisticated SQL.
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

# Assuming DataRepository and T are defined as in the previous examples
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
          # Add other fields here
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
            #region_name="your-region",  # Optional: Specify region
            #aws_secret_access_key="YOUR_SECRET_KEY",   # Only if not using env vars or IAM roles
            #aws_access_key_id="YOUR_ACCESS_KEY",       # Only if not using env vars or IAM roles
        )

    async def _upload_to_s3(self, file_content: bytes, filename: str) -> str:
        """Uploads a file to S3 and returns the S3 key."""
        try:
            # Generate a unique key for the S3 object
            s3_key = f"uploads/{uuid.uuid4()}-{filename}"

            async with self.s3 as s3:
                await s3.put_object(
                    Bucket=self.s3_bucket_name, Key=s3_key, Body=file_content
                )
            return s3_key
        except ClientError as e:
            raise Exception(f"S3 upload failed: {e}") from e

    def _generate_presigned_url(self, s3_key: str) -> str:
      """Generates a presigned URL for accessing an S3 object."""
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
        """Creates a new image record in the database and uploads the image to S3."""

        # Validate input data
        try:
          validated_data = entity.model_dump() # No longer need to catch validation errors, handled in upload_image
        except ValidationError as e:
            raise ValueError(f"Invalid input data: {e}") from e
        
        # Upload to S3.
        s3_key = await self._upload_to_s3(file_content, validated_data.get("original_filename") or "untitled_image")

        async with self.async_session() as session:
            async with session.begin():
                db_entity = self.model_class(
                    s3_key=s3_key,
                    original_filename = validated_data.get("original_filename"),
                    original_description=validated_data.get("original_description"),
                    tags=validated_data.get("tags")
                )
                session.add(db_entity)
            await session.refresh(db_entity)
            result = db_entity.to_dict()
            # Generate a presigned URL and add it to the result.
            result['image_url'] = self._generate_presigned_url(result['s3_key'])
            return result


    async def read(self, id: Any) -> Optional[dict]:
        """Reads image data by ID and generates a presigned URL."""
        async with self.async_session() as session:
            result = await session.get(self.model_class, id)
            if result:
                image_data = result.to_dict()
                #Generate presigned url
                image_data['image_url'] = self._generate_presigned_url(image_data['s3_key'])
                return image_data
            return None

    async def read_all(self) -> List[dict]:
      """Reads all images and generates presigned URLS"""
      async with self.async_session() as session:
          result = await session.execute(select(self.model_class))
          results = []
          for entity in result.scalars():
            entity_dict = entity.to_dict()
            entity_dict['image_url'] = self._generate_presigned_url(entity_dict['s3_key'])
            results.append(entity_dict)
          return results

    async def update(self, entity: dict) -> dict:
        """Updates image metadata (not the image itself)."""
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
        """Deletes an image record from the database AND the image from S3."""
        async with self.async_session() as session:
            async with session.begin():
                # 1. Get the S3 key before deleting from the database
                image = await session.get(self.model_class, id)
                if not image:
                    return False
                s3_key = image.s3_key

                # 2. Delete from the database
                stmt = delete(self.model_class).where(self.model_class.id == id)
                result = await session.execute(stmt)
                if result.rowcount == 0:
                    return False #should never get here, but leave it.

            # 3. Delete from S3 *after* successful database deletion
            try:
                async with self.s3 as s3:
                    await s3.delete_object(Bucket=self.s3_bucket_name, Key=s3_key)
            except ClientError as e:
                # Log the error, but don't re-raise it (the database record is deleted)
                print(f"Error deleting from S3: {e}")
                # Consider adding the S3 key to a queue for later deletion attempts.
            return True

    async def read_by_criteria(self, criteria:Dict[str, Any])->List[dict]:
      """Reads by criteria and generates presigned URLS"""
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
        """Closes the aiohttp session and database engine."""
        await self.s3.close()
        await self.engine.dispose()

# --- Example Usage ---
# Create a repository for the User model
# repo = SQLAlchemyDataRepository[dict]("sqlite:///test.db", User)

# Create a new user
# new_user = {"name": "Charlie", "age": 40}
# created_user = repo.create(new_user)
# print(f"Created User: {created_user}")

# # Read a user by ID
# read_user = repo.read(created_user["user_id"])
# print(f"Read User: {read_user}")

# # Update the user
# created_user["age"] = 41
# updated_user = repo.update(created_user)
# print(f"Updated User: {updated_user}")

# #Read by Criteria
# print(f"Criteria search result: {repo.read_by_criteria({'age':41})}")

# # Delete the user
# repo.delete(created_user["user_id"])

# #count
# print(f"Number of users: {repo.count()}")

import base64
import json
import requests
from typing import Dict, Any, List, Optional, TypeVar, Generic, Protocol

# --- Define the DataRepository Interface (if you haven't already) ---
T = TypeVar('T')

class DataRepository(Protocol[T]):
    def create(self, entity: T) -> T: ...
    def read(self, id: Any) -> Optional[T]: ...
    def read_all(self) -> List[T]: ...
    def read_by_criteria(self, criteria: Dict[str, Any]) -> List[T]: ...
    def update(self, entity: T) -> T: ...
    def delete(self, id: Any) -> bool: ...
    def count(self) -> int: ...


class GPT4VDataRepository(Generic[T]):
    """
    A DataRepository implementation using the GPT-4V (Vision) API.
    This implementation is highly conceptual and depends heavily on the
    specific capabilities of the GPT-4V API and the nature of your data.
    It assumes you're storing and querying based on visual data and descriptions.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1/chat/completions", model:str = "gpt-4-vision-preview") -> None:
        """
        Initializes the repository.

        Args:
            api_key: Your OpenAI API key.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        #This is a simplified representation, for demonstration
        #It would be VERY unusual to store data directly inside of an API wrapper.
        #It's included here to allow create/read/update/delete/count methods without
        #having an underlying storage mechanism.
        self._data: Dict[str, T] = {}
        self._next_id = 1;


    def _encode_image(self, image_path: str) -> str:
        """Encodes an image to base64."""
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
            # Extract the JSON array from the response.  This assumes GPT-4V
            # returns a JSON array as requested.
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


# --- Example Usage ---
# IMPORTANT: Replace "YOUR_API_KEY" with your actual OpenAI API key.
api_key = "YOUR_API_KEY"
repo = GPT4VDataRepository[dict](api_key=api_key)

# # "Create" (analyze) an image and its description.  In a real scenario, you'd
# # likely store the image and description in your own database, and use GPT-4V
# # for analysis and querying.
# new_entity = {"image_path": "path/to/your/image.jpg", "description": "A cat sitting on a mat"}
# created_entity = repo.create(new_entity)
# print(f"Created entity (conceptual): {created_entity}")


# # Find images matching a description.
# matching_entities = repo.read_by_criteria({"description": "cat", "color": "orange", "image_path": "path/to/your/image.jpg"})
# print(f"Entities matching criteria: {matching_entities}")

# # Describe an image.
# description = repo.describe_image("path/to/another/image.jpg")
# print(f"Image Description: {description}")

# # "Read", "Update" and "Delete" operations would depend on your own storage system
# # and how you use the information returned by GPT-4V.
# # The following are conceptual and wouldn't work as-is without that context.
# # read_entity = repo.read("some_identifier")
# # updated_entity = repo.update({"identifier": "some_identifier", "new_description": "A fluffy cat"})
# # repo.delete("some_identifier")


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
        #clean up file if it exists
        if os.path.exists(self.filepath):
            os.remove(self.filepath)


    def tearDown(self):
        # Clean up the test file after each test
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_create_and_read(self):
      created_user = self.repository.create(self.user1)
      self.assertEqual(created_user["user_id"], 1)
      read_user = self.repository.read(1)
      self.assertEqual(read_user, self.user1)

      # Verify file contents
      with open(self.filepath, 'r') as f:
          data = json.load(f)
          self.assertEqual(data, {"1": self.user1})
    
    def test_create_existing_id(self):
        self.repository.create(self.user1)
        self.user2["user_id"] = 1
        with self.assertRaises(ValueError):
          self.repository.create(self.user2)

    def test_read_from_existing_file(self):
        # Create a file with initial data
        initial_data = {"1": self.user1, "2": self.user2}
        with open(self.filepath, 'w') as f:
            json.dump(initial_data, f)
        
        #create a new instance
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
        # Verify file contents
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

        #verify file is empty
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
        #create an empty file
        with open(self.filepath, 'w') as f:
            pass
        repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        self.assertEqual(repository.count(), 0)
        created_user = repository.create(self.user1)
        self.assertEqual(created_user["user_id"], 1)
        self.assertEqual(repository.count(), 1)

    def test_invalid_json(self):
        #create an invalid json file.
        with open(self.filepath, 'w') as f:
            f.write("{invalid json")

        repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        self.assertEqual(repository.count(), 0)
        created_user = repository.create(self.user1)
        self.assertEqual(created_user["user_id"], 1)
        self.assertEqual(repository.count(), 1)

    def test_mixed_id_types(self):
        # Test with a mix of integer and string IDs
        initial_data = {"1": self.user1, "abc": {"user_id": "abc", "name": "Charlie", "age": 40}}
        with open(self.filepath, 'w') as f:
            json.dump(initial_data, f)

        repository = FileDataRepository[dict](filepath=self.filepath, id_field="user_id")
        self.assertEqual(repository.count(), 2)

        # Create a new user (should get id 2)
        user3 = {"user_id": None, "name": "Dave", "age":50}
        created = repository.create(user3)
        self.assertEqual(created["user_id"], 2)

        #read a user
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
        # We can't directly compare the dictionaries due to potential UUID differences,
        # so we compare relevant fields.
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
# Mocking the requests library.
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

# --- Application Logic (Example) ---

async def upload_image(
    image_file: bytes,
    filename: str,
    image_data: ImageUpload,
    db_repo: SQLAlchemyDataRepository,
    gpt4v_repo: GPT4VDataRepository,
):
    """Uploads an image, stores it in S3, and gets a GPT-4V description."""

    # 1. Store the image in S3 and the metadata in the database.
    try:
      created_image = await db_repo.create(image_data, image_file)
    except Exception as e:
       raise Exception(f"Database/S3 error: {e}")

    # 2. Get a description from GPT-4V (using the now-available S3 URL)
    try:
        gpt4v_description = await gpt4v_repo.describe_image(created_image['image_url'])
        # 3. Update the image metadata with the GPT-4V description
        created_image["gpt4v_description"] = gpt4v_description
        await db_repo.update(created_image) #update with the GPT-4V desc

    except Exception as e:
        # Handle GPT-4V errors (e.g., log them, use a default description)
        print(f"GPT-4V error: {e}")
        #You may decide to raise, or continue with a default.

    return created_image



async def main():
    db_url = "sqlite+aiosqlite:///./test.db"  # Use aiosqlite for async SQLite
    s3_bucket_name = "your-s3-bucket-name"  # REPLACE WITH YOUR BUCKET NAME
    db_repo = SQLAlchemyDataRepository(db_url, Image, s3_bucket_name)
    gpt4v_repo = GPT4VDataRepository(api_key="YOUR_API_KEY") #Replace with your key.

    async with db_repo.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # --- Example Usage ---
    try:
        # Simulate image file upload (replace with actual file reading)
        image_file = b"fake image content"  # Replace with actual image bytes
        filename = "example.jpg"
        image_data = ImageUpload(original_description="A beautiful sunset", tags="sunset,nature")


        uploaded_image = await upload_image(image_file, filename, image_data, db_repo, gpt4v_repo)
        print(f"Uploaded image: {uploaded_image}")

        #Read an Image
        retrieved_image = await db_repo.read(uploaded_image['id'])
        print(f"Retrieved Image: {retrieved_image}")

        #Read All Images
        all_images = await db_repo.read_all()
        print(f"All Images: {all_images}")

        #Delete an image
        deleted = await db_repo.delete(uploaded_image['id'])
        print(f"Deleted: {deleted}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await db_repo.close() # Important: Close the database connections


if __name__ == "__main__":
    asyncio.run(main())
