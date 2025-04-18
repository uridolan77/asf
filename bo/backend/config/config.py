# config.py - starter file

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# MySQL connection settings
DB_USER = os.getenv('BO_DB_USER', 'root')
DB_PASSWORD = os.getenv('BO_DB_PASSWORD', 'Dt%g_9W3z0*!I')
DB_HOST = os.getenv('BO_DB_HOST', 'localhost')
DB_PORT = os.getenv('BO_DB_PORT', '3306')
DB_NAME = os.getenv('BO_DB_NAME', 'bo_admin')

SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
