import pandas as pd
import logging
from abc import ABC, abstractmethod
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    Abstract base class for data loaders.
    """

    @abstractmethod
    def load(self, config: dict) -> pd.DataFrame:
        """
        Loads data from a source and returns a pandas DataFrame.

        Args:
            config (dict): The data source configuration.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        pass


class CsvLoader(BaseLoader):
    """
    Loads data from a CSV file.
    """

    def load(self, config: dict) -> pd.DataFrame:
        """
        Loads data from a CSV file and returns a pandas DataFrame.

        Args:
            config (dict): The data source configuration.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        csv_path = config.get("path")
        if not csv_path:
            logger.error("CSV 'path' not specified in config.")
            return pd.DataFrame()

        logger.info(f"Loading CSV from '{csv_path}'...")
        try:
            return pd.read_csv(csv_path)
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            return pd.DataFrame()


class SqlLoader(BaseLoader):
    """
    Loads data from a SQL database.
    """

    def load(self, config: dict) -> pd.DataFrame:
        """
        Loads data from a SQL database and returns a pandas DataFrame.

        Args:
            config (dict): The data source configuration.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        logger.info(f"Loading data from SQL database...")
        try:
            db_type = config.get("type")
            user = config.get("user")
            host = config.get("host")
            port = config.get("port")
            db = config.get("db")
            password = config.get("password")

            connection_url = f"{db_type}://{user}:{password}@{host}:{port}/{db}"
            engine = create_engine(connection_url)
            query = config.get("query")

            if not query:
                logger.error("'query' not specified in config.")
                return pd.DataFrame()

            logger.info(f"Executing query: {query}")

            with engine.connect() as conn:
                df = pd.read_sql(query, conn)

            logger.info(f"Loaded {len(df)} rows from database.")
            return df

        except ImportError:
            logger.error(f"DB driver for '{db_type}' not installed.")
            logger.error(
                f"Please run: pip install [driver_name] (e.g., psycopg2-binary)"
            )
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to load data from SQL: {e}")
            return pd.DataFrame()
