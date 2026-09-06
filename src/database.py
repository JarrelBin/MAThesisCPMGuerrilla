import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    """
    Open a PostgreSQL connection using environment variables.
    """

    return psycopg.connect(
        host=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
