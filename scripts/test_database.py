from src.database import get_connection


EXPECTED_TABLES = {
    "document",
    "page",
    "ocr_run",
    "ocr_region",
    "token",
    "token_correction",
}


def main():

    with get_connection() as connection:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename;
                """
            )

            tables = {
                row[0]
                for row in cursor.fetchall()
            }

    print("Database: defector_reports")
    print()

    print("Tables:")

    for table in sorted(tables):
        print(f"  {table}")

    missing = EXPECTED_TABLES - tables

    if missing:
        print()
        print("Missing tables:")

        for table in sorted(missing):
            print(f"  {table}")

        raise SystemExit(1)

    print()
    print("Schema OK.")


if __name__ == "__main__":
    main()
