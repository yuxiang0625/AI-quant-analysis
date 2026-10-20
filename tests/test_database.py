from database.db import initialize_database, get_table_names


def test_database_tables():
    initialize_database()

    tables = get_table_names()

    assert "gold_prices" in tables
    assert "comex_inventory" in tables
    assert "macro_data" in tables
    assert "institution_reports" in tables