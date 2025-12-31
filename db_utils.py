"""
Utility functions for Schweizer Wanderwege project.

This module provides reusable functions for database operations,
display settings, and common configurations to reduce code redundancy
across multiple notebooks.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_CONFIG = {
    'user': 'root',
    'password': 'password',
    'host': 'localhost',
    'port': '3306',
    'database': 'wanderwege_db'
}


def get_db_engine():
    """
    Create and return a SQLAlchemy database engine.

    Returns:
        sqlalchemy.engine.Engine: Database engine for MySQL connection
    """
    connection_string = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(connection_string)


def read_from_db(query):
    """
    Execute a SQL query and return results as a pandas DataFrame.

    Args:
        query (str): SQL query to execute

    Returns:
        pd.DataFrame: Query results as DataFrame

    Example:
        >>> df = read_from_db("SELECT * FROM wanderwege")
        >>> print(f"Loaded {len(df)} rows")
    """
    engine = get_db_engine()
    df = pd.read_sql(query, con=engine)
    print(f"✅ Loaded {len(df)} rows, {df.shape[1]} columns from database")
    return df


def write_to_db(df, table_name, if_exists='replace'):
    """
    Write a DataFrame to MySQL database.

    Args:
        df (pd.DataFrame): DataFrame to write
        table_name (str): Name of the table in database
        if_exists (str): How to behave if table exists ('fail', 'replace', 'append')

    Returns:
        int: Number of rows written

    Example:
        >>> write_to_db(df, 'wanderwege', if_exists='replace')
    """
    engine = get_db_engine()

    rows_written = df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=1000,
        method='multi'
    )

    print(f"✅ Successfully stored {len(df)} rows in table '{table_name}'")
    return len(df)


# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

def setup_display_settings(float_format='{:.2f}', precision=2):
    """
    Configure pandas display settings for better readability.

    Args:
        float_format (str): Format string for float display
        precision (int): Decimal precision for display
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.precision', precision)
    pd.set_option('display.float_format', float_format.format)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    print("✅ Display settings configured")


def setup_visualization_settings(style='seaborn-v0_8-darkgrid', palette='Set2',
                                 figsize=(12, 6), fontsize=11):
    """
    Configure matplotlib and seaborn visualization settings.

    Args:
        style (str): Matplotlib style
        palette (str): Seaborn color palette
        figsize (tuple): Default figure size
        fontsize (int): Default font size
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use(style)
    sns.set_palette(palette)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.size'] = fontsize

    print("✅ Visualization settings configured")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_basic_info(df, name="Dataset"):
    """
    Print basic information about a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the dataset for display
    """
    print("=" * 80)
    print(f"BASIC INFO: {name}")
    print("=" * 80)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts())
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print("=" * 80)

