
import sqlite3
import pandas as pd

conn = sqlite3.connect("security_audit.db")

df = pd.read_sql_query(
    """
    SELECT
        timestamp,
        user_id,
        role,
        tool_name,
        status,
        duration_ms,
        error_message
    FROM tool_audit
    ORDER BY id DESC
    """,
    conn,
)

print(df.to_string(index=False, col_space=15))

conn.close()
