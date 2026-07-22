import sqlite3

def delete_unwanted_cache_entries():
    conn = sqlite3.connect("query_cache.db")
    cursor = conn.cursor()
    
    # Delete rows with normalized_query 'who is hitler' or 'who is donald trump' or rowid in (14, 15)
    cursor.execute("""
        DELETE FROM query_cache 
        WHERE normalized_query IN ('who is hitler', 'who is donald trump')
           OR original_query LIKE '%hitler%' 
           OR original_query LIKE '%donald trump%'
    """)
    conn.commit()
    deleted = cursor.rowcount
    print(f"[SUCCESS] Deleted {deleted} unwanted non-policy queries from query_cache.db")
    
    cursor.execute("SELECT rowid, original_query, normalized_query FROM query_cache")
    rows = cursor.fetchall()
    print(f"\nRemaining valid queries in query_cache.db ({len(rows)} entries):")
    for r in rows:
        print(f"  • RowID {r[0]}: '{r[1]}'")

    conn.close()

if __name__ == "__main__":
    delete_unwanted_cache_entries()
