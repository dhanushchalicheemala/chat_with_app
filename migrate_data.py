#!/usr/bin/env python3
"""
Data migration script to set up the counties table in Railway PostgreSQL database
"""

import psycopg2
import os
import sys

def get_db_params():
    """Get database parameters from environment variables"""
    return {
        'dbname': os.getenv('PGDATABASE', 'USCountyDB'),
        'user': os.getenv('PGUSER', 'dhanush'),
        'password': os.getenv('PGPASSWORD', ''),
        'host': os.getenv('PGHOST', 'localhost'),
        'port': int(os.getenv('PGPORT', 5432))
    }

def create_counties_table():
    """Create the counties table with PostGIS support"""
    db_params = get_db_params()
    
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Enable PostGIS extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        
        # Create counties table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS counties (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255),
            state_name VARCHAR(255),
            state_abbr VARCHAR(10),
            aland BIGINT,
            awater BIGINT,
            geom GEOMETRY(MULTIPOLYGON, 4326),
            geom_4326 GEOMETRY(MULTIPOLYGON, 4326),
            intptlat DECIMAL(10, 7),
            intptlon DECIMAL(10, 7)
        );
        """
        
        cur.execute(create_table_sql)
        conn.commit()
        
        print("✅ Counties table created successfully!")
        
        # Create spatial index
        cur.execute("CREATE INDEX IF NOT EXISTS idx_counties_geom ON counties USING GIST (geom);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_counties_geom_4326 ON counties USING GIST (geom_4326);")
        
        conn.commit()
        print("✅ Spatial indexes created!")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        return False

def load_sample_data():
    """Load some sample county data for testing"""
    db_params = get_db_params()
    
    sample_data = [
        {
            'name': 'Los Angeles',
            'state_name': 'California',
            'state_abbr': 'CA',
            'aland': 12300000000,
            'awater': 100000000,
            'intptlat': 34.0522,
            'intptlon': -118.2437
        },
        {
            'name': 'Cook',
            'state_name': 'Illinois',
            'state_abbr': 'IL',
            'aland': 24000000000,
            'awater': 500000000,
            'intptlat': 41.8781,
            'intptlon': -87.6298
        }
    ]
    
    try:
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        for county in sample_data:
            insert_sql = """
            INSERT INTO counties (name, state_name, state_abbr, aland, awater, intptlat, intptlon)
            VALUES (%(name)s, %(state_name)s, %(state_abbr)s, %(aland)s, %(awater)s, %(intptlat)s, %(intptlon)s)
            ON CONFLICT DO NOTHING;
            """
            cur.execute(insert_sql, county)
        
        conn.commit()
        print("✅ Sample data loaded successfully!")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting database migration...")
    
    if create_counties_table():
        print("📊 Loading sample data...")
        load_sample_data()
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration failed!")
        sys.exit(1)
