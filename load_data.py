#!/usr/bin/env python3
"""
Load TIGER2018 US Counties CSV data into PostGIS database
"""

import csv
import sys
import psycopg2
from psycopg2.extras import execute_batch

# Increase CSV field size limit for large geometry fields
csv.field_size_limit(sys.maxsize)

# Database connection parameters
DB_PARAMS = {
    'dbname': 'USCountyDB',
    'user': 'dhanush',  
    'password': '', 
    'host': 'localhost',
    'port': 5432
}

# CSV file path
CSV_FILE = '/Users/dhanush/Desktop/Capstone/TIGER2018_COUNTY_with_state.csv'

def load_counties_data():
    """Load counties data from CSV into PostgreSQL database"""

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    print(f"Reading CSV file: {CSV_FILE}")

    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row

        # Prepare batch insert
        insert_query = """
            INSERT INTO counties (
                geom, statefp, countyfp, countyns, geoid, name, namelsad,
                lsad, classfp, mtfcc, csafp, cbsafp, metdivfp, funcstat,
                aland, awater, intptlat, intptlon, state_name, state_abbr
            ) VALUES (
                ST_GeomFromText(%s, 4269), %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
        """

        batch_data = []
        row_count = 0

        for row in reader:
            # Parse row (20 columns total: 0-19)
            wkt_geom = row[0]  
            statefp = row[1]
            countyfp = row[2]
            countyns = row[3]
            geoid = row[4]
            name = row[5]
            namelsad = row[6]
            lsad = row[7]
            classfp = row[8]
            mtfcc = row[9]
            csafp = row[10] if row[10] and row[10].strip() else None
            cbsafp = row[11] if row[11] and row[11].strip() else None
            metdivfp = row[12] if row[12] and row[12].strip() else None
            funcstat = row[13]
            aland = int(row[14]) if row[14] and row[14].strip() else 0
            awater = int(row[15]) if row[15] and row[15].strip() else 0
            intptlat = float(row[16].replace('+', '')) if row[16] and row[16].strip() else None
            intptlon = float(row[17].replace('+', '')) if row[17] and row[17].strip() else None
            state_name = row[18]
            state_abbr = row[19]

            batch_data.append((
                wkt_geom, statefp, countyfp, countyns, geoid, name, namelsad,
                lsad, classfp, mtfcc, csafp, cbsafp, metdivfp, funcstat,
                aland, awater, intptlat, intptlon, state_name, state_abbr
            ))

            row_count += 1

            # Insert in batches of 100
            if len(batch_data) >= 100:
                execute_batch(cur, insert_query, batch_data)
                conn.commit()
                print(f"Loaded {row_count} rows...")
                batch_data = []

        # Insert remaining rows
        if batch_data:
            execute_batch(cur, insert_query, batch_data)
            conn.commit()
            print(f"Loaded {row_count} rows...")

    print(f"\nTotal rows loaded: {row_count}")

    # Verify data
    cur.execute("SELECT COUNT(*) FROM counties;")
    count = cur.fetchone()[0]
    print(f"Database row count: {count}")

    # Show sample data
    print("\nSample data (first 3 counties):")
    cur.execute("""
        SELECT name, state_abbr, ST_Area(geom::geography)/1000000 as area_km2
        FROM counties
        LIMIT 3;
    """)
    for row in cur.fetchall():
        print(f"  {row[0]}, {row[1]} - Area: {row[2]:.2f} km²")

    cur.close()
    conn.close()
    print("\nData loading complete!")

if __name__ == '__main__':
    load_counties_data()
