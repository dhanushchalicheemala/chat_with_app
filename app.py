
"""
Chat with the Map - Streamlit Application with LangGraph ReAct Agent
Natural Language Interface for US Counties PostGIS Database
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
import subprocess
import psycopg2
import json
import requests
import os
import re
import pydeck as pdk
from typing import Tuple, List, Optional, Dict, Any
from langchain_core.tools import tool
# from langchain_community.llms import Ollama # Deprecated
from langchain_ollama import OllamaLLM as Ollama # New import
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Database configuration - Railway will provide these as environment variables
DB_PARAMS = {
    'dbname': os.getenv('PGDATABASE', 'USCountyDB'),
    'user': os.getenv('PGUSER', 'dhanush'),
    'password': os.getenv('PGPASSWORD', ''),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', 5432))
}

# Ollama configuration - Railway service URL
OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
MODEL_NAME = "llama3.2:3b"
MAPBOX_TOKEN = os.getenv('MAPBOX_API_KEY') or os.getenv('MAPBOX_TOKEN')

# Schema information for the LLM
SCHEMA_INFO = """
Database: USCountyDB

CRITICAL: THERE IS ONLY ONE TABLE IN THIS DATABASE!
- Table: counties (this is the ONLY table)
- DO NOT use JOINs with other tables - there is NO states table, NO USStateDB table, NO other tables
- DO NOT reference tables like "states", "USStateDB", "state_boundaries", etc. - they DO NOT EXIST
- All state information is stored in the counties table via state_name and state_abbr columns

Table: counties

Schema:
- id: SERIAL PRIMARY KEY
- geom: GEOMETRY(MULTIPOLYGON, 4269) -- County boundary geometry (NAD83)
- statefp: VARCHAR(2) -- State FIPS code
- countyfp: VARCHAR(3) -- County FIPS code
- countyns: VARCHAR(8) -- County ANSI code
- geoid: VARCHAR(5) -- Geographic identifier (statefp + countyfp)
- name: VARCHAR(100) -- County name
- namelsad: VARCHAR(100) -- County name with legal/statistical description
- lsad: VARCHAR(2) -- Legal/statistical area description code
- classfp: VARCHAR(2) -- Class FIPS code
- mtfcc: VARCHAR(5) -- MAF/TIGER feature class code
- csafp: VARCHAR(3) -- Combined statistical area code
- cbsafp: VARCHAR(5) -- Metropolitan statistical area code
- metdivfp: VARCHAR(5) -- Metropolitan division code
- funcstat: VARCHAR(1) -- Functional status
- aland: BIGINT -- Land area in square meters
- awater: BIGINT -- Water area in square meters
- intptlat: NUMERIC -- Latitude of interior point
- intptlon: NUMERIC -- Longitude of interior point
- state_name: VARCHAR(100) -- Full state name (e.g., "California")
- state_abbr: VARCHAR(2) -- State abbreviation (e.g., "CA")

Spatial Functions Available:
- ST_Area(geom::geography) -- Returns area in square meters
- ST_Perimeter(geom::geography) -- Returns perimeter in meters
- ST_Distance(geom1::geography, geom2::geography) -- Distance in meters
- ST_Touches(geom1, geom2) -- Returns true if geometries touch
- ST_Contains(geom1, geom2) -- Returns true if geom1 contains geom2
- ST_Within(geom1, geom2) -- Returns true if geom1 is within geom2
- ST_Intersects(geom1, geom2) -- Returns true if geometries intersect
- ST_Centroid(geom) -- Returns center point
- ST_Union(geom1, geom2) -- Combines geometries
- ST_AsGeoJSON(geom) -- Returns GeoJSON representation

Notes:
- Use ::geography for accurate area/distance calculations
- Areas are in square meters (divide by 1,000,000 for km²)
- Distances are in meters (divide by 1,000 for km)
- SRID 4269 is NAD83 coordinate system

CRITICAL RULES:
- DO NOT create new geometry objects with ST_GeomFromText or ST_MakePoint
- DO NOT use ST_Touches, ST_Contains with hand-written geometry
- DO use the existing geom column for spatial operations
- For spatial queries, use the geom column that already exists in the table
- DO NOT use JOINs with other tables - there is ONLY the counties table
- For "counties in [state]" questions, use WHERE state_name = '[State]' or WHERE state_abbr = '[XX]' - DO NOT use ST_Within or ST_Contains
- DO NOT try to JOIN with a states table - use state_name/state_abbr columns directly from counties table
- DO NOT try to JOIN with a states table - use state_name/state_abbr columns directly from counties table
- Use ILIKE for string comparisons to be case-insensitive (e.g. state_name ILIKE 'california')

IMPORTANT: BORDER/NEIGHBOR QUERIES
- If the user asks for "bordering", "adjacent", "next to", "neighbors" of a county:
- YOU MUST USE A SELF-JOIN on the counties table!
- Pattern: SELECT c2.name, c2.state_name, ... FROM counties c1, counties c2 WHERE c1.name ILIKE '[Target County]' AND ST_Touches(c1.geom, c2.geom) AND c1.id != c2.id
- DO NOT create geometries with ST_GeomFromText - use the existing geom column!

IMPORTANT: MAP VISUALIZATION
- If the user asks to "show", "map", "visualize", or "display", YOU MUST SELECT GEOMETRY COLUMNS!
- Include: ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon
- Example: SELECT name, state_name, ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon FROM counties WHERE ...

IMPORTANT: AGGREGATION FUNCTIONS
- When question asks for "total", "sum", "combined", "all together" → USE SUM() aggregation
- When question asks for "average", "mean" → USE AVG() aggregation
- When question asks for "how many", "count" → USE COUNT() aggregation
- When question asks for "largest", "biggest", "maximum" → USE MAX() aggregation
- When question asks for "smallest", "minimum" → USE MIN() aggregation
- CRITICAL: For "total area" questions, you MUST use SUM(ST_Area(...)) NOT just ST_Area(...)
- CRITICAL: When using SUM(), AVG(), etc., do NOT select individual rows - return a single aggregated result

EXAMPLE QUERIES (24 COMPREHENSIVE PATTERNS):

-- BASIC QUERIES --
1. County in specific state - "Show me Madison County, Idaho":
   SELECT name, state_name, ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon
   FROM counties WHERE name ILIKE 'Madison' AND state_name ILIKE 'Idaho';

2. County name across all states - "Visualize Madison County in all states":
   SELECT name, state_name, ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon
   FROM counties WHERE name ILIKE 'Madison';

3. Count county name occurrences - "How many counties are called Madison County?":
   SELECT COUNT(*) as count FROM counties WHERE name ILIKE 'Madison';

4. Most frequent county names - "What are the three most frequent county names?":
   SELECT name, COUNT(*) as count FROM counties GROUP BY name ORDER BY count DESC LIMIT 3;

5. Out-of-scope error - "Show me provinces in Canada":
   DETECT: If question mentions non-US geography (Canada, provinces, etc.) → Return error message

6. List counties in state - "List all counties in Florida":
   SELECT name, state_name FROM counties WHERE state_name ILIKE 'Florida' ORDER BY name;

7. List by abbreviation - "List all counties in WA":
   SELECT name, state_name FROM counties WHERE state_abbr ILIKE 'WA' ORDER BY name;

-- PATTERN MATCHING --
8. Starts with pattern - "Counties starting with 'San ' in California":
   SELECT name, state_name FROM counties WHERE state_name ILIKE 'California' AND name ILIKE 'San %';

9. Name equals state - "Counties whose name equals their state":
   SELECT c.name, c.state_name FROM counties c WHERE LOWER(c.name) = LOWER(c.state_name);

10. Multi-word names - "Counties with multi-word names in Minnesota":
    SELECT name, state_name FROM counties WHERE state_name ILIKE 'Minnesota' AND name LIKE '% %';

-- GEOGRAPHIC CONSTRAINTS --
11. Counties touching another state - "Counties in California that touch Nevada":
    SELECT DISTINCT c1.name, c1.state_name 
    FROM counties c1, counties c2 
    WHERE c1.state_name ILIKE 'California' AND c2.state_name ILIKE 'Nevada' 
      AND ST_Touches(c1.geom, c2.geom);

-- MEASUREMENTS --
12. Land area of county - "Land area of Riverside County, California":
    SELECT name, state_name, aland/1000000 as area_km2, aland*0.000000386102 as area_mi2
    FROM counties WHERE name ILIKE 'Riverside' AND state_name ILIKE 'California';

13. Rank by area - "Rank all counties in Arizona by area":
    SELECT name, state_name, aland/1000000 as area_km2 
    FROM counties WHERE state_name ILIKE 'Arizona' ORDER BY area_km2 DESC;

14. Smallest county - "Which county in NC has the smallest area?":
    SELECT name, state_name, aland/1000000 as area_km2 
    FROM counties WHERE state_abbr ILIKE 'NC' AND aland > 0 ORDER BY area_km2 ASC LIMIT 1;

15. Area filter - "Counties with area < 100 mi²":
    SELECT name, state_name, aland*0.000000386102 as area_mi2 
    FROM counties WHERE aland*0.000000386102 < 100 ORDER BY area_mi2 ASC;

16. Perimeter length - "Perimeter of Orange County, California":
    SELECT name, state_name, ST_Perimeter(geom::geography)/1000 as perimeter_km, ST_Perimeter(geom::geography)*0.000621371 as perimeter_mi
    FROM counties WHERE name ILIKE 'Orange' AND state_name ILIKE 'California';

17. Perimeter filter - "Counties in South Dakota with perimeter > 800 mi":
    SELECT name, state_name, ST_Perimeter(geom::geography)*0.000621371 as perimeter_mi
    FROM counties WHERE state_name ILIKE 'South Dakota' AND ST_Perimeter(geom::geography)*0.000621371 > 800;

-- ADVANCED GEOMETRY --
18. Interior rings (holes) - "Counties with holes in their geometry":
    SELECT name, state_name, ST_NumInteriorRings((ST_Dump(geom)).geom) as num_holes
    FROM counties WHERE ST_NumInteriorRings((ST_Dump(geom)).geom) > 0;

19. Multipart polygons - "Counties that are non-contiguous MultiPolygons":
    SELECT name, state_name, ST_NumGeometries(geom) as num_parts
    FROM counties WHERE ST_NumGeometries(geom) > 1;

20. Centroid outside polygon - "Counties whose centroid falls outside the polygon":
    SELECT name, state_name 
    FROM counties WHERE NOT ST_Within(ST_Centroid(geom), geom);

21. Largest hole area - "For Ramsey County, MN, area of largest interior hole":
    SELECT name, state_name, 
           ST_Area((ST_InteriorRingN((ST_Dump(geom)).geom, 1))::geography)/1000000 as hole_area_km2
    FROM counties WHERE name ILIKE 'Ramsey' AND state_abbr ILIKE 'MN';

-- NEIGHBOR ANALYSIS --
22. Count neighbors - "How many neighbors does Utah County, UT have?":
    SELECT COUNT(DISTINCT c2.id) as neighbor_count
    FROM counties c1, counties c2 
    WHERE c1.name ILIKE 'Utah' AND c1.state_abbr ILIKE 'UT' 
      AND ST_Touches(c1.geom, c2.geom) AND c1.id != c2.id;

23. County with most neighbors - "Which county in AL has the most neighbors?":
    SELECT c1.name, c1.state_name, COUNT(DISTINCT c2.id) as neighbor_count
    FROM counties c1, counties c2 
    WHERE c1.state_abbr ILIKE 'AL' AND ST_Touches(c1.geom, c2.geom) AND c1.id != c2.id
    GROUP BY c1.id, c1.name, c1.state_name ORDER BY neighbor_count DESC LIMIT 1;

24. Exact neighbor count - "Counties in CA with exactly two neighbors":
    SELECT c1.name, c1.state_name, COUNT(DISTINCT c2.id) as neighbor_count
    FROM counties c1, counties c2 
    WHERE c1.state_abbr ILIKE 'CA' AND ST_Touches(c1.geom, c2.geom) AND c1.id != c2.id
    GROUP BY c1.id, c1.name, c1.state_name HAVING COUNT(DISTINCT c2.id) = 2;

-- STATE-LEVEL QUERIES --
25. Largest state by area - "Which state has the largest area?" or "Largest state":
    SELECT state_name, SUM(aland)/1000000 as total_area_km2 
    FROM counties GROUP BY state_name ORDER BY total_area_km2 DESC LIMIT 1;

26. Smallest state by area - "Which state has the smallest area?" or "Smallest state":
    SELECT state_name, SUM(aland)/1000000 as total_area_km2 
    FROM counties GROUP BY state_name ORDER BY total_area_km2 ASC LIMIT 1;

27. State rankings - "Rank all states by area":
    SELECT state_name, SUM(aland)/1000000 as total_area_km2 
    FROM counties GROUP BY state_name ORDER BY total_area_km2 DESC;

28. State with most counties - "Which state has the most counties?":
    SELECT state_name, COUNT(*) as county_count 
    FROM counties GROUP BY state_name ORDER BY county_count DESC LIMIT 1;

CRITICAL: For queries asking to "show", "visualize", or "map", ALWAYS include geometry columns!
CRITICAL: There is NO states table - use state_name/state_abbr columns from counties table!
CRITICAL: For "counties in [state]" questions, use WHERE state_name = '[State]' NOT ST_Within or ST_Contains!
"""

# =============================================================================
# TEXT & STATE UTILITIES
# =============================================================================

STATE_NAME_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "U.S. Virgin Islands": "VI",
}

STATE_ABBR_TO_NAME = {abbr: name for name, abbr in STATE_NAME_TO_ABBR.items()}

STATE_NORMALIZER = {}
for state_name, abbr in STATE_NAME_TO_ABBR.items():
    STATE_NORMALIZER[state_name.lower()] = state_name
    STATE_NORMALIZER[abbr.lower()] = state_name
STATE_NORMALIZER.update({
    "washington dc": "District of Columbia",
    "washington d.c.": "District of Columbia",
    "dc": "District of Columbia",
    "d.c.": "District of Columbia",
    "district of columbia": "District of Columbia",
    "virgin islands": "U.S. Virgin Islands",
    "us virgin islands": "U.S. Virgin Islands",
    "u.s. virgin islands": "U.S. Virgin Islands",
})

COUNTY_SUFFIXES = [
    " county",
    " parish",
    " borough",
    " census area",
    " municipality",
    " city and borough",
    " municipality of",
    " city",
    " municipality",
    " municipio",
    " borough",
]

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21,
    "twenty two": 22,
    "twenty-two": 22,
    "twenty three": 23,
    "twenty-three": 23,
    "twenty four": 24,
    "twenty-four": 24,
    "twenty five": 25,
    "twenty-five": 25,
}

AREA_KM2_EXPR = "ST_Area(geom::geography)/1000000"
AREA_MI2_EXPR = "ST_Area(geom::geography)/2589988.110336"
PERIM_KM_EXPR = "ST_Perimeter(geom::geography)/1000"
PERIM_MI_EXPR = "ST_Perimeter(geom::geography)/1609.34"


def sql_literal(value: str) -> str:
    """Return a safely quoted SQL literal."""
    return "'" + value.replace("'", "''") + "'"


def extract_bracketed_values(text: str) -> List[str]:
    """Return strings enclosed in [brackets]."""
    return re.findall(r'\[([^\]]+)\]', text)


def strip_quotes(value: str) -> str:
    """Strip whitespace and common quote characters."""
    return value.strip().strip(" \"'`“”‘’")


def remove_outer_quotes(value: Optional[str]) -> str:
    """Remove only the outermost quote characters without trimming interior whitespace."""
    if value is None:
        return ""
    cleaned = value
    quote_chars = "\"'`“”‘’"
    while cleaned and cleaned[0] in quote_chars:
        cleaned = cleaned[1:]
    while cleaned and cleaned[-1] in quote_chars:
        cleaned = cleaned[:-1]
    return cleaned


def clean_free_text_value(value: Optional[str]) -> str:
    """Normalize arbitrary free-text values extracted from brackets."""
    if value is None:
        return ""
    return remove_outer_quotes(value).strip()


def normalize_county_name(value: Optional[str]) -> Optional[str]:
    """Normalize county name by removing suffixes such as 'County'."""
    if not value:
        return None
    cleaned = strip_quotes(value)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    lower_cleaned = cleaned.lower()
    updated = True
    while updated:
        updated = False
        for suffix in COUNTY_SUFFIXES:
            if lower_cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].strip()
                lower_cleaned = cleaned.lower()
                updated = True
    return cleaned if cleaned else None


def normalize_state(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Normalize a state name or abbreviation."""
    if not value:
        return (None, None)
    cleaned = strip_quotes(value)
    cleaned = cleaned.replace(".", "").lower().strip()
    cleaned = cleaned.replace("state of ", "").replace("commonwealth of ", "")
    state_name = STATE_NORMALIZER.get(cleaned)
    if state_name:
        return state_name, STATE_NAME_TO_ABBR.get(state_name)
    return (None, None)


def parse_county_location(value: Optional[str]) -> Dict[str, Optional[str]]:
    """Parse a '[County, State]' style string into components."""
    result: Dict[str, Optional[str]] = {"county": None, "state_name": None, "state_abbr": None}
    if not value:
        return result
    cleaned = value.strip()
    parts = [part.strip() for part in cleaned.split(",")]
    if parts:
        preliminary_state, preliminary_abbr = normalize_state(parts[0])
        contains_county_keyword = any(word in parts[0].lower() for word in ["county", "parish", "borough", "census area"])
        if preliminary_state and not contains_county_keyword:
            result["state_name"] = preliminary_state
            result["state_abbr"] = preliminary_abbr
            result["county"] = None
        else:
            result["county"] = normalize_county_name(parts[0])
    if len(parts) > 1:
        state_name, state_abbr = normalize_state(parts[1])
        result["state_name"] = state_name or result["state_name"]
        result["state_abbr"] = state_abbr or result["state_abbr"]
    return result


def lower_match(column: str, value: str) -> str:
    """Case-insensitive equality predicate."""
    return f"LOWER({column}) = LOWER({sql_literal(value)})"


def state_filter_clause(state_name: Optional[str], state_abbr: Optional[str]) -> Optional[str]:
    """Return a SQL predicate limiting rows to a specific state."""
    if state_name:
        return f"state_name = {sql_literal(state_name)}"
    if state_abbr:
        return f"state_abbr = {sql_literal(state_abbr)}"
    return None


def state_filter_with_alias(state_name: Optional[str], state_abbr: Optional[str], alias: str) -> Optional[str]:
    """Return a SQL predicate for a specific table alias."""
    if state_name:
        return f"{alias}.state_name = {sql_literal(state_name)}"
    if state_abbr:
        return f"{alias}.state_abbr = {sql_literal(state_abbr)}"
    return None


def geometry_columns(extra_columns: Optional[List[str]] = None, alias: Optional[str] = None) -> str:
    """Return a comma-separated list of geometry-related columns for map display."""
    column_prefix = f"{alias}." if alias else ""
    geom_ref = f"{alias}.geom" if alias else "geom"
    base_columns = [
        f"{column_prefix}name AS name" if alias else "name",
        f"{column_prefix}state_name AS state_name" if alias else "state_name",
        f"{column_prefix}state_abbr AS state_abbr" if alias else "state_abbr",
        f"{column_prefix}geoid AS geoid" if alias else "geoid",
        f"ST_AsGeoJSON({geom_ref}) AS geom_geojson",
        f"ST_Y(ST_Centroid({geom_ref})) AS centroid_lat",
        f"ST_X(ST_Centroid({geom_ref})) AS centroid_lon",
    ]
    if extra_columns:
        base_columns.extend(extra_columns)
    return ",\n       ".join(base_columns)


def build_geometry_query(
    where_clause: str,
    order_clause: Optional[str] = None,
    limit_clause: Optional[int] = None,
    extra_columns: Optional[List[str]] = None,
) -> str:
    """Build a SELECT query that includes geometry + centroid information."""
    query = f"SELECT {geometry_columns(extra_columns)}\nFROM counties\nWHERE {where_clause}"
    if order_clause:
        query += f"\n{order_clause}"
    if limit_clause is not None:
        query += f"\nLIMIT {limit_clause}"
    return query


def parse_numeric_token(value: Optional[str]) -> Optional[int]:
    """Parse a numeric value that may be spelled out (e.g., 'three')."""
    if not value:
        return None
    cleaned = re.sub(r'[^0-9a-zA-Z\s-]', ' ', value).strip().lower()
    if not cleaned:
        return None
    if cleaned.isdigit():
        return int(cleaned)
    if cleaned in NUMBER_WORDS:
        return NUMBER_WORDS[cleaned]
    for token in cleaned.split():
        if token.isdigit():
            return int(token)
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return None


def parse_numeric_from_brackets(bracket_values: List[str]) -> Optional[int]:
    """Return the first numeric value found inside any bracket."""
    for value in bracket_values:
        parsed = parse_numeric_token(value)
        if parsed is not None:
            return parsed
    return None


def parse_measurement_condition(value: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse expressions like 'area < 100 mi²' or 'perimeter > 800 mi'.
    Returns metric (area/perimeter), operator, numeric value, and canonical unit.
    """
    if not value:
        return None
    cleaned = value.lower()
    cleaned = cleaned.replace("sq.", "square ").replace("sq ", "square ")
    cleaned = cleaned.replace("²", "2").replace("^2", "2")
    cleaned = cleaned.replace("square miles", "mi2").replace("square mile", "mi2")
    cleaned = cleaned.replace("square kilometers", "km2").replace("square kilometer", "km2")
    match = re.search(r'(area|perimeter)\s*(<=|>=|<|>)\s*([0-9]+(?:\.[0-9]+)?)\s*([a-z0-9\s/]*)', cleaned)
    if not match:
        return None
    metric = match.group(1)
    operator = match.group(2)
    value_num = float(match.group(3))
    unit_hint = match.group(4).strip()
    if metric == "area":
        if "mi" in unit_hint or "mile" in unit_hint:
            unit = "mi2"
        else:
            unit = "km2"
    else:
        if "mi" in unit_hint or "mile" in unit_hint:
            unit = "mi"
        else:
            unit = "km"
    return {"metric": metric, "operator": operator, "value": value_num, "unit": unit}


def build_county_where_clause(county_name: Optional[str], state_name: Optional[str], state_abbr: Optional[str]) -> Optional[str]:
    """Build a WHERE clause targeting a county (optionally scoped to a state)."""
    if not county_name:
        return None
    clauses = [lower_match("name", county_name)]
    state_clause = state_filter_clause(state_name, state_abbr)
    if state_clause:
        clauses.append(state_clause)
    return " AND ".join(clauses)


def build_county_alias_clause(alias: str, county_name: Optional[str], state_name: Optional[str], state_abbr: Optional[str]) -> Optional[str]:
    """Same as build_county_where_clause but scoped to a specific table alias."""
    if not county_name:
        return None
    clauses = [lower_match(f"{alias}.name", county_name)]
    if state_name:
        clauses.append(f"{alias}.state_name = {sql_literal(state_name)}")
    elif state_abbr:
        clauses.append(f"{alias}.state_abbr = {sql_literal(state_abbr)}")
    return " AND ".join(clauses)

# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

def fix_sql_aggregation(sql: str, question: str) -> str:
    """
    Post-process SQL to fix missing aggregations for "total", "sum", etc.
    """
    sql_upper = sql.upper()
    question_lower = question.lower()
    
    # Check for out-of-scope queries (non-US geography)
    out_of_scope_terms = ['canada', 'canadian', 'province', 'provinces', 'mexico', 'mexican', 'europe', 'asia', 'africa', 'australia', 'britain', 'england']
    if any(term in question_lower for term in out_of_scope_terms):
        return "SELECT 'ERROR: This database only contains US county data. Queries about Canada, provinces, or other non-US geography are out of scope.' as error_message"
    
    # Check if question asks for total/sum but SQL doesn't have aggregation
    needs_sum = any(word in question_lower for word in ['total', 'sum', 'combined', 'all together', 'entire'])
    has_sum = 'SUM(' in sql_upper or 'SUM (' in sql_upper
    has_area = 'ST_AREA' in sql_upper
    
    # If question asks for total area but SQL doesn't use SUM, fix it
    if needs_sum and has_area and not has_sum:
        # Find the SELECT clause and extract area expression
        # Pattern: SELECT ... ST_Area(...) ... FROM
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Find ST_Area expression
            area_match = re.search(r'ST_Area\([^)]+\)/1000000', select_clause, re.IGNORECASE)
            if area_match:
                area_expr = area_match.group(0)
                # Check if WHERE clause exists
                where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER|\s+LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
                where_clause = ''
                if where_match:
                    where_clause = ' WHERE ' + where_match.group(1).strip()
                
                # Build new SQL with SUM aggregation
                sql = f"SELECT SUM({area_expr}) as total_area_km2 FROM counties{where_clause}"
                
                # Remove ORDER BY and LIMIT (not needed for aggregation)
                sql = re.sub(r'\s+ORDER\s+BY\s+[^\s]+', '', sql, flags=re.IGNORECASE)
                sql = re.sub(r'\s+LIMIT\s+\d+', '', sql, flags=re.IGNORECASE)
    
    return sql

def extract_state_from_question(question: str) -> Optional[str]:
    """Extract a state name mentioned in the question (outside of explicit brackets)."""
    # Prioritize bracketed values
    for value in extract_bracketed_values(question):
        state_name, _ = normalize_state(value)
        if state_name:
            return state_name

    # Look for uppercase two-letter abbreviations (e.g., "in NC")
    for token in re.findall(r'\b[A-Z]{2}\b', question):
        state_name = STATE_ABBR_TO_NAME.get(token)
        if state_name:
            return state_name

    # Fallback to searching for full state names
    question_clean = re.sub(r'[^a-z\s]', ' ', question.lower())
    for state_name in sorted(STATE_NAME_TO_ABBR.keys(), key=len, reverse=True):
        if re.search(r'\b' + re.escape(state_name.lower()) + r'\b', question_clean):
            return state_name

    return None

def is_state_adjacency_question(question: str) -> bool:
    """Check if question is asking about states bordering/adjacent to another state"""
    question_lower = question.lower()
    state_adjacency_keywords = [
        'states bordering', 'states border', 'states borders', 'states adjacent', 'states adjacent to',
        'which states border', 'which states borders', 'which states are adjacent', 
        'what states border', 'what states borders', 'what states are adjacent',
        'states that border', 'states that borders', 'states that are adjacent',
        'borders texas', 'borders california', 'borders new york',  # common patterns
        'adjacent to', 'adjacent states'
    ]
    # Also check for pattern: "what/which states [verb] [state]"
    if re.search(r'(what|which)\s+states?\s+(border|borders|adjacent)', question_lower):
        return True
    return any(keyword in question_lower for keyword in state_adjacency_keywords)

def fix_invalid_table_references(sql: str, question: str) -> str:
    """
    Post-process SQL to fix references to non-existent tables (like USStateDB, states table).
    Detects and fixes queries that try to JOIN with non-existent tables.
    Also fixes queries that try to create geometry objects with ST_GeomFromText.
    """
    sql_upper = sql.upper()
    
    # Detect attempts to create geometry objects (forbidden)
    if 'ST_GEOMFROMTEXT' in sql_upper or 'ST_MAKEPOINT' in sql_upper:
        # Check if this is a state adjacency query using multiple methods
        question_lower = question.lower()
        state_adjacency_patterns = [
            r'(what|which)\s+states?\s+(border|borders|adjacent)',
            r'states?\s+(border|borders|adjacent)\s+to',
            r'states?\s+that\s+(border|borders)',
        ]
        
        is_adjacency = is_state_adjacency_question(question) or any(
            re.search(pattern, question_lower) for pattern in state_adjacency_patterns
        )
        
        # Also check if SQL contains "border" or "adjacent" keywords
        if 'border' in question_lower or 'adjacent' in question_lower or 'borders' in question_lower:
            is_adjacency = True
        
        if is_adjacency:
            state = extract_state_from_question(question)
            if state:
                return f"SELECT DISTINCT c2.state_name, c2.state_abbr FROM counties c1, counties c2 WHERE c1.state_name = '{state}' AND ST_Touches(c1.geom, c2.geom) AND c2.state_abbr != c1.state_abbr ORDER BY c2.state_name"
            else:
                return "SELECT DISTINCT c2.state_name, c2.state_abbr FROM counties c1, counties c2 WHERE ST_Touches(c1.geom, c2.geom) AND c2.state_abbr != c1.state_abbr ORDER BY c2.state_name LIMIT 0"
        # If not state adjacency, return error message will be handled by execute_sql_query
    
    # Detect references to non-existent tables
    invalid_table_patterns = [
        r'JOIN\s+USStateDB',
        r'JOIN\s+states',
        r'FROM\s+states',
        r'JOIN\s+state_boundaries',
        r'USStateDB',
        r'\bstates\s+s\b',  # states table with alias 's'
    ]
    
    has_invalid_table = any(re.search(pattern, sql_upper, re.IGNORECASE) for pattern in invalid_table_patterns)
    
    if has_invalid_table:
        # Try to extract state name from question or SQL
        state = extract_state_from_question(question)
        
        # Check if this is a "counties in state" query
        question_lower = question.lower()
        is_counties_in_state = any(phrase in question_lower for phrase in [
            'counties in', 'counties of', 'counties within', 'counties that are in'
        ])
        
        # Check if this is trying to use ST_Within/ST_Contains incorrectly
        has_st_within = 'ST_WITHIN' in sql_upper or 'ST_Contains' in sql_upper
        
        if is_counties_in_state or has_st_within:
            # Replace with simple WHERE clause
            if state:
                # Extract what columns to select
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                if select_match:
                    select_clause = select_match.group(1).strip()
                    # If selecting from invalid table, default to county name
                    if 's.' in select_clause.lower() or 'states.' in select_clause.lower():
                        select_clause = 'name, state_name'
                    return f"SELECT {select_clause} FROM counties WHERE state_name = '{state}' ORDER BY name"
                else:
                    return f"SELECT name, state_name FROM counties WHERE state_name = '{state}' ORDER BY name"
            else:
                return "SELECT name, state_name FROM counties ORDER BY name LIMIT 100"
        
        # If it's a state adjacency query, fix it
        if is_state_adjacency_question(question):
            if state:
                return f"SELECT DISTINCT c2.state_name, c2.state_abbr FROM counties c1, counties c2 WHERE c1.state_name = '{state}' AND ST_Touches(c1.geom, c2.geom) AND c2.state_abbr != c1.state_abbr ORDER BY c2.state_name"
            else:
                return "SELECT DISTINCT c2.state_name, c2.state_abbr FROM counties c1, counties c2 WHERE ST_Touches(c1.geom, c2.geom) AND c2.state_abbr != c1.state_abbr ORDER BY c2.state_name LIMIT 0"
    
    # Also check for incorrect ST_Within/ST_Contains usage for state filtering
    if 'ST_WITHIN' in sql_upper or 'ST_CONTAINS' in sql_upper:
        question_lower = question.lower()
        if any(phrase in question_lower for phrase in ['counties in', 'counties of', 'counties within']):
            state = extract_state_from_question(question)
            if state:
                # Extract SELECT clause
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                select_clause = 'name, state_name'
                if select_match:
                    # Try to preserve what was being selected
                    original_select = select_match.group(1).strip()
                    if 'name' in original_select.lower():
                        select_clause = original_select.split(',')[0].strip() + ', state_name'
                
                return f"SELECT {select_clause} FROM counties WHERE state_name = '{state}' ORDER BY name"
    
    return sql

def fetch_geometry_for_feature(
    conn,
    name: Optional[str],
    state_name: Optional[str],
    state_abbr: Optional[str],
    geoid: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch GeoJSON polygon and centroid for a county or entire state (union of counties)."""
    try:
        cur = conn.cursor()
        if geoid:
            query = """
                SELECT name, state_name,
                       ST_AsGeoJSON(geom) AS geom_geojson,
                       ST_Y(ST_Centroid(geom)) AS centroid_lat,
                       ST_X(ST_Centroid(geom)) AS centroid_lon
                FROM counties
                WHERE geoid = %s
                LIMIT 1
            """
            params: List[Any] = [geoid]
        elif name:
            query = """
                SELECT name, state_name,
                       ST_AsGeoJSON(geom) AS geom_geojson,
                       ST_Y(ST_Centroid(geom)) AS centroid_lat,
                       ST_X(ST_Centroid(geom)) AS centroid_lon
                FROM counties
                WHERE name = %s
            """
            params: List[Any] = [name]
            if state_name:
                query += " AND state_name = %s"
                params.append(state_name)
            if state_abbr:
                query += " AND state_abbr = %s"
                params.append(state_abbr)
            query += " LIMIT 1"
        elif state_name or state_abbr:
            query = """
                SELECT %s AS name, %s AS state_name,
                       ST_AsGeoJSON(ST_Union(geom)) AS geom_geojson,
                       ST_Y(ST_Centroid(ST_Union(geom))) AS centroid_lat,
                       ST_X(ST_Centroid(ST_Union(geom))) AS centroid_lon
                FROM counties
                WHERE 1=1
            """
            params = [state_name or state_abbr or "Unknown", state_name or state_abbr or "Unknown"]
            if state_name:
                query += " AND state_name = %s"
                params.append(state_name)
            if state_abbr:
                query += " AND state_abbr = %s"
                params.append(state_abbr)
        else:
            return None

        cur.execute(query, params)
        result = cur.fetchone()
        cur.close()

        if not result or not result[2]:
            return None

        return {
            "name": result[0],
            "state_name": result[1],
            "geom_geojson": result[2],
            "centroid_lat": float(result[3]) if result[3] is not None else None,
            "centroid_lon": float(result[4]) if result[4] is not None else None,
        }
    except Exception:
        return None

def build_map_features(conn, rows: List[tuple], columns: List[str]) -> List[Dict[str, Any]]:
    """Build map features (polygon + centroid) from query results."""
    map_features: List[Dict[str, Any]] = []
    if not rows or not columns:
        return map_features
    max_features = min(len(rows), 25)
    lower_columns = [col.lower() for col in columns]

    name_keys = [
        'name', 'county_name', 'county', 'countyname', 'county_nm',
        'parish_name', 'parish', 'borough_name', 'borough', 'county_label'
    ]
    state_name_keys = [
        'state_name', 'state', 'statename', 'state_nm', 'state_full',
        'state_full_name', 'statefullname'
    ]
    state_abbr_keys = ['state_abbr', 'stateabbr', 'state_abbreviation', 'state_code', 'state_cd', 'st_abbr', 'stusps']
    geoid_keys = ['geoid', 'geoid10']
    geojson_keys = ['geom_geojson', 'geojson', 'geometry', 'st_asgeojson']
    lat_keys = ['centroid_lat', 'intptlat', 'lat', 'latitude', 'centroidlatitude']
    lon_keys = ['centroid_lon', 'intptlon', 'lon', 'longitude', 'centroidlongitude']

    def first_value(row_map: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        for key in keys:
            if key in row_map:
                value = row_map[key]
                if value is not None and value != '':
                    return value
        # Try fuzzy matching if exact match fails
        for key in keys:
            for row_key in row_map:
                if key in row_key or row_key in key:
                    value = row_map[row_key]
                    if value is not None and value != '':
                        return value
        return None

    def to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for row in rows[:max_features]:
        row_map = {lower_columns[idx]: row[idx] for idx in range(len(lower_columns))}

        name = first_value(row_map, name_keys)
        state_name = first_value(row_map, state_name_keys)
        state_abbr = first_value(row_map, state_abbr_keys)
        geoid = first_value(row_map, geoid_keys)

        geom_geojson = first_value(row_map, geojson_keys)
        if isinstance(geom_geojson, bytes):
            try:
                geom_geojson = geom_geojson.decode('utf-8')
            except Exception:
                geom_geojson = None

        centroid_lat = to_float(first_value(row_map, lat_keys))
        centroid_lon = to_float(first_value(row_map, lon_keys))

        if geom_geojson is None or centroid_lat is None or centroid_lon is None:
            fetched = fetch_geometry_for_feature(conn, name, state_name, state_abbr, geoid)
            if fetched:
                geom_geojson = geom_geojson or fetched.get('geom_geojson')
                centroid_lat = centroid_lat if centroid_lat is not None else fetched.get('centroid_lat')
                centroid_lon = centroid_lon if centroid_lon is not None else fetched.get('centroid_lon')
                name = name or fetched.get('name')
                state_name = state_name or fetched.get('state_name')

        if geom_geojson and centroid_lat is not None and centroid_lon is not None:
            feature: Dict[str, Any] = {
                'name': name or state_name or state_abbr or geoid or "Location",
                'state_name': state_name,
                'state_abbr': state_abbr,
                'geom_geojson': geom_geojson,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
            }
            map_features.append(feature)

    return map_features

def render_map(features: List[Dict[str, Any]]):
    """Render map with polygons and centroid markers using PyDeck."""
    if not features:
        return

    polygon_data = []
    centroid_data = []
    avg_lat = 37.0902
    avg_lon = -95.7129

    for feat in features:
        try:
            geometry = json.loads(feat['geom_geojson'])
        except Exception:
            continue

        polygon_data.append({
            "name": feat.get("name"),
            "state_name": feat.get("state_name"),
            "geometry": geometry,
        })
        centroid_data.append({
            "name": feat.get("name"),
            "state_name": feat.get("state_name"),
            "lat": feat.get("centroid_lat"),
            "lon": feat.get("centroid_lon"),
        })

    if centroid_data:
        avg_lat = sum(item['lat'] for item in centroid_data if item['lat'] is not None) / len(centroid_data)
        avg_lon = sum(item['lon'] for item in centroid_data if item['lon'] is not None) / len(centroid_data)

    if MAPBOX_TOKEN:
        pdk.settings.mapbox_api_key = MAPBOX_TOKEN

    map_style = "mapbox://styles/mapbox/light-v9" if MAPBOX_TOKEN else "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

    polygon_layer = pdk.Layer(
        "GeoJsonLayer",
        polygon_data,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="[0, 128, 255, 80]",
        get_line_color="[0, 128, 255]",
        line_width_min_pixels=2,
        auto_highlight=True,
    )

    centroid_layer = pdk.Layer(
        "ScatterplotLayer",
        centroid_data,
        pickable=True,
        get_position='[lon, lat]',
        get_fill_color=[255, 0, 0],
        get_radius=30000,
    )

    tooltip = {"html": "<b>{name}</b><br/>{state_name}", "style": {"backgroundColor": "steelblue", "color": "white"}}

    deck = pdk.Deck(
        layers=[polygon_layer, centroid_layer],
        initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=4, pitch=0),
        tooltip=tooltip,
        map_style=map_style,
    )

    st.pydeck_chart(deck)


def generate_structured_sql(question: str) -> Optional[str]:
    """
    Handle common natural language patterns without calling the LLM.
    Returns an SQL string when a deterministic template is available.
    """
    q_lower = question.lower()
    bracket_values = extract_bracketed_values(question)
    parsed_brackets = [parse_county_location(value) for value in bracket_values]
    primary_location = parsed_brackets[0] if parsed_brackets else {"county": None, "state_name": None, "state_abbr": None}

    state_mentions: List[Tuple[Optional[str], Optional[str]]] = []
    for parsed in parsed_brackets:
        state_name = parsed.get("state_name")
        state_abbr = parsed.get("state_abbr")
        if state_name:
            state_mentions.append((state_name, state_abbr))
    if not state_mentions:
        extracted_state = extract_state_from_question(question)
        if extracted_state:
            state_mentions.append((extracted_state, STATE_NAME_TO_ABBR.get(extracted_state)))

    first_state_name = state_mentions[0][0] if state_mentions else None
    first_state_abbr = state_mentions[0][1] if state_mentions else None
    second_state_name = state_mentions[1][0] if len(state_mentions) > 1 else None
    second_state_abbr = state_mentions[1][1] if len(state_mentions) > 1 else None

    county_name = primary_location.get("county")
    county_state_name = primary_location.get("state_name")
    county_state_abbr = primary_location.get("state_abbr")

    area_condition: Optional[Dict[str, Any]] = None
    perimeter_condition: Optional[Dict[str, Any]] = None
    for value in bracket_values:
        parsed_condition = parse_measurement_condition(value)
        if parsed_condition:
            if parsed_condition["metric"] == "area" and area_condition is None:
                area_condition = parsed_condition
            elif parsed_condition["metric"] == "perimeter" and perimeter_condition is None:
                perimeter_condition = parsed_condition

    # Out-of-scope guard
    if any(keyword in q_lower for keyword in ["canada", "province", "provinces"]):
        message = "Error: This application only supports United States counties."
        return f"SELECT {sql_literal(message)} AS error_message"

    # Most frequent county names nationwide
    if "frequent county name" in q_lower or "common county name" in q_lower:
        limit_value = parse_numeric_from_brackets(bracket_values) or 10
        return (
            "SELECT name, COUNT(*) AS occurrences\n"
            "FROM counties\n"
            "GROUP BY name\n"
            "ORDER BY occurrences DESC, name ASC\n"
            f"LIMIT {limit_value}"
        )

    # Visualize specific county/county name
    show_keywords = ["show me", "visualize", "visualise", "display", "map of", "map me"]
    if county_name and any(keyword in q_lower for keyword in show_keywords):
        where_clause = build_county_where_clause(county_name, county_state_name, county_state_abbr)
        if where_clause:
            return build_geometry_query(where_clause, "ORDER BY state_name, name")

    # Count how many counties share a given name
    if (
        county_name
        and "how many" in q_lower
        and "counties" in q_lower
        and any(keyword in q_lower for keyword in ["called", "named"])
    ):
        where_clause = lower_match("name", county_name)
        return f"SELECT COUNT(*) AS county_count\nFROM counties\nWHERE {where_clause}"

    # Counties starting with a prefix in a state
    if ("start with" in q_lower or "starting with" in q_lower) and bracket_values:
        prefix_value = clean_free_text_value(bracket_values[0])
        target_state_name = None
        target_state_abbr = None
        if len(parsed_brackets) > 1 and parsed_brackets[1].get("state_name"):
            target_state_name = parsed_brackets[1]["state_name"]
            target_state_abbr = parsed_brackets[1]["state_abbr"]
        else:
            target_state_name = first_state_name
            target_state_abbr = first_state_abbr
        if prefix_value:
            prefix_pattern = sql_literal(f"{prefix_value}%")
            conditions = [f"name ILIKE {prefix_pattern}"]
            state_clause = state_filter_clause(target_state_name, target_state_abbr)
            if state_clause:
                conditions.insert(0, state_clause)
            where_clause = " AND ".join(conditions)
            return build_geometry_query(where_clause, "ORDER BY name")

    # Counties whose name equals their state
    if ("name equals their state" in q_lower) or (
        "name" in q_lower and "equals" in q_lower and "state" in q_lower and "county" in q_lower
    ):
        where_clause = "LOWER(name) = LOWER(state_name)"
        return build_geometry_query(where_clause, "ORDER BY state_name, name")

    # Multi-word county names
    if "multi-word" in q_lower or "multi word" in q_lower:
        conditions = ["name LIKE '% %'"]
        state_clause = state_filter_clause(first_state_name, first_state_abbr)
        if state_clause:
            conditions.insert(0, state_clause)
        where_clause = " AND ".join(conditions)
        return build_geometry_query(where_clause, "ORDER BY name")

    # Counties in one state touching another state
    adjacency_keywords = ["touch", "border", "adjacent"]
    if any(keyword in q_lower for keyword in adjacency_keywords) and len(state_mentions) >= 2:
        clause_a = state_filter_with_alias(first_state_name, first_state_abbr, "c1")
        clause_b = state_filter_with_alias(second_state_name, second_state_abbr, "c2")
        if clause_a and clause_b:
            return (
                f"SELECT DISTINCT {geometry_columns(alias='c1')}\n"
                "FROM counties c1\n"
                "JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)\n"
                f"WHERE {clause_a}\n"
                f"  AND {clause_b}\n"
                "ORDER BY c1.name"
            )

    # Rank counties in a state by area
    if "rank" in q_lower and "area" in q_lower:
        state_clause = state_filter_clause(first_state_name, first_state_abbr) or "1=1"
        order_clause = "ORDER BY area_km2 DESC"
        extra_cols = [f"{AREA_KM2_EXPR} AS area_km2", f"{AREA_MI2_EXPR} AS area_mi2"]
        return build_geometry_query(state_clause, order_clause, extra_columns=extra_cols)

    # Smallest area county in a state
    if ("smallest area" in q_lower or "least area" in q_lower) and (first_state_name or first_state_abbr):
        state_clause = state_filter_clause(first_state_name, first_state_abbr)
        extra_cols = [f"{AREA_KM2_EXPR} AS area_km2", f"{AREA_MI2_EXPR} AS area_mi2"]
        return build_geometry_query(state_clause, "ORDER BY area_km2 ASC", limit_clause=1, extra_columns=extra_cols)

    # Counties with area condition (nationwide or filtered)
    if area_condition:
        area_expr = AREA_MI2_EXPR if area_condition["unit"] == "mi2" else AREA_KM2_EXPR
        comparison = f"{area_expr} {area_condition['operator']} {area_condition['value']}"
        conditions = [comparison]
        state_clause = state_filter_clause(first_state_name, first_state_abbr)
        if state_clause and "nationwide" not in q_lower:
            conditions.insert(0, state_clause)
        extra_cols = [f"{AREA_KM2_EXPR} AS area_km2", f"{AREA_MI2_EXPR} AS area_mi2"]
        where_clause = " AND ".join(conditions)
        return build_geometry_query(where_clause, "ORDER BY area_km2 ASC", extra_columns=extra_cols)

    # Land area for a specific county
    if county_name and "land area" in q_lower:
        where_clause = build_county_where_clause(county_name, county_state_name, county_state_abbr)
        if where_clause:
            extra_cols = [f"{AREA_KM2_EXPR} AS area_km2", f"{AREA_MI2_EXPR} AS area_mi2"]
            return build_geometry_query(where_clause, extra_columns=extra_cols)

    # Counties with perimeter condition inside a state
    if perimeter_condition:
        perim_expr = PERIM_MI_EXPR if perimeter_condition["unit"] == "mi" else PERIM_KM_EXPR
        comparison = f"{perim_expr} {perimeter_condition['operator']} {perimeter_condition['value']}"
        conditions = [comparison]
        state_clause = state_filter_clause(first_state_name, first_state_abbr)
        if state_clause:
            conditions.insert(0, state_clause)
        extra_cols = [f"{PERIM_KM_EXPR} AS perimeter_km", f"{PERIM_MI_EXPR} AS perimeter_mi"]
        where_clause = " AND ".join(conditions)
        return build_geometry_query(where_clause, "ORDER BY perimeter_mi DESC", extra_columns=extra_cols)

    # Perimeter of a specific county
    if county_name and (("perimeter length" in q_lower) or ("perimeter of" in q_lower)):
        where_clause = build_county_where_clause(county_name, county_state_name, county_state_abbr)
        if where_clause:
            extra_cols = [f"{PERIM_KM_EXPR} AS perimeter_km", f"{PERIM_MI_EXPR} AS perimeter_mi"]
            return build_geometry_query(where_clause, extra_columns=extra_cols)

    # Counties with interior holes
    if "interior ring" in q_lower or "holes" in q_lower:
        extra_cols = ["ST_NumInteriorRings(geom) AS interior_rings"]
        where_clause = "ST_NumInteriorRings(geom) > 0"
        return build_geometry_query(where_clause, "ORDER BY interior_rings DESC, state_name, name", extra_columns=extra_cols)

    # Multipart counties (non-contiguous)
    if any(keyword in q_lower for keyword in ["multipart", "multi-part", "non-contiguous", "non contiguous", "noncontiguous"]):
        extra_cols = ["ST_NumGeometries(geom) AS part_count"]
        where_clause = "ST_NumGeometries(geom) > 1"
        return build_geometry_query(where_clause, "ORDER BY part_count DESC, state_name, name", extra_columns=extra_cols)

    # Centroids falling outside polygon
    if "centroid" in q_lower and ("outside" in q_lower or "label" in q_lower or "risk" in q_lower):
        where_clause = "NOT ST_Contains(geom, ST_Centroid(geom))"
        return build_geometry_query(where_clause, "ORDER BY state_name, name")

    # Largest interior hole for a county
    if county_name and "largest interior hole" in q_lower:
        county_clause = build_county_where_clause(county_name, county_state_name, county_state_abbr)
        if county_clause:
            return (
                "WITH target AS (\n"
                f"    SELECT geom FROM counties WHERE {county_clause} LIMIT 1\n"
                "), rings AS (\n"
                "    SELECT (ST_DumpRings(geom)).geom AS ring_geom,\n"
                "           (ST_DumpRings(geom)).path AS ring_path\n"
                "    FROM target\n"
                "), interior AS (\n"
                "    SELECT ring_geom\n"
                "    FROM rings\n"
                "    WHERE ring_path[array_length(ring_path, 1)] > 0\n"
                ")\n"
                "SELECT COALESCE(\n"
                "    MAX(ST_Area(ST_MakePolygon(ring_geom)::geography)/1000000),\n"
                "    0\n"
                ") AS largest_hole_area_km2\n"
                "FROM interior"
            )

    # Neighbor count for a specific county
    if county_name and "how many neighbors" in q_lower:
        county_clause = build_county_alias_clause("c1", county_name, county_state_name, county_state_abbr)
        if county_clause:
            return (
                "SELECT COUNT(DISTINCT c2.geoid) AS neighbor_count\n"
                "FROM counties c1\n"
                "JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)\n"
                f"WHERE {county_clause}"
            )

    # County with the most neighbors in a state
    if "most neighbors" in q_lower and (first_state_name or first_state_abbr):
        state_clause = state_filter_with_alias(first_state_name, first_state_abbr, "c1")
        if state_clause:
            return (
                "WITH neighbor_counts AS (\n"
                "    SELECT c1.geoid,\n"
                "           COUNT(DISTINCT c2.geoid) AS neighbor_count\n"
                "    FROM counties c1\n"
                "    JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)\n"
                f"    WHERE {state_clause}\n"
                "    GROUP BY c1.geoid\n"
                ")\n"
                f"SELECT {geometry_columns(extra_columns=['neighbor_counts.neighbor_count AS neighbor_count'], alias='c')}\n"
                "FROM neighbor_counts\n"
                "JOIN counties c ON c.geoid = neighbor_counts.geoid\n"
                "ORDER BY neighbor_counts.neighbor_count DESC, c.name\n"
                "LIMIT 1"
            )

    # Counties with an exact neighbor count in a state
    if "exactly two neighbors" in q_lower and (first_state_name or first_state_abbr):
        state_clause = state_filter_with_alias(first_state_name, first_state_abbr, "c1")
        if state_clause:
            return (
                "WITH neighbor_counts AS (\n"
                "    SELECT c1.geoid,\n"
                "           COUNT(DISTINCT c2.geoid) AS neighbor_count\n"
                "    FROM counties c1\n"
                "    JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)\n"
                f"    WHERE {state_clause}\n"
                "    GROUP BY c1.geoid\n"
                ")\n"
                f"SELECT {geometry_columns(extra_columns=['neighbor_counts.neighbor_count AS neighbor_count'], alias='c')}\n"
                "FROM neighbor_counts\n"
                "JOIN counties c ON c.geoid = neighbor_counts.geoid\n"
                "WHERE neighbor_count = 2\n"
                "ORDER BY c.name"
            )

    # List all counties in a state (default)
    if (
        "counties" in q_lower
        and "in" in q_lower
        and (first_state_name or first_state_abbr)
        and not any(keyword in q_lower for keyword in ["start with", "touch", "border", "adjacent", "perimeter", "area", "neighbors", "multi-word", "multi word", "rank"])
    ):
        state_clause = state_filter_clause(first_state_name, first_state_abbr)
        if state_clause:
            return build_geometry_query(state_clause, "ORDER BY name")

    return None


@tool
def convert_nl_to_sql(question: str) -> str:
    """
    Convert a natural language question about US counties to a PostgreSQL SQL query.
    
    Args:
        question: The natural language question about US counties
        
    Returns:
        A PostgreSQL SQL query string
    """
    question_lower = question.lower()

    structured_sql = generate_structured_sql(question)
    if structured_sql:
        return structured_sql
    
    # Pattern matching for state adjacency queries (before LLM call)
    # Check for patterns like "what states borders [state]", "which states border [state]", etc.
    state_adjacency_patterns = [
        r'(what|which)\s+states?\s+(border|borders|adjacent)',
        r'states?\s+(border|borders|adjacent)\s+to',
        r'states?\s+that\s+(border|borders)',
    ]
    
    is_adjacency = is_state_adjacency_question(question) or any(
        re.search(pattern, question_lower) for pattern in state_adjacency_patterns
    )
    
    if is_adjacency:
        state = extract_state_from_question(question)
        if state:
            return f"SELECT DISTINCT c2.state_name, c2.state_abbr FROM counties c1, counties c2 WHERE c1.state_name = '{state}' AND ST_Touches(c1.geom, c2.geom) AND c2.state_abbr != c1.state_abbr ORDER BY c2.state_name"
        else:
            # If no state found, return a query that will return no results with proper structure
            return "SELECT DISTINCT c2.state_name, c2.state_abbr FROM counties c1, counties c2 WHERE ST_Touches(c1.geom, c2.geom) AND c2.state_abbr != c1.state_abbr ORDER BY c2.state_name LIMIT 0"
    
    # Pattern matching for "counties in [state]" questions (before LLM call)
    counties_in_state_patterns = [
        'counties in', 'counties of', 'counties within', 'counties that are in',
        'list counties in', 'show counties in', 'what counties are in'
    ]
    if any(pattern in question_lower for pattern in counties_in_state_patterns):
        state = extract_state_from_question(question)
        if state:
            return f"SELECT name, state_name, ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon FROM counties WHERE state_name = '{state}' ORDER BY name"
    
    prompt = f"""{SCHEMA_INFO}

    Question: {question}

    You are an expert SQL query generator for a PostGIS spatial database.
    Given a natural language question, generate ONLY the SQL query without any explanation.
    IMPORTANT: BORDER/NEIGHBOR QUERIES
    - If the user asks for "bordering", "adjacent", "next to", "neighbors", "touching" of a county:
    - YOU MUST USE A SELF-JOIN on the counties table!
    - Pattern: SELECT c2.name, c2.state_name, ST_AsGeoJSON(c2.geom) as geom_geojson, ST_Y(ST_Centroid(c2.geom)) as centroid_lat, ST_X(ST_Centroid(c2.geom)) as centroid_lon FROM counties c1, counties c2 WHERE c1.name ILIKE '%[Target]%' AND ST_Touches(c1.geom, c2.geom) AND c1.id != c2.id
    - NOTE: The 'name' column usually does NOT contain "County" (e.g. 'Los Angeles'). Use wildcards (e.g. '%Los Angeles%') or remove "County" from the target name.
    
    IMPORTANT: If the user asks to "show", "map", "visualize", or "display", YOU MUST SELECT GEOMETRY COLUMNS:
    ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon
    
    IMPORTANT: Use ILIKE for string comparisons to be case-insensitive (e.g. state_name ILIKE 'california').
    State names are stored as Title Case (e.g. 'California', 'New York').

    IMPORTANT: AGGREGATION & MAPS
    - If the user asks for "count", "how many", "total area", or "sum" for a region (e.g. "counties in California", "area of Texas"):
    - DO NOT use COUNT() or SUM() in SQL!
    - INSTEAD, SELECT the individual rows (name, state, area, geom) so they can be mapped!
    - Python will calculate the total/count for you.
    - Example: "How many counties in CA?" -> SELECT name, state_name, ST_AsGeoJSON(geom) as geom_geojson... FROM counties WHERE state_abbr='CA'
    - Example: "Total area of Texas?" -> SELECT name, state_name, aland, ST_AsGeoJSON(geom) as geom_geojson... FROM counties WHERE state_name='Texas'
    """

    try:
        # Try Ollama via HTTP API (for Railway deployment)
        ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        
        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                sql = result.get('response', '').strip()
                # Clean up SQL (remove markdown code blocks if present)
                sql = sql.replace('```sql', '').replace('```', '').strip()
                return sql
            else:
                print(f"Ollama error: {response.text}")
                
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            
        # Fallback to OpenAI if Ollama fails or returns error
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            try:
                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_key)
                response = llm.invoke(prompt)
                sql = response.content.strip()
                sql = sql.replace('```sql', '').replace('```', '').strip()
                if sql:
                    return sql
            except Exception as e:
                st.warning(f"OpenAI API error: {e}")
        
        # Final fallback - generate a basic query based on the question
        if 'count' in question_lower or 'how many' in question_lower:
            state = extract_state_from_question(question)
            if state:
                return f"SELECT COUNT(*) FROM counties WHERE state_name = '{state}'"
            else:
                return "SELECT COUNT(*) FROM counties"
        elif 'total' in question_lower and 'area' in question_lower:
            state = extract_state_from_question(question)
            if state:
                return f"SELECT SUM(ST_Area(geom::geography)/1000000) as total_area_km2 FROM counties WHERE state_name = '{state}'"
            else:
                return "SELECT SUM(ST_Area(geom::geography)/1000000) as total_area_km2 FROM counties"
        elif 'state' in question_lower and ('largest' in question_lower or 'biggest' in question_lower):
            # State-level query: largest state by total area
            return "SELECT state_name, SUM(aland)/1000000 as total_area_km2 FROM counties GROUP BY state_name ORDER BY total_area_km2 DESC LIMIT 1"
        elif 'state' in question_lower and 'smallest' in question_lower:
            # State-level query: smallest state by total area
            return "SELECT state_name, SUM(aland)/1000000 as total_area_km2 FROM counties GROUP BY state_name ORDER BY total_area_km2 ASC LIMIT 1"
        elif 'largest' in question_lower or 'biggest' in question_lower:
            # County-level query: largest counties
            return "SELECT name, state_name, aland/1000000 as area_km2 FROM counties ORDER BY area_km2 DESC LIMIT 5"
        elif 'smallest' in question_lower:
            # County-level query: smallest counties
            return "SELECT name, state_name, aland/1000000 as area_km2 FROM counties WHERE aland > 0 ORDER BY area_km2 ASC LIMIT 5"
        else:
            return "SELECT name, state_name FROM counties LIMIT 10"
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return f"SELECT COUNT(*) FROM counties LIMIT 1;"

def synthesize_answer(question: str, sql_query: str, results: str) -> str:
    """
    Synthesize a natural language answer based on the question, SQL, and results.
    Uses only the display data (no geometry) to ensure consistency with map.
    """
    # Get metadata from session state
    metadata = st.session_state.get("query_metadata", {})
    row_count = metadata.get("row_count", 0)
    sample_data = metadata.get("sample_data", [])
    
    # Build a clean summary of the data
    data_summary = f"Total rows: {row_count}\n"
    if sample_data:
        data_summary += f"Sample data (first {len(sample_data)} rows):\n"
        for i, row in enumerate(sample_data, 1):
            row_str = ", ".join([f"{k}: {v}" for k, v in row.items()])
            data_summary += f"{i}. {row_str}\n"
    
    prompt = f"""
    You are a helpful data assistant for a US Counties map application.
    
    User Question: {question}
    Executed SQL: {sql_query}
    Data Summary:
    {data_summary}
    
    Please provide a clear, concise natural language answer to the user's question based on the data summary.
    - If there are multiple rows, mention the count and list a few examples by name.
    - If it's a single number or aggregate, state it clearly.
    - If no data was found, explain that.
    - Do NOT mention technical details like "MultiPolygon" or coordinates.
    - Keep your answer brief and focused on what the user asked.
    """
    
    try:
        # Try Ollama via HTTP API
        ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get('response', '').strip()
            
    except Exception as e:
        print(f"Synthesis error: {e}")
        
    # Fallback if synthesis fails
    return f"Here are the results:\n{results}"

@tool
def execute_sql_query(sql_query: str, user_question: str = "") -> str:
    """
    Execute a SQL query against the US counties database and return the results.
    
    Args:
        sql_query: The SQL query to execute
        
    Returns:
        A formatted string with the query results
    """
    try:
        # Validate SQL - check for dangerous patterns
        sql_upper = sql_query.upper()
        
        # Reject queries that try to create geometry
        if 'ST_GEOMFROMTEXT' in sql_upper or 'ST_MAKEPOINT' in sql_upper:
            return f"Error: Cannot create new geometry objects. Please use the existing geom column instead.\n\nGenerated SQL:\n{sql_query}"
        
        # Check for unterminated strings
        if sql_query.count("'") % 2 != 0:
            return f"Error: Invalid SQL with unterminated string. Please try rephrasing your question.\n\nGenerated SQL:\n{sql_query}"
        
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        cur.execute(sql_query)

        # Get column names
        columns = [desc[0] for desc in cur.description] if cur.description else []

        # Fetch results
        rows = cur.fetchall()

        if not rows:
            st.session_state["map_features"] = []
            cur.close()
            conn.close()
            return "No results found."

        # Format results
        # Check if this is an aggregation result (single row, single column)
        if len(rows) == 1 and len(columns) == 1:
            col_name = columns[0].lower()
            value = rows[0][0]
            
            # Format aggregation results nicely
            if 'total' in col_name or 'sum' in col_name:
                if isinstance(value, (int, float)):
                    if 'area' in col_name:
                        return f"**Total Area:** {value:,.2f} km²"
                    else:
                        return f"**Total:** {value:,.2f}"
            elif 'count' in col_name:
                return f"**Count:** {value:,}"
            elif 'avg' in col_name or 'average' in col_name:
                if isinstance(value, (int, float)):
                    return f"**Average:** {value:,.2f}"
            elif 'max' in col_name or 'maximum' in col_name:
                if isinstance(value, (int, float)):
                    return f"**Maximum:** {value:,.2f}"
            elif 'min' in col_name or 'minimum' in col_name:
                if isinstance(value, (int, float)):
                    return f"**Minimum:** {value:,.2f}"
            
            # Fallback: If we have a result but no map features, try to fetch them for the state
            if not st.session_state.get("map_features"):
                 state = extract_state_from_question(user_question)
                 if state:
                     try:
                         map_query = f"SELECT name, state_name, ST_AsGeoJSON(geom) as geom_geojson, ST_Y(ST_Centroid(geom)) as centroid_lat, ST_X(ST_Centroid(geom)) as centroid_lon FROM counties WHERE state_name = '{state}'"
                         cur.execute(map_query)
                         map_rows = cur.fetchall()
                         map_cols = [desc[0] for desc in cur.description]
                         st.session_state["map_features"] = build_map_features(map_rows, map_cols)
                     except Exception as e:
                         print(f"Fallback map fetch failed: {e}")

            # Default single value format
            return f"**Result:** {value}"

        result_text = f"**Found {len(rows)} result(s):**\n\n"

        # Python-side Aggregation Logic
        question_lower = user_question.lower()
        if len(rows) > 1:
            if "count" in question_lower or "how many" in question_lower:
                result_text = f"**Count:** {len(rows)}\n\n" + result_text
            elif "area" in question_lower and ("total" in question_lower or "sum" in question_lower):
                # Try to find an area column
                area_col_idx = -1
                for idx, col in enumerate(columns):
                    if "area" in col.lower() or "aland" in col.lower():
                        area_col_idx = idx
                        break
                
                if area_col_idx != -1:
                    try:
                        total_area = sum(float(row[area_col_idx]) for row in rows if row[area_col_idx] is not None)
                        # Convert sq meters to sq km if needed (aland is usually sq meters)
                        # Assuming aland is sq meters, divide by 1,000,000 for sq km
                        if "aland" in columns[area_col_idx].lower():
                             total_area_km = total_area / 1_000_000
                             result_text = f"**Total Area:** {total_area_km:,.2f} km²\n\n" + result_text
                        else:
                             result_text = f"**Total Area:** {total_area:,.2f}\n\n" + result_text
                    except Exception as e:
                        print(f"Error calculating area: {e}")

        # Columns to exclude from text output (but keep for map)
        exclude_cols = ['geom', 'geometry', 'geom_geojson', 'centroid_lat', 'centroid_lon', 'st_asgeojson', 'st_y', 'st_x']
        
        # Filter columns for display
        display_indices = [i for i, col in enumerate(columns) if col.lower() not in exclude_cols]
        display_columns = [columns[i] for i in display_indices]

        # Build clean sample data for synthesis (without geometry)
        sample_rows = []
        for row in rows[:10]:
            clean_row = {display_columns[i]: row[display_indices[i]] for i in range(len(display_columns))}
            sample_rows.append(clean_row)

        for i, row in enumerate(rows[:10], 1):  
            if len(display_columns) == 0:
                 # If all columns are hidden (e.g. just asked for map), show name if available or generic message
                 result_text += f"{i}. [Map Data]\n"
            elif len(display_columns) == 1:
                result_text += f"{i}. {row[display_indices[0]]}\n"
            else:
                row_data = ", ".join([f"{col}: {row[idx]}" for idx, col in zip(display_indices, display_columns)])
                result_text += f"{i}. {row_data}\n"

        if len(rows) > 10:
            result_text += f"\n*...and {len(rows) - 10} more*"

        try:
            map_features = build_map_features(conn, rows, columns)
            st.session_state["map_features"] = map_features
        except Exception:
            st.session_state["map_features"] = []

        # Store metadata for synthesis
        st.session_state["query_metadata"] = {
            "row_count": len(rows),
            "display_columns": display_columns,
            "sample_data": sample_rows
        }

        cur.close()
        conn.close()

        return result_text

    except Exception as e:
        st.session_state["map_features"] = []
        return f"Database connection error: {e}. Please check if the database is properly configured and the counties table exists."

@tool
def get_database_schema() -> str:
    """
    Get information about the database schema for US counties.
    
    Returns:
        A string describing the database schema
    """
    return SCHEMA_INFO

# =============================================================================
# LANGGRAPH AGENT SETUP
# =============================================================================

def create_agent():
    """Create a simple agent that uses tools directly"""
    
    def simple_agent(question: str) -> str:
        """Simple agent that converts NL to SQL and executes it"""
        try:
            # Step 1: Convert natural language to SQL
            sql_query = convert_nl_to_sql.invoke({"question": question})
            
            # Step 2: Execute the SQL query
            results = execute_sql_query.invoke({"sql_query": sql_query, "user_question": question})
            
            # Step 3: Format the response (raw database results)
            response = f"""I'll help you answer that question about US counties.

**Generated SQL Query:**
```sql
{sql_query}
```

**Results:**
{results}

Is there anything else you'd like to know about US counties?"""
            
            return response
            
        except Exception as e:
            return f"I encountered an error while processing your question: {e}. Please try rephrasing your question."
    
    return simple_agent

# Initialize the agent
@st.cache_resource
def get_agent():
    """Get or create the agent (cached for performance)"""
    return create_agent()

# =============================================================================
# AGENT EXECUTION FUNCTIONS
# =============================================================================

def run_agent(question: str) -> str:
    """Run the agent with a user question and return the response"""
    try:
        # Clear previous map features
        st.session_state["map_features"] = []
        
        agent = get_agent()
        return agent(question)
    except Exception as e:
        return f"Error running agent: {e}"

# Streamlit App Configuration
st.set_page_config(
    page_title="Chat with the Map",
    page_icon="🗺️",
    layout="wide"
)

# App Title
st.title("🗺️ Chat with the Map")
st.markdown("Ask questions about US Counties in natural language! Powered by LangGraph ReAct Agent with Llama 3.2-3B")

# Sidebar with example questions
with st.sidebar:
    st.header("💡 Example Questions")
    example_questions = [
        "How many counties are in California?",
        "What are the 5 largest counties by area?",
        "Which counties border Los Angeles County?",
        "What is the total area of Texas?",
        "List all counties in New York",
        "Show me counties with area greater than 10,000 km²"
    ]
    
    for question in example_questions:
        if st.button(question, key=f"example_{question}"):
            st.session_state.user_input = question

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "map_features" not in st.session_state:
    st.session_state.map_features = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about US counties..."):

    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking and using tools..."):
            try:
                # Run the agent
                response = run_agent(prompt)
                
                # Display the response
                st.markdown(response)
                
                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Handle example question clicks
if hasattr(st.session_state, 'user_input'):
    prompt = st.session_state.user_input
    del st.session_state.user_input
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using agent
    with st.chat_message("assistant"):
        with st.spinner("🤖 Agent is thinking and using tools..."):
            try:
                # Run the agent
                response = run_agent(prompt)
                
                # Display the response
                st.markdown(response)
                
                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Map visualization
if st.session_state.get("map_features"):
    st.subheader("🗺️ Map View")
    render_map(st.session_state["map_features"])
