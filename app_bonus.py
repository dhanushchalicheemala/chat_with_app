# -*- coding: utf-8 -*-

#Chat with the Map - Streamlit Application with LangGraph ReAct Agent
#Natural Language Interface for US Counties PostGIS Database

import streamlit as st
import subprocess
import psycopg2
import json
import requests
import os
import re
import time
import pydeck as pdk
from groq import Groq
from typing import Tuple, List, Optional, Dict, Any
from langchain_core.tools import tool
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Streamlit page configuration (must be first Streamlit call)
st.set_page_config(page_title="Chat with the Map", page_icon="🗺️", layout="wide")

#Database configuration - Railway will provide these as environment variables

DB_PARAMS = {
    'dbname': os.getenv('PGDATABASE', 'USCountyDB'),
    'user': os.getenv('PGUSER', 'anishkale'),
    'password': os.getenv('PGPASSWORD', ''),
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', 5432))
}

#Ollama configuration - Railway service URL

OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
MODEL_NAME = "llama3.2:3b"
MAPBOX_TOKEN = os.getenv('MAPBOX_API_KEY') or os.getenv('MAPBOX_TOKEN')

# Schema information for the LLM
SCHEMA_INFO = """
Database: USCountyDB

SCOPE OF DATA
- This database contains ONLY **United States counties** (including county-equivalent units).
- There is exactly ONE table: `counties`.
- There is NO table for states, cities, GDP, population, demographics, or anything else.
- If a question asks about any country other than the United States (e.g. Canada, provinces, Mexico),
  you MUST answer by returning a single-row query of the form:
    SELECT 'Error: This application only supports United States counties.' AS error_message;

TABLES
- Table: counties  (THIS IS THE ONLY TABLE)

Schema (columns):
- id         : SERIAL PRIMARY KEY
- geom       : GEOMETRY(MULTIPOLYGON, 4269)  -- county boundary geometry (NAD83)
- statefp    : VARCHAR(2)    -- State FIPS code
- countyfp   : VARCHAR(3)    -- County FIPS code
- countyns   : VARCHAR(8)    -- County ANSI code
- geoid      : VARCHAR(5)    -- Geographic identifier (statefp + countyfp)
- name       : VARCHAR(100)  -- County name (e.g. 'Riverside', NOT 'Riverside County')
- namelsad   : VARCHAR(100)  -- Name + legal/statistical description (e.g. 'Riverside County')
- lsad       : VARCHAR(2)    -- Legal/statistical area description code
- classfp    : VARCHAR(2)    -- Class FIPS code
- mtfcc      : VARCHAR(5)    -- MAF/TIGER feature class code
- csafp      : VARCHAR(3)    -- Combined statistical area code
- cbsafp     : VARCHAR(5)    -- Metropolitan statistical area code
- metdivfp   : VARCHAR(5)    -- Metropolitan division code
- funcstat   : VARCHAR(1)    -- Functional status
- aland      : BIGINT        -- Land area in square meters
- awater     : BIGINT        -- Water area in square meters
- intptlat   : NUMERIC       -- Latitude of interior point
- intptlon   : NUMERIC       -- Longitude of interior point
- state_name : VARCHAR(100)  -- Full state name (e.g. 'California')
- state_abbr : VARCHAR(2)    -- State abbreviation (e.g. 'CA')

SPATIAL FUNCTIONS AVAILABLE
- ST_Area(geom::geography)          -- area in square meters
- ST_Perimeter(geom::geography)     -- perimeter in meters
- ST_Distance(geom1::geography, geom2::geography) -- distance in meters
- ST_Touches(geom1, geom2)          -- true if geometries share a boundary
- ST_Contains(geom1, geom2)         -- true if geom1 fully contains geom2
- ST_Within(geom1, geom2)           -- true if geom1 is within geom2
- ST_Intersects(geom1, geom2)       -- true if geometries intersect
- ST_Centroid(geom)                 -- centroid point of a geometry
- ST_NumInteriorRings(geom)         -- number of interior rings (holes)
- ST_NumGeometries(geom)            -- number of parts in a (Multi)Polygon
- ST_DumpRings(geom)                -- dump outer + inner rings
- ST_Union(geom)                    -- union of multiple geometries

UNIT CONVENTIONS
- Area in square kilometers: ST_Area(geom::geography) / 1000000.0 AS area_km2
- Area in square miles     : ST_Area(geom::geography) / 2589988.110336 AS area_mi2
- Perimeter in kilometers  : ST_Perimeter(geom::geography) / 1000.0   AS perimeter_km
- Perimeter in miles       : ST_Perimeter(geom::geography) / 1609.34  AS perimeter_mi

ABSOLUTE RULES ABOUT TABLES
- There is ONLY the `counties` table.
- NEVER reference or JOIN to tables like `states`, `USStateDB`, `state_boundaries`, `cities`, etc.
- All state information (name, abbreviation) is already in `state_name` and `state_abbr` columns.
- For queries like "counties in [state]" use only:
    WHERE state_name = 'Full State Name'
  or:
    WHERE state_abbr = 'XX'

ABSOLUTE RULES ABOUT GEOMETRY CREATION
- DO NOT create new geometries using ST_GeomFromText, ST_MakePoint, ST_MakeEnvelope, etc.
- DO NOT hand-write polygons in WKT.
- ALWAYS use the existing `geom` column in the `counties` table for spatial operations.

ATTRIBUTES THAT DO **NOT** EXIST (GDP, population, etc.)
- The dataset DOES NOT contain:
  GDP, population, income, median_household_income, race, age, unemployment, education, etc.
- If a question asks for any attribute that is not one of the listed columns above (for example GDP
  or population), you MUST NOT invent a column.
- Instead, answer with a single-row SQL query of the form:
    SELECT 'Parameter not present in the dataset' AS error_message;

AGGREGATION RULES
- "total", "sum", "combined", "all together"  → use SUM(...)
- "average", "mean"                          → use AVG(...)
- "how many", "number of", "count of"       → use COUNT(*)
- "largest", "biggest", "maximum"           → use MAX(...)
- "smallest", "least", "minimum"            → use MIN(...)

- For "total area" questions, you MUST use SUM(ST_Area(geom::geography)/1000000.0)
  (or SUM of the miles² expression) rather than just ST_Area on a single row.

AGGREGATION + GEOMETRY SELECTION RULES
- When the user asks for a **count** ("how many ...", "number of counties"), generate:

    SELECT COUNT(*) AS count
    FROM counties
    WHERE <conditions>;

  and DO NOT include any geometry columns (NO geom, NO ST_AsGeoJSON, NO centroids) in this query.

- For other *purely numeric* aggregations (total area, average area, min/max area/perimeter),
  return only the numeric aggregations unless the user explicitly asks to “show on a map” or
  “visualize the counties”.

- Only include geometry in the SELECT list (e.g. ST_AsGeoJSON(geom) AS geom_geojson or
  centroid coordinates) if the user clearly asks to see a map, visualize, display, or list
  counties with geometry.

NAME NORMALIZATION (IMPORTANT FOR THE 24 QUESTIONS)
- In the data, `name` does NOT include suffixes like "County", "Parish", "Borough", etc.
  Examples:
    - Stored as name = 'Riverside', not 'Riverside County'
    - Stored as name = 'Orange', not 'Orange County'
- When the NL question says "Riverside County, California" you must filter using:
    WHERE name = 'Riverside' AND state_name = 'California'
  or:
    WHERE name = 'Riverside' AND state_abbr = 'CA'

STATE FILTERING PATTERNS
- Full state name:     WHERE state_name = 'California'
- State abbreviation:  WHERE state_abbr = 'CA'
- You may use table aliases:
    c.state_name, c.state_abbr
  when joining `counties` to itself.

TYPICAL QUERY PATTERNS FOR THE 24 ASSIGNMENT QUESTIONS

1. Show me [Madison County, Idaho]
   → Single county geometry & centroid:
   SELECT name, state_name, state_abbr, geoid,
          ST_AsGeoJSON(geom) AS geom_geojson,
          ST_Y(ST_Centroid(geom)) AS centroid_lat,
          ST_X(ST_Centroid(geom)) AS centroid_lon
   FROM counties
   WHERE name = 'Madison' AND state_name = 'Idaho';

2. Visualize me [Madison County] in all states
   → All counties with the same name, across states:
   SELECT name, state_name, state_abbr, geoid,
          ST_AsGeoJSON(geom) AS geom_geojson,
          ST_Y(ST_Centroid(geom)) AS centroid_lat,
          ST_X(ST_Centroid(geom)) AS centroid_lon
   FROM counties
   WHERE name = 'Madison'
   ORDER BY state_name;

3. How many counties are called [Madison County]?
   → Count of counties with that name (no geometry):
   SELECT COUNT(*) AS county_count
   FROM counties
   WHERE name = 'Madison';

4. [three] most frequent county names in USA
   → Top-k county names with counts:
   SELECT name, COUNT(*) AS occurrences
   FROM counties
   GROUP BY name
   ORDER BY occurrences DESC, name ASC
   LIMIT 3;

5. Out-of-scope geography (Canada, provinces, etc.)
   → MUST return:
   SELECT 'Error: This application only supports United States counties.' AS error_message;

6–8. List counties in a state / starting with prefix
   - All counties in Florida:
       SELECT name, state_name, state_abbr, geoid,
              ST_AsGeoJSON(geom) AS geom_geojson,
              ST_Y(ST_Centroid(geom)) AS centroid_lat,
              ST_X(ST_Centroid(geom)) AS centroid_lon
       FROM counties
       WHERE state_name = 'Florida'
       ORDER BY name;

   - Counties in WA:
       WHERE state_abbr = 'WA'

   - Counties starting with 'San ' in California:
       WHERE state_name = 'California' AND name ILIKE 'San %'

9. Counties whose name equals their state:
   WHERE LOWER(name) = LOWER(state_name)

10. Multi-word names in [Minnesota]:
   WHERE state_name = 'Minnesota' AND name LIKE '% %'

11. Counties in [California] that touch [Nevada]:
   Join counties to itself with ST_Touches, using state filters on c1 (CA) and c2 (NV).

12. Land area of [Riverside County, California]:
   SELECT ST_Area(geom::geography)/1000000.0 AS area_km2,
          ST_Area(geom::geography)/2589988.110336 AS area_mi2
   FROM counties
   WHERE name = 'Riverside' AND state_name = 'California';

13. Rank all counties in [Arizona] by area:
   WHERE state_name = 'Arizona'
   ORDER BY area_km2 DESC

14. Smallest area county in [NC]:
   WHERE state_abbr = 'NC'
   ORDER BY area_km2 ASC
   LIMIT 1

15. Counties nationwide with [area < 100 mi²]:
   WHERE (ST_Area(geom::geography)/2589988.110336) < 100.0

16. Perimeter length of [Orange County, California]:
   SELECT ST_Perimeter(geom::geography)/1000.0 AS perimeter_km,
          ST_Perimeter(geom::geography)/1609.34 AS perimeter_mi
   FROM counties
   WHERE name = 'Orange' AND state_name = 'California';

17. Counties in [South Dakota] with [perimeter > 800 mi]:
   WHERE state_name = 'South Dakota'
     AND (ST_Perimeter(geom::geography)/1609.34) > 800.0

18. Counties with holes (interior rings):
   WHERE ST_NumInteriorRings(geom) > 0

19. Multipart / non-contiguous counties:
   WHERE ST_NumGeometries(geom) > 1

20. Counties whose centroid falls outside the polygon:
   WHERE NOT ST_Contains(geom, ST_Centroid(geom))

21. Largest interior hole of [Ramsey County, MN]:
   Use a WITH query with ST_DumpRings to compute the largest interior ring area.

22. Number of neighbors of [Utah County, UT]:
   Join `counties` AS c1 and c2 with ST_Touches and COUNT DISTINCT neighbors of c1.

23. County in [AL] with the most neighbors:
   Use a CTE computing neighbor counts with ST_Touches within Alabama, then select the max.

24. Counties in [CA] with exactly two neighbors:
   Similar neighbor-count CTE filtered to California, then WHERE neighbor_count = 2.

REMINDERS
- Always respect the SINGLE TABLE (`counties`) constraint.
- Never invent columns: if an attribute is not listed above, return the
  'Parameter not present in the dataset' error_message query.
- Prefer explicit, simple SQL that the database can run directly.
"""


#=============================================================================
#TEXT & STATE UTILITIES
#=============================================================================

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
    Works both on short bracket fragments and on full questions.
    """
    if not value:
        return None

    cleaned = value.lower()

    # Normalise unit phrases
    cleaned = cleaned.replace("sq.", "square ").replace("sq ", "square ")
    cleaned = cleaned.replace("²", "2").replace("^2", "2")
    cleaned = cleaned.replace("square miles", "mi2").replace("square mile", "mi2")
    cleaned = cleaned.replace("square kilometers", "km2").replace("square kilometer", "km2")
    cleaned = cleaned.replace("square kilometres", "km2").replace("square kilometre", "km2")

    # Normalise verbal comparison operators to symbols
    cleaned = cleaned.replace("greater than or equal to", ">=")
    cleaned = cleaned.replace("greater than or equal", ">=")
    cleaned = cleaned.replace("greater than", ">")
    cleaned = cleaned.replace("more than", ">")
    cleaned = cleaned.replace("less than or equal to", "<=")
    cleaned = cleaned.replace("less than or equal", "<=")
    cleaned = cleaned.replace("less than", "<")
    cleaned = cleaned.replace("fewer than", "<")

    # Remove thousands separators in numbers (e.g., 10,000 -> 10000)
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)

    # Look for "area > 10000 km2" or "perimeter < 800 mi", etc.
    match = re.search(
        r"(area|perimeter)\s*(<=|>=|<|>)\s*([0-9]+(?:\.[0-9]+)?)\s*([a-z0-9\s/]*)",
        cleaned,
    )
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

#=============================================================================
#LANGCHAIN TOOLS
#=============================================================================

def fix_sql_aggregation(sql: str, question: str) -> str:
    """
    Post-process SQL to fix missing aggregations for "total", "sum", etc.
    """
    sql_upper = sql.upper()
    question_lower = question.lower()

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
def fix_county_suffix(sql: str) -> str:
    """
    Removes county-level suffixes (County, Parish, Borough, etc.)
    from name='...' conditions so SQL matches real TIGER data.
    """
    def clean_name(match):
        full = match.group(1)
        normalized = normalize_county_name(full)
        return f"name = '{normalized}'"

    # regex to capture name='Something'
    return re.sub(
        r"name\s*=\s*'([^']+)'",
        lambda m: clean_name(m),
        sql,
        flags=re.IGNORECASE
    )
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

def build_single_feature_from_sql(conn, sql_query: str) -> List[Dict[str, Any]]:
    """
    Fallback: if the query returns only aggregated values, try to detect a
    referenced county or state from the SQL and fetch its geometry.

    Now supports:
    - state_name IN ('Texas','California',...)
    - state_abbr IN ('TX','CA',...)
    returning ONE feature per state (union of that state's counties).
    """
    try:
        sql_upper = sql_query.upper()

        # ------------------------------------------------------
        # 1) MULTI-STATE PATTERNS (state_name/state_abbr IN (...))
        # ------------------------------------------------------
        features: List[Dict[str, Any]] = []

        # Pattern: state_name IN ('Texas','California',...)
        m = re.search(r"state_name\s+IN\s*\(([^)]+)\)", sql_query, re.IGNORECASE)
        if m:
            raw = m.group(1)
            tokens = raw.split(",")
            for token in tokens:
                token_clean = token.strip().strip("'\"")
                if not token_clean:
                    continue

                # Normalise via STATE_NORMALIZER so things like "tx" or "TX"
                # are mapped back to "Texas" if needed.
                full_name = STATE_NORMALIZER.get(token_clean.lower()) or token_clean
                full_abbr = STATE_NAME_TO_ABBR.get(full_name)

                geom_info = fetch_geometry_for_feature(
                    conn,
                    name=None,
                    state_name=full_name,
                    state_abbr=full_abbr,
                    geoid=None,
                )
                if geom_info:
                    features.append(
                        {
                            "name": geom_info.get("name")
                            or geom_info.get("state_name")
                            or "Location",
                            "state_name": geom_info.get("state_name"),
                            "state_abbr": full_abbr,
                            "geom_geojson": geom_info.get("geom_geojson"),
                            "centroid_lat": geom_info.get("centroid_lat"),
                            "centroid_lon": geom_info.get("centroid_lon"),
                        }
                    )

            if features:
                return features

        # Pattern: state_abbr IN ('TX','CA',...)
        m = re.search(r"state_abbr\s+IN\s*\(([^)]+)\)", sql_query, re.IGNORECASE)
        if m:
            raw = m.group(1)
            tokens = raw.split(",")
            for token in tokens:
                ab = token.strip().strip("'\"").upper()
                if not ab:
                    continue

                full_name = STATE_ABBR_TO_NAME.get(ab)

                geom_info = fetch_geometry_for_feature(
                    conn,
                    name=None,
                    state_name=full_name,
                    state_abbr=ab,
                    geoid=None,
                )
                if geom_info:
                    features.append(
                        {
                            "name": geom_info.get("name")
                            or geom_info.get("state_name")
                            or "Location",
                            "state_name": full_name,
                            "state_abbr": ab,
                            "geom_geojson": geom_info.get("geom_geojson"),
                            "centroid_lat": geom_info.get("centroid_lat"),
                            "centroid_lon": geom_info.get("centroid_lon"),
                        }
                    )

            if features:
                return features

        # ------------------------------------------------------
        # 2) EXISTING SINGLE-STATE / COUNTY LOGIC (unchanged)
        # ------------------------------------------------------

        # Try to extract equality predicates like:
        # name='Riverside', state_name='California', state_abbr='CA', geoid='06065'
        # and also LOWER(state_name) = LOWER('Texas'), with or without table aliases.
        def extract_eq(field: str) -> Optional[str]:
            # Allow an optional table alias, e.g. c.state_name
            field_pattern = rf"(?:[A-Za-z_][A-Za-z0-9_]*\.)?{field}"

            # 1) Plain equality:   state_name = 'Texas'
            m_local = re.search(
                rf"{field_pattern}\s*=\s*'([^']+)'",
                sql_query,
                flags=re.IGNORECASE,
            )
            if m_local:
                return m_local.group(1).strip()

            # 2) Case-insensitive form:   LOWER(state_name) = LOWER('Texas')
            m_local = re.search(
                rf"LOWER\(\s*{field_pattern}\s*\)\s*=\s*LOWER\('([^']+)'\)",
                sql_query,
                flags=re.IGNORECASE,
            )
            if m_local:
                return m_local.group(1).strip()

            return None

        name = extract_eq("name")
        state_name = extract_eq("state_name")
        state_abbr = extract_eq("state_abbr")
        geoid = extract_eq("geoid")

        # If both name and state_name happen to be the same string
        # (e.g., 'Texas'), treat it as a state-level query, not a county.
        if name and state_name and name.lower() == state_name.lower():
            name = None

        # If nothing useful found, give up
        if not any([name, state_name, state_abbr, geoid]):
            return []

        geom_info = fetch_geometry_for_feature(
            conn,
            name=name,
            state_name=state_name,
            state_abbr=state_abbr,
            geoid=geoid,
        )
        if not geom_info:
            return []

        feature = {
            "name": geom_info.get("name") or geom_info.get("state_name") or "Location",
            "state_name": geom_info.get("state_name"),
            "state_abbr": state_abbr,
            "geom_geojson": geom_info.get("geom_geojson"),
            "centroid_lat": geom_info.get("centroid_lat"),
            "centroid_lon": geom_info.get("centroid_lon"),
        }

        if (
            feature["geom_geojson"]
            and feature["centroid_lat"] is not None
            and feature["centroid_lon"] is not None
        ):
            return [feature]

        return []
    except Exception:
        return []

def build_map_features(conn, rows: List[tuple], columns: List[str]) -> List[Dict[str, Any]]:
    """Build map features (polygon + centroid) from query results."""
    map_features: List[Dict[str, Any]] = []
    if not rows or not columns:
        return map_features
    max_features = min(len(rows), 500)
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
    """
    Render map with polygons and centroid markers using PyDeck.

    ALWAYS shows a map:
    - If county features exist → highlight them
    - If features missing → USA fallback polygon
    - NEVER silently return
    """
    try:
        # Fallback if nothing is provided
        if not features:
            features = [{
                "name": "United States",
                "state_name": "",
                "state_abbr": "",
                "geom_geojson": json.dumps({
                    "type": "Polygon",
                    "coordinates": [[
                        [-125, 24], [-66, 24], [-66, 49], [-125, 49], [-125, 24]
                    ]]
                }),
                "centroid_lat": 39.5,
                "centroid_lon": -98.35,
            }]

        polygon_data = []
        centroid_data = []
        avg_lat = 37.0902
        avg_lon = -95.7129

        # Build polygon and centroid lists
        for feat in features:
            try:
                geometry = json.loads(feat.get("geom_geojson"))
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

        # If no valid polygons → fallback USA
        if not polygon_data:
            polygon_data = [{
                "name": "United States",
                "state_name": "",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-125, 24], [-66, 24], [-66, 49], [-125, 49], [-125, 24]
                    ]]
                }
            }]
            centroid_data = [{
                "name": "United States",
                "state_name": "",
                "lat": 39.5,
                "lon": -98.35,
            }]
            avg_lat = 39.5
            avg_lon = -98.35

        else:
            # Center map on centroid(s)
            valid_lats = [pt["lat"] for pt in centroid_data if pt.get("lat")]
            valid_lons = [pt["lon"] for pt in centroid_data if pt.get("lon")]
            if valid_lats and valid_lons:
                avg_lat = sum(valid_lats) / len(valid_lats)
                avg_lon = sum(valid_lons) / len(valid_lons)

        # Select basemap
        if MAPBOX_TOKEN:
            pdk.settings.mapbox_api_key = MAPBOX_TOKEN
            map_style = "mapbox://styles/mapbox/light-v9"
        else:
            map_style = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

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

        deck = pdk.Deck(
            layers=[polygon_layer, centroid_layer],
            initial_view_state=pdk.ViewState(
                latitude=avg_lat,
                longitude=avg_lon,
                zoom=4,
                pitch=0,
            ),
            map_style=map_style,
        )

        st.pydeck_chart(deck)

    except Exception as e:
        st.warning(f"Map rendering error: {e}")

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

    # 1) States that appear inside brackets, e.g. [California], [AZ]
    for parsed in parsed_brackets:
        state_name = parsed.get("state_name")
        state_abbr = parsed.get("state_abbr")
        if state_name:
            state_mentions.append((state_name, state_abbr))

    # 2) If no bracketed states, scan the whole question for ALL states
    if not state_mentions:
        question_clean = re.sub(r"[^a-z0-9\s]", " ", question.lower())

        # Track which states we already added
        seen: set[str] = set()
        found_full_name = False

        # 2a) Full state names (California, Arizona, …)
        for state_name, abbr in STATE_NAME_TO_ABBR.items():
            pattern = r"\b" + re.escape(state_name.lower()) + r"\b"
            if re.search(pattern, question_clean):
                if state_name not in seen:
                    state_mentions.append((state_name, abbr))
                    seen.add(state_name)
                    found_full_name = True

        # 2b) Two-letter abbreviations (CA, AZ, TX, …)
        # Only if we didn't already find any full state names
        if not found_full_name:
            # Ambiguous abbreviations that collide with English words
            ambiguous_abbrs = {"IN", "OR", "ME", "HI"}

            tokens = re.findall(r"\b[a-z]{2}\b", question_clean)
            for tok in tokens:
                abbr = tok.upper()
                if abbr in ambiguous_abbrs:
                    continue
                full = STATE_ABBR_TO_NAME.get(abbr)
                if full and full not in seen:
                    state_mentions.append((full, STATE_NAME_TO_ABBR[full]))
                    seen.add(full)
    # Convenience aliases
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

        # If no [area ...] / [perimeter ...] condition was found in brackets,
    # also try to parse a condition written directly in the question text,
    # e.g. "area greater than 10,000 km2".
    if area_condition is None and perimeter_condition is None:
        parsed_q_cond = parse_measurement_condition(question)
        if parsed_q_cond:
            if parsed_q_cond["metric"] == "area":
                area_condition = parsed_q_cond
            elif parsed_q_cond["metric"] == "perimeter":
                perimeter_condition = parsed_q_cond

    # Out-of-scope guard
    if any(keyword in q_lower for keyword in ["canada", "mexico", "province", "provinces"]):
        message = "Error: This application only supports United States counties."
        return f"SELECT {sql_literal(message)} AS error_message"

    # Most frequent county names nationwide
        # Most frequent / common county names nationwide
    if (
        "frequent county name" in q_lower
        or "frequent county names" in q_lower
        or "most frequent county names" in q_lower
        or "common county name" in q_lower
        or "common county names" in q_lower
        or "most common county names" in q_lower
    ):
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

    # County-level adjacency
    if "counties" in q_lower and any(k in q_lower for k in ["border", "adjacent", "touch"]):
        # Prefer bracket parsing (e.g. [Los Angeles County, California])
        target_county = county_name
        target_state_name = county_state_name or first_state_name
        target_state_abbr = county_state_abbr or first_state_abbr

        # If no county from brackets, try to pull "<X> County" from the free text
        if not target_county:
            m = re.search(r"border\s+([a-z\s]+?)\s+county", q_lower)
            if m:
                # normalize_county_name removes the trailing "County"
                target_county = normalize_county_name(m.group(1))

        if target_county:
            county_clause = build_county_alias_clause(
                "c1",
                target_county,
                target_state_name,
                target_state_abbr,
            )

            if county_clause:
                # Return all neighboring counties (c2) that touch the target county (c1)
                return (
                    f"SELECT {geometry_columns(alias='c2')}\n"
                    "FROM counties c1\n"
                    "JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)\n"
                    f"WHERE {county_clause}\n"
                    "ORDER BY c2.state_name, c2.name"
                )

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

    # Counties with interior holes (handle MultiPolygons)
    if "interior ring" in q_lower or "holes" in q_lower:
        return (
            "WITH exploded AS (\n"
            "    SELECT geoid,\n"
            "           (ST_Dump(geom)).geom AS poly\n"
            "    FROM counties\n"
            "), counts AS (\n"
            "    SELECT geoid,\n"
            "           SUM(CASE WHEN ST_NumInteriorRings(poly) > 0 THEN 1 ELSE 0 END) AS num_with_holes\n"
            "    FROM exploded\n"
            "    GROUP BY geoid\n"
            ")\n"
            f"SELECT {geometry_columns(extra_columns=['counts.num_with_holes AS num_with_holes'], alias='c')}\n"
            "FROM counts\n"
            "JOIN counties c ON c.geoid = counts.geoid\n"
            "WHERE counts.num_with_holes > 0\n"
            "ORDER BY counts.num_with_holes DESC, c.state_name, c.name"
        )

    # Multipart counties (non-contiguous)
    multipart_keywords = [
        "multipart",
        "multi-part",
        "non-contiguous",
        "non contiguous",
        "noncontiguous",
        "multipolygon",
        "multi polygon",
        "multi polygons",
        "multi-polygon",
    ]
    if any(keyword in q_lower for keyword in multipart_keywords):
        extra_cols = ["ST_NumGeometries(geom) AS part_count"]

        conditions = ["ST_NumGeometries(geom) > 1"]

        # If one or more states are mentioned, restrict to those states
        where_states: Optional[str] = None
        if state_mentions:
            if len(state_mentions) == 1:
                # Single state: simple equality
                where_states = state_filter_clause(first_state_name, first_state_abbr)
            else:
                # Multiple states → build IN (...) clause on state_name or state_abbr
                state_names = [s for (s, a) in state_mentions if s]
                state_abbrs = [a for (s, a) in state_mentions if a]

                if state_names:
                    unique_names = sorted(set(state_names))
                    in_list = ", ".join(sql_literal(s) for s in unique_names)
                    where_states = f"state_name IN ({in_list})"
                elif state_abbrs:
                    unique_abbrs = sorted(set(state_abbrs))
                    in_list = ", ".join(sql_literal(a) for a in unique_abbrs)
                    where_states = f"state_abbr IN ({in_list})"

        if where_states:
            conditions.insert(0, where_states)

        where_clause = " AND ".join(conditions)

        return build_geometry_query(
            where_clause,
            "ORDER BY part_count DESC, state_name, name",
            extra_columns=extra_cols,
        )

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
                "    -- Dump MultiPolygon into individual polygons\n"
                f"    SELECT (ST_Dump(geom)).geom AS geom\n"
                "    FROM counties\n"
                f"    WHERE {county_clause}\n"
                "    LIMIT 1\n"
                "), rings AS (\n"
                "    -- Dump outer + inner rings for that polygon\n"
                "    SELECT (ST_DumpRings(geom)).geom AS ring_geom,\n"
                "           (ST_DumpRings(geom)).path AS ring_path\n"
                "    FROM target\n"
                "), interior AS (\n"
                "    -- Interior rings have path[1] > 0\n"
                "    SELECT ring_geom\n"
                "    FROM rings\n"
                "    WHERE ring_path[1] > 0\n"
                "), areas AS (\n"
                "    SELECT ST_Area(ring_geom::geography)/1000000.0 AS hole_area_km2\n"
                "    FROM interior\n"
                ")\n"
                "SELECT COALESCE(MAX(hole_area_km2), 0) AS largest_hole_area_km2\n"
                "FROM areas"
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

    # County with the most neighbors in a state (neighbors also within the same state)
    if "most neighbors" in q_lower and (first_state_name or first_state_abbr):
        state_clause_c1 = state_filter_with_alias(first_state_name, first_state_abbr, "c1")
        state_clause_c2 = state_filter_with_alias(first_state_name, first_state_abbr, "c2")
        if state_clause_c1 and state_clause_c2:
            return (
                "WITH neighbor_counts AS (\n"
                "    SELECT c1.geoid,\n"
                "           COUNT(DISTINCT c2.geoid) AS neighbor_count\n"
                "    FROM counties c1\n"
                "    JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)\n"
                f"    WHERE {state_clause_c1}\n"
                f"      AND {state_clause_c2}\n"
                "    GROUP BY c1.geoid\n"
                ")\n"
                f"SELECT {geometry_columns(extra_columns=['neighbor_counts.neighbor_count AS neighbor_count'], alias='c')}\n"
                "FROM neighbor_counts\n"
                "JOIN counties c ON c.geoid = neighbor_counts.geoid\n"
                "ORDER BY neighbor_counts.neighbor_count DESC, c.name\n"
                "LIMIT 1"
            )


    # Counties with an exact neighbor count in a state (neighbors also within that state)
    if "exactly two neighbors" in q_lower and (first_state_name or first_state_abbr):
        state_clause_c1 = state_filter_with_alias(first_state_name, first_state_abbr, "c1")
        state_clause_c2 = state_filter_with_alias(first_state_name, first_state_abbr, "c2")
        if state_clause_c1 and state_clause_c2:
            return (
                "WITH neighbor_counts AS (\n"
                "    SELECT c1.geoid,\n"
                "           COUNT(DISTINCT c2.geoid) AS neighbor_count\n"
                "    FROM counties c1\n"
                "    JOIN counties c2 ON ST_Touches(c1.geom, c2.geom)\n"
                f"    WHERE {state_clause_c1}\n"
                f"      AND {state_clause_c2}\n"
                "    GROUP BY c1.geoid\n"
                ")\n"
                f"SELECT {geometry_columns(extra_columns=['neighbor_counts.neighbor_count AS neighbor_count'], alias='c')}\n"
                "FROM neighbor_counts\n"
                "JOIN counties c ON c.geoid = neighbor_counts.geoid\n"
                "WHERE neighbor_count = 2\n"
                "ORDER BY c.name"
            )


        # "How many counties in [state]?" → return COUNT(*), not full geometry list
    if (
        ("how many counties" in q_lower or "number of counties" in q_lower)
        and state_mentions
    ):
        where_clause: Optional[str] = None

        if len(state_mentions) == 1:
            # Single state → simple equality
            where_clause = state_filter_clause(first_state_name, first_state_abbr)
        else:
            # Multiple states → build an IN (...) clause
            state_names = [s for (s, a) in state_mentions if s]
            state_abbrs = [a for (s, a) in state_mentions if a]

            if state_names:
                unique_names = sorted(set(state_names))
                in_list = ", ".join(sql_literal(s) for s in unique_names)
                where_clause = f"state_name IN ({in_list})"
            elif state_abbrs:
                unique_abbrs = sorted(set(state_abbrs))
                in_list = ", ".join(sql_literal(a) for a in unique_abbrs)
                where_clause = f"state_abbr IN ({in_list})"

        if where_clause:
            return (
                "SELECT COUNT(*) AS county_count\n"
                "FROM counties\n"
                f"WHERE {where_clause}"
            )

    # List all counties in one or more states (default geometry query)
    if (
        "counties" in q_lower
        and "in" in q_lower
        and state_mentions
        and not any(
            keyword in q_lower
            for keyword in [
                "start with",
                "touch",
                "border",
                "adjacent",
                "perimeter",
                "area",
                "neighbors",
                "multi-word",
                "multi word",
                "rank",
            ]
        )
    ):
        where_clause: Optional[str] = None

        if len(state_mentions) == 1:
            # Single state
            where_clause = state_filter_clause(first_state_name, first_state_abbr)
        else:
            # Multiple states → build an IN (...) clause
            state_names = [s for (s, a) in state_mentions if s]
            state_abbrs = [a for (s, a) in state_mentions if a]

            if state_names:
                unique_names = sorted(set(state_names))
                in_list = ", ".join(sql_literal(s) for s in unique_names)
                where_clause = f"state_name IN ({in_list})"
            elif state_abbrs:
                unique_abbrs = sorted(set(state_abbrs))
                in_list = ", ".join(sql_literal(a) for a in unique_abbrs)
                where_clause = f"state_abbr IN ({in_list})"

        if where_clause:
            # Order by state then county for multi-state clarity
            return build_geometry_query(where_clause, "ORDER BY state_name, name")
    return None

@tool
def convert_nl_to_sql(question: str) -> str:
    """
    Convert a natural language question about US counties to a PostgreSQL SQL query.

    Respects a Streamlit toggle:
        st.session_state["llm_backend"] in {"local", "remote"}

    And records a human-readable label in:
        st.session_state["last_backend_label"]
    """
    question_lower = question.lower()

    # Default assumption: rule-based templates, no LLM
    try:
        st.session_state["last_backend_label"] = "Rule-based templates (no LLM)"
    except Exception:
        # In case Streamlit is not initialized (rare)
        pass

    # ------------------------------------------------------------------
    # Hard out-of-scope guard for non-US countries
    # ------------------------------------------------------------------
    if any(term in question_lower for term in ["canada", "mexico", "province", "provinces"]):
        msg = "Error: This application only supports United States counties."
        return f"SELECT {sql_literal(msg)} AS error_message"

    # ------------------------------------------------------------------
    # Guard: attributes NOT present in the counties dataset
    # ------------------------------------------------------------------
    unsupported_terms = [
        "population",
        "gdp",
        "gross domestic product",
        "median income",
        "median household income",
        "household income",
        "unemployment",
        "crime rate",
        "crime rates",
        "poverty rate",
        "life expectancy",
    ]
    bad_terms = [t for t in unsupported_terms if t in question_lower]
    if bad_terms:
        msg = (
            "Error: The requested attribute(s) "
            + ", ".join(f"'{t}'" for t in bad_terms)
            + " are not present in the US counties dataset. "
            "This database only contains geometry and basic TIGER/Line attributes "
            "such as names, FIPS codes, and land/water area."
        )
        return f"SELECT {sql_literal(msg)} AS error_message"

    # ------------------------------------------------------------------
    # 1) Try deterministic patterns first (no LLM)
    # ------------------------------------------------------------------
    structured_sql = generate_structured_sql(question)
    if structured_sql:
        # We already set last_backend_label to rule-based at the top
        return structured_sql

    # ------------------------------------------------------------------
    # 2) Special pre-LLM patterns: state adjacency / counties in state
    # ------------------------------------------------------------------
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
            return (
                "SELECT DISTINCT c2.state_name, c2.state_abbr "
                "FROM counties c1, counties c2 "
                f"WHERE c1.state_name = '{state}' "
                "AND ST_Touches(c1.geom, c2.geom) "
                "AND c2.state_abbr != c1.state_abbr "
                "ORDER BY c2.state_name"
            )
        else:
            return (
                "SELECT DISTINCT c2.state_name, c2.state_abbr "
                "FROM counties c1, counties c2 "
                "WHERE ST_Touches(c1.geom, c2.geom) "
                "AND c2.state_abbr != c1.state_abbr "
                "ORDER BY c2.state_name LIMIT 0"
            )

    counties_in_state_patterns = [
        'counties in', 'counties of', 'counties within', 'counties that are in',
        'list counties in', 'show counties in', 'what counties are in'
    ]
    if any(pattern in question_lower for pattern in counties_in_state_patterns):
        state = extract_state_from_question(question)
        if state:
            return (
                "SELECT name, state_name "
                "FROM counties "
                f"WHERE state_name = '{state}' "
                "ORDER BY name"
            )

    # ------------------------------------------------------------------
    # 3) LLM-based SQL generation, controlled by backend toggle
    # ------------------------------------------------------------------
    prompt = f"""{SCHEMA_INFO}
    Question: {question}
    You are an expert SQL query generator for a PostGIS spatial database.
    Given a natural language question, generate ONLY the SQL query without any explanation:"""

    # Backend selection from sidebar
    backend_choice = None
    try:
        backend_choice = st.session_state.get("llm_backend", "local")
    except Exception:
        backend_choice = "local"

    # Helper to run Ollama (local)
    def _try_ollama(prompt_text: str) -> Optional[str]:
        try:
            ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt_text,
                    "stream": False,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                result = resp.json()
                sql_text = result.get('response', '').strip()
                sql_text = sql_text.replace('```sql', '').replace('```', '').strip()
                return sql_text or None
            else:
                st.warning(f"Ollama HTTP error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.warning(f"Ollama not available: {e}")
        return None
    
    # Helper to run Groq (remote)
    def _try_groq(prompt_text: str) -> Optional[str]:
        groq_key = os.getenv('GROQ_API_KEY')
        if not groq_key:
            st.warning("GROQ_API_KEY not set.")
            return None

        try:
            client = Groq(api_key=groq_key)

            chat_completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You generate SQL only."},
                    {"role": "user", "content": prompt_text},
                ],
            )

            sql_text = chat_completion.choices[0].message.content.strip()
            sql_text = sql_text.replace("```sql","").replace("```","").strip()
            return sql_text
        except Exception as e:
            st.warning(f"Groq error: {e}")
            return None

    sql_from_llm: Optional[str] = None

    # Decide which backend to actually call
    if backend_choice == "local":
        # Local Ollama only
        try:
            st.session_state["last_backend_label"] = f"Local LLM (Ollama: {MODEL_NAME})"
        except Exception:
            pass
        sql_from_llm = _try_ollama(prompt)

    elif backend_choice == "groq":
        # Remote Groq backend
        try:
            st.session_state["last_backend_label"] = "Remote LLM (Groq llama-3.3-70b-versatile)"
        except Exception:
            pass
        sql_from_llm = _try_groq(prompt)

    else:
        # Fallback: behave like local
        try:
            st.session_state["last_backend_label"] = f"Local LLM (Ollama: {MODEL_NAME})"
        except Exception:
            pass
        sql_from_llm = _try_ollama(prompt)


    # If LLM gave us something, post-process and return
    if sql_from_llm:
        sql = sql_from_llm
        # Fix aggregation issues, invalid tables, and county suffixes
        sql = fix_sql_aggregation(sql, question)
        sql = fix_invalid_table_references(sql, question)
        sql = fix_county_suffix(sql)
        return sql

    # ------------------------------------------------------------------
    # 4) Final fallbacks (no LLM available / failed) – rule-based
    # ------------------------------------------------------------------
    try:
        st.session_state["last_backend_label"] = "Rule-based fallback (no LLM - backend failed)"
    except Exception:
        pass

    if 'count' in question_lower or 'how many' in question_lower:
        state = extract_state_from_question(question)
        if state:
            return f"SELECT COUNT(*) FROM counties WHERE state_name = {sql_literal(state)}"
        else:
            return "SELECT COUNT(*) FROM counties"

    if 'total' in question_lower and 'area' in question_lower:
        # Support multiple states, e.g. "Texas and California"
        state_mentions: List[Tuple[Optional[str], Optional[str]]] = []
        question_clean = re.sub(r"[^a-z0-9\s]", " ", question_lower)
        seen: set[str] = set()

        # 1) Full state names
        for state_name, abbr in STATE_NAME_TO_ABBR.items():
            pattern = r"\b" + re.escape(state_name.lower()) + r"\b"
            if re.search(pattern, question_clean):
                if state_name not in seen:
                    state_mentions.append((state_name, abbr))
                    seen.add(state_name)

        # 2) If no full names found, try 2-letter abbreviations
        if not state_mentions:
            tokens = re.findall(r"\b[a-z]{2}\b", question_clean)
            for tok in tokens:
                abbr = tok.upper()
                full = STATE_ABBR_TO_NAME.get(abbr)
                if full and full not in seen:
                    state_mentions.append((full, abbr))
                    seen.add(full)

        where_clause: Optional[str] = None
        if state_mentions:
            if len(state_mentions) == 1:
                state_name = state_mentions[0][0]
                where_clause = f"state_name = {sql_literal(state_name)}"
            else:
                names = sorted(set(s for (s, a) in state_mentions if s))
                in_list = ", ".join(sql_literal(s) for s in names)
                where_clause = f"state_name IN ({in_list})"

        if where_clause:
            return (
                "SELECT SUM(ST_Area(geom::geography)/1000000) AS total_area_km2\n"
                "FROM counties\n"
                f"WHERE {where_clause}"
            )
        else:
            return (
                "SELECT SUM(ST_Area(geom::geography)/1000000) AS total_area_km2\n"
                "FROM counties"
            )

    if 'largest' in question_lower or 'biggest' in question_lower:
        return (
            "SELECT name, state_name, "
            "ST_Area(geom::geography)/1000000 as area_km2 "
            "FROM counties "
            "ORDER BY area_km2 DESC LIMIT 5"
        )

    if 'smallest' in question_lower:
        return (
            "SELECT name, state_name, "
            "ST_Area(geom::geography)/1000000 as area_km2 "
            "FROM counties "
            "WHERE ST_Area(geom::geography) > 0 "
            "ORDER BY area_km2 ASC LIMIT 5"
        )

    # Generic super-fallback
    return "SELECT name, state_name FROM counties LIMIT 10"


@tool
def execute_sql_query(sql_query: str) -> str:
    """
    Execute a SQL query against the US counties database and return the results.
    ALWAYS updates st.session_state["map_features"] so the map reflects the latest query.
    """
    try:
        sql_upper = sql_query.upper()

        # Forbidden geometry creation
        if 'ST_GEOMFROMTEXT' in sql_upper or 'ST_MAKEPOINT' in sql_upper:
            return (
                "Error: Cannot create new geometry objects. "
                "Use the existing geom column.\n\n"
                f"Generated SQL:\n{sql_query}"
            )

        # Unterminated string check
        if sql_query.count("'") % 2 != 0:
            return (
                "Error: Invalid SQL with unterminated string.\n\n"
                f"Generated SQL:\n{sql_query}"
            )

        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute(sql_query)

        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()

        if not rows:
            st.session_state["map_features"] = []
            cur.close()
            conn.close()
            return "No results found."

        # ----------------------------------------------------------
        # TEXT RESULT FORMATTING (unchanged)
        # ----------------------------------------------------------
        if len(rows) == 1 and len(columns) == 1:
            col_name = columns[0].lower()
            value = rows[0][0]

            if 'total' in col_name or 'sum' in col_name:
                if isinstance(value, (int, float)):
                    if 'area' in col_name:
                        result_text = f"**Total Area:** {value:,.2f} km²"
                    else:
                        result_text = f"**Total:** {value:,.2f}"
                else:
                    result_text = f"**Total:** {value}"

            elif 'count' in col_name:
                result_text = f"**Count:** {value:,}"

            elif 'avg' in col_name or 'average' in col_name:
                result_text = f"**Average:** {value:,.2f}" if isinstance(value, (int, float)) else f"**Average:** {value}"

            elif 'max' in col_name or 'maximum' in col_name:
                result_text = f"**Maximum:** {value:,.2f}" if isinstance(value, (int, float)) else f"**Maximum:** {value}"

            elif 'min' in col_name or 'minimum' in col_name:
                result_text = f"**Minimum:** {value:,.2f}" if isinstance(value, (int, float)) else f"**Minimum:** {value}"

            else:
                result_text = f"**Result:** {value}"

        else:
            result_text = f"**Found {len(rows)} result(s):**\n\n"

            # Hide heavy geometry columns in the text output
            hidden_cols = {"geom", "geom_geojson"}
            visible_indices = [
                idx for idx, col in enumerate(columns)
                if col.lower() not in hidden_cols
            ]
            visible_columns = [columns[idx] for idx in visible_indices]

            for i, row in enumerate(rows[:10], 1):
                if len(visible_columns) == 1:
                    val = row[visible_indices[0]]
                    result_text += f"{i}. {visible_columns[0]}: {val}\n"
                else:
                    row_data_parts = [
                        f"{col}: {row[idx]}"
                        for col, idx in zip(visible_columns, visible_indices)
                    ]
                    row_data = ", ".join(row_data_parts)
                    result_text += f"{i}. {row_data}\n"

            if len(rows) > 10:
                result_text += f"\n*...and {len(rows) - 10} more*"

        # ----------------------------------------------------------
        # MAP FEATURE GENERATION (ENHANCED: supports COUNT(*) queries)
        # ----------------------------------------------------------
        try:
            # By default, use the rows/columns from the main query
            map_rows = rows
            map_columns = columns

            lower_cols = [c.lower() for c in columns]
            has_geom = any(c in ("geom", "geom_geojson") for c in lower_cols)

            if not has_geom:
                # 1) COUNT(*) AS county_count → fetch those counties' geometries
                m_count = re.match(
                    r"SELECT\s+COUNT\(\*\)\s+AS\s+county_count\s+FROM\s+counties\s+WHERE\s+(.+)$",
                    sql_query.strip(),
                    re.IGNORECASE | re.DOTALL,
                )
                if m_count:
                    where_clause = m_count.group(1).rstrip(" ;")

                    geom_sql = (
                        "SELECT name, state_name, state_abbr, geoid,\n"
                        "       ST_AsGeoJSON(geom) AS geom_geojson,\n"
                        "       ST_Y(ST_Centroid(geom)) AS centroid_lat,\n"
                        "       ST_X(ST_Centroid(geom)) AS centroid_lon\n"
                        "FROM counties\n"
                        f"WHERE {where_clause}\n"
                        "ORDER BY name"
                    )

                    with conn.cursor() as cur2:
                        cur2.execute(geom_sql)
                        map_rows = cur2.fetchall()
                        map_columns = [desc[0] for desc in cur2.description]

                else:
                    # 2) SUM(ST_Area(...)) AS total_area_km2 → build per-state polygons
                    m_area = re.match(
                        r"SELECT\s+SUM\s*\(\s*ST_Area\(geom::geography\)\s*/\s*1000000\s*\)\s+AS\s+total_area_km2\s+FROM\s+counties\s+WHERE\s+(.+)$",
                        sql_query.strip(),
                        re.IGNORECASE | re.DOTALL,
                    )
                    if m_area:
                        where_clause = m_area.group(1).rstrip(" ;")

                        # One feature per state (so TX and CA both appear)
                        geom_sql = (
                            "SELECT state_name, state_abbr,\n"
                            "       ST_AsGeoJSON(ST_Union(geom)) AS geom_geojson,\n"
                            "       ST_Y(ST_Centroid(ST_Union(geom))) AS centroid_lat,\n"
                            "       ST_X(ST_Centroid(ST_Union(geom))) AS centroid_lon\n"
                            "FROM counties\n"
                            f"WHERE {where_clause}\n"
                            "GROUP BY state_name, state_abbr\n"
                            "ORDER BY state_name"
                        )

                        with conn.cursor() as cur2:
                            cur2.execute(geom_sql)
                            map_rows = cur2.fetchall()
                            map_columns = [desc[0] for desc in cur2.description]


            # Build map features from either the original query or the geometry query
            map_features = build_map_features(conn, map_rows, map_columns)

            # Fallback: for weird queries, try to parse county/state from the SQL itself
            if not map_features:
                map_features = build_single_feature_from_sql(conn, sql_query)

            st.session_state["map_features"] = map_features or []

        except Exception:
            st.session_state["map_features"] = []

        cur.close()
        conn.close()
        return result_text

    except Exception as e:
        st.session_state["map_features"] = []
        return (
            "Database connection error: "
            f"{e}. Check database configuration."
        )

@tool
def get_database_schema() -> str:
    """
    Get information about the database schema for US counties.

    Returns:
        A string describing the database schema
    """
    return SCHEMA_INFO

#=============================================================================
#LANGGRAPH / AGENT SETUP
#=============================================================================
def create_agent():
    """Create a simple agent that uses our tools directly and logs interactions."""

    def simple_agent(question: str) -> str:
        """Simple agent that converts NL to SQL and executes it."""
        try:
            t0 = time.time()
            # Step 1: Convert natural language to SQL
            sql_query = convert_nl_to_sql.invoke({"question": question})
            t1 = time.time()

            # Which backend was actually used (set inside convert_nl_to_sql)
            backend_label = st.session_state.get(
                "last_backend_label",
                "Unknown backend",
            )

            # Step 2: Execute the SQL query
            results = execute_sql_query.invoke({"sql_query": sql_query})
            t2 = time.time()

            nl_to_sql_ms = (t1 - t0) * 1000.0
            db_query_ms = (t2 - t1) * 1000.0

            # Step 3: Log this interaction for later evaluation
            if "eval_log" not in st.session_state:
                st.session_state["eval_log"] = []

            st.session_state["eval_log"].append(
                {
                    "question": question,
                    "backend": backend_label,
                    "sql": sql_query,
                    "nl_to_sql_ms": round(nl_to_sql_ms, 1),
                    "db_query_ms": round(db_query_ms, 1),
                    "result_preview": str(results)[:500],
                }
            )

            # Step 4: Format the response (include backend + timings)
            response = (
                f"_Backend used: {backend_label}_\n\n"
                "I'll help you answer that question about US counties.\n\n"
                f"**Generated SQL query:**\n{sql_query}\n\n"
                f"**Results:**\n{results}\n\n"
                f"_(NL→SQL time: {nl_to_sql_ms:.1f} ms, DB time: {db_query_ms:.1f} ms)_\n\n"
                "Is there anything else you'd like to know about US counties?"
            )

            return response

        except Exception as e:
            return (
                "I encountered an error while processing your question: "
                f"{e}. Please try rephrasing your question."
            )

    return simple_agent


# Initialize the agent
@st.cache_resource
def get_agent():
    """Get or create the agent (cached for performance)"""
    return create_agent()


#=============================================================================
#AGENT EXECUTION FUNCTIONS
#=============================================================================


def run_agent(question: str) -> str:
    """Run the agent with a user question and return the response"""
    try:
        agent = get_agent()
        return agent(question)
    except Exception as e:
        return f"Error running agent: {e}"

"""Streamlit App Configuration"""


"""App Title"""

st.title("🗺️ Chat with the Map")
st.markdown("Ask questions about US Counties in natural language! Powered by LangGraph ReAct Agent with Llama 3.2-3B")

"""Sidebar with example questions + backend toggle"""

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

    st.markdown("---")
    st.header("LLM Backend")

    # Default backend = local (Ollama)
    if "llm_backend" not in st.session_state:
        st.session_state["llm_backend"] = "local"


    options = ["Local (Ollama)", "Remote (Groq)"]

    current = st.session_state["llm_backend"]
    if current == "local":
        default_index = 0
    elif current == "groq":
        default_index = 1
    else:
        default_index = 0 

    backend_label = st.radio(
        "Choose SQL generator:",
        options,
        index=default_index,
    )

    # Store a simple internal code
    if backend_label.startswith("Local"):
        st.session_state["llm_backend"] = "local"
    else:
        st.session_state["llm_backend"] = "groq"

    st.caption(
        "• Local = Ollama model at OLLAMA_URL\n"
        "• Remote (Groq) = `llama-3.3-70b-versatile` via `GROQ_API_KEY`"
    )

    # Tiny in-app evaluation log (helps for screenshots)
    if "eval_log" in st.session_state and st.session_state["eval_log"]:
        st.markdown("---")
        st.subheader("📊 Recent runs (this session)")
        for item in st.session_state["eval_log"][-5:][::-1]:
            st.markdown(
                f"- **Backend:** {item.get('backend', 'Unknown')}  \n"
                f"  **Q:** {item['question'][:60]}…  \n"
                f"  **SQL:** `{item['sql'][:60]}…`"
            )

"""Initialize chat history"""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "map_features" not in st.session_state:
    st.session_state.map_features = []

if "logs" not in st.session_state:
    st.session_state.logs = []

"""Display chat history"""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

"""Chat input"""

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
                error_msg = f"Error: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

"""Handle example question clicks"""

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
                error_msg = f" Error: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

"""Log panel"""

st.subheader("📜 Log Panel")

logs = st.session_state.get("eval_log", [])

if logs:
    with st.expander("Show detailed logs", expanded=False):
        # Show newest first, like you were doing
        for idx, entry in enumerate(reversed(logs), 1):
            q = entry.get("question", "")
            sql = entry.get("sql", "")
            backend = entry.get("backend", "Unknown backend")
            nl2sql = entry.get("nl_to_sql_ms", "N/A")
            db_ms = entry.get("db_query_ms", "N/A")

            st.markdown(
                f"**#{len(logs) - idx + 1}**\n\n"
                f"- **Backend:** `{backend}`\n"
                f"- **Prompt:** `{q}`\n"
                f"- **SQL:** `{sql}`\n"
                f"- **NL→SQL time:** {nl2sql} ms\n"
                f"- **DB query time:** {db_ms} ms\n"
                "---"
            )
else:
    st.markdown("_No log entries yet. Ask a question to see logs here._")

"""Map visualization"""

st.subheader("🗺 Map View")
render_map(st.session_state.get("map_features", []))

