## Chat with the Map – Natural Language Interface for US Counties

**Chat with the Map** is a Streamlit application that lets you ask natural‑language questions about U.S. counties and see results both as text and on an interactive map.

![Chat with the Map demo](Untitled%20design%20(1).gif)

Under the hood, it:
- **Converts natural language to SQL** using a LangGraph / LangChain ReAct-style agent
- **Queries a PostGIS database** containing all U.S. counties (TIGER 2018 data)
- **Visualizes results** with PyDeck (optionally using Mapbox basemaps)
- Uses **Ollama (Llama 3.2–3B)** by default, with **OpenAI** as a fallback if configured

For the original repository, see [`chat_with_app` on GitHub](https://github.com/dhanushchalicheemala/chat_with_app).

---

## 1. Requirements

- **Python**: 3.10+ (virtual environment recommended)
- **PostgreSQL**: 17 (or compatible) with **PostGIS** extension
- **TIGER 2018 counties CSV**: `TIGER2018_COUNTY_with_state.csv` (already included in this repo)
- Optional but recommended:
  - **Ollama** running a Llama 3.2–3B model (`llama3.2:3b`)
  - **Mapbox account** for nicer basemaps
  - **OpenAI API key** (only if you want to use OpenAI as a fallback LLM)

---

## 2. Setup & Installation

From your terminal:

```bash
cd /Users/dhanush/Desktop/Capstone  # or the directory where you cloned the repo

# (Recommended) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

---

## 3. Environment Variables (`env.example`)

An example environment file is provided as `env.example` in the project root.

Copy it and rename to `.env` (or load it via your shell / process manager), then fill in your own values:

```bash
cp env.example .env
```

The key variables used by the app (`app.py`) are:

- **Database**
  - `PGDATABASE` – PostgreSQL database name (default used in code: `USCountyDB`)
  - `PGUSER` – PostgreSQL username
  - `PGPASSWORD` – PostgreSQL password
  - `PGHOST` – PostgreSQL host (default: `localhost`)
  - `PGPORT` – PostgreSQL port (default: `5432`)

- **LLM / Agents**
  - `OLLAMA_URL` – Base URL for your Ollama instance  
    - Example (local): `http://localhost:11434`
  - `OPENAI_API_KEY` – OpenAI API key (used only if set, as a fallback when Ollama is not available)

- **Map Visualization**
  - `MAPBOX_API_KEY` – Mapbox token (or alternatively `MAPBOX_TOKEN`) for PyDeck basemaps

If these variables are not set, the app falls back to the defaults shown in `app.py` (e.g. `USCountyDB`, user `dhanush`, host `localhost`, etc.).

---

## 4. Database Setup (PostgreSQL + PostGIS)

The application expects a PostGIS database with a single table called `counties` containing all U.S. counties.

Full step‑by‑step database setup commands are documented in `testing/DATABASE_SETUP_COMMANDS.md`.  
At a high level:

1. **Install PostgreSQL and PostGIS** (example for macOS via Homebrew):
   ```bash
   brew install postgresql@17 postgis
   brew services start postgresql@17
   ```
2. **Create the database and enable PostGIS**:
   ```bash
   createdb USCountyDB
   psql -d USCountyDB -c "CREATE EXTENSION IF NOT EXISTS postgis;"
   ```
3. **Create the `counties` table and spatial index**  
   Use the SQL in `testing/DATABASE_SETUP_COMMANDS.md` (section “Create Counties Table”).

4. **Load the TIGER 2018 data** using the provided loader script:
   ```bash
   # From the project root
   python load_data.py
   ```

This reads `TIGER2018_COUNTY_with_state.csv`, parses the WKT geometries, and inserts all ~3,232 counties into the `counties` table.

---

## 5. Running the Application

Once:
- Python dependencies are installed
- The database is created, PostGIS enabled, and data loaded
- Environment variables are configured (via `.env` or your shell)

You can start the Streamlit app with:

```bash
cd /Users/dhanush/Desktop/Capstone  # project root
streamlit run app.py
```

Streamlit will print a local URL such as `http://localhost:8501`.  
Open that URL in your browser to use the app.

---

## 6. How the App Works

- The Streamlit frontend provides:
  - A **chat interface** where you ask questions like:
    - “How many counties are in California?”
    - “Show me counties with area greater than 10,000 km²”
    - “Which states border Texas?”
  - A **map view** that visualizes the resulting counties using PyDeck.

- The backend logic in `app.py`:
  - Uses a LangGraph / LangChain **tool‑calling agent** to:
    1. Convert natural language into SQL targeting a single `counties` table
    2. Execute the SQL against your PostGIS database
    3. Build map features (GeoJSON + centroids) for visualization
  - Uses **Ollama** at `OLLAMA_URL` first; if that fails and `OPENAI_API_KEY` is set, it falls back to **OpenAI**.

The schema and many example queries are embedded in `app.py` and also exposed via a `get_database_schema` tool.

---

## 7. Troubleshooting

- **Database connection errors**  
  - Verify PostgreSQL is running and you can connect with:
    ```bash
    psql -d "$PGDATABASE" -U "$PGUSER" -h "$PGHOST" -p "$PGPORT"
    ```
  - Make sure `PGDATABASE`, `PGUSER`, `PGHOST`, `PGPORT`, and `PGPASSWORD` (if needed) are set correctly.

- **No map appears / empty results**  
  - Confirm the `counties` table exists and contains data:
    ```bash
    psql -d USCountyDB -c "SELECT COUNT(*) FROM counties;"
    ```

- **LLM not responding**  
  - For Ollama: ensure the Ollama server is running and `OLLAMA_URL` is reachable.
  - For OpenAI: confirm `OPENAI_API_KEY` is set and valid.

For detailed database setup and management commands, see `testing/DATABASE_SETUP_COMMANDS.md`.

