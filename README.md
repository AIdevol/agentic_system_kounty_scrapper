# King County Agentic FastAPI Backend

This project is a **FastAPI backend** that exposes an API for an **agentic LLM system** focused on **King County** (e.g. property info, zoning, neighborhoods, etc.).

The backend is structured with:

- `app/` – FastAPI application, routes and dependencies
- `llm/` – LLM orchestration, prompt templates and agent logic for King County

---

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate    # on macOS/Linux
# .venv\Scripts\activate     # on Windows (PowerShell)

pip install -r requirements.txt
```

Set **only** `GROQ_API_KEY` in `.env` or the environment:

- **GROQ_API_KEY** – required. Used for all LLM answers (Groq Llama 3.1 8B). Get a key at [console.groq.com](https://console.groq.com).
- **DATASET_PATH** or **DATASET_JSON_PATH** – optional. Path to your dataset file (CSV or JSON). Defaults to `app/us_parcel_dataset_2000.json` or project root.

RAG embeddings use a **local** model (sentence-transformers); no Gemini, Hugging Face token, or other API key is required.

Example:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Parcel dataset:** Questions over `us_parcel_dataset_2000.json` are served by `POST /v1/dataset-rag/query`. The system indexes the JSON file at startup (or on first request) and uses RAG to generate answers from the retrieved rows. The dataset may include an `owners` array per parcel (junction-table style).

---

## Running the API

Start the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

- Interactive docs (Swagger): `http://127.0.0.1:8000/docs`
- ReDoc docs: `http://127.0.0.1:8000/redoc`
- **Frontend (public chat UI):** `http://127.0.0.1:8000/app/` — the app serves the frontend so anyone with the URL can use it.

### Public access (anyone on the network or internet)

To allow others to open the frontend and use the app:

1. Run the server bound to all interfaces: `make public` or `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
2. Share the URL: `http://<your-ip>:8000/` or `http://<your-ip>:8000/app/`. Root redirects to `/app/`.
3. The frontend automatically uses the same host for API calls, so no extra config is needed.

Behind Nginx or a reverse proxy, use your public domain (e.g. `https://yourdomain.com/app/`).

---

## Docker (Backend Only)

This Docker setup runs **only the FastAPI backend**.

1. Build the image:

```bash
docker build -t king-county-backend .
```

2. Run the backend container:

```bash
docker run --env-file .env -p 8000:8000 --name king-county-backend king-county-backend
```

3. API will be available at:

- `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

You can also run with Docker Compose (backend only):

```bash
docker compose -f docker-compose.backend.yml up --build -d
```

Stop it with:

```bash
docker compose -f docker-compose.backend.yml down
```

---

## High-level Architecture

- `app/main.py`
  - Creates the FastAPI app
  - Includes the King County agent router under `/v1/king-county`
  - Provides a health check endpoint

- `app/api/v1/king_county.py`
  - Defines request/response schemas
  - Exposes a `/query` endpoint to interact with the King County agent

- `llm/king_county_agent.py`
  - Encapsulates the **agentic workflow** for King County queries
  - Uses `llm/client.py` to talk to the underlying LLM provider
  - Defines a structured "view" of King County (domain-specific tools and context)

- `llm/prompts.py`
  - Contains prompt templates and system messages for the King County domain

This layout is intentionally simple so you can extend tools, memory, and multi-step agent behavior as needed.

---

## Makefile Commands

This repository now includes a `Makefile` to simplify local and production runs.

```bash
make install    # create .venv + install dependencies
make dev        # local dev server with reload
make prod       # production-style server on 127.0.0.1:8000 (for Nginx)
make public     # direct public server on 0.0.0.0:8000 (no Nginx)
make health     # checks /health endpoint
```

---

## Deploy Publicly with Nginx (Recommended)

Use Nginx as a reverse proxy and keep Uvicorn bound to localhost.

1. **Run backend in production mode**

```bash
make install
make prod
```

2. **Install and enable Nginx site config on your server**

```bash
sudo apt update
sudo apt install -y nginx

sudo cp deploy/nginx/agentic_system_kounty_scrapper.conf /etc/nginx/sites-available/agentic_backend
sudo ln -s /etc/nginx/sites-available/agentic_backend /etc/nginx/sites-enabled/agentic_backend
sudo rm -f /etc/nginx/sites-enabled/default
```

3. **Set your domain in the Nginx config**

Edit:

```bash
sudo nano /etc/nginx/sites-available/agentic_backend
```

Replace:

- `server_name example.com www.example.com;`

with your real domain(s).

4. **Validate and reload Nginx**

```bash
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
```

5. **Open firewall (if enabled)**

```bash
sudo ufw allow 'Nginx Full'
```

6. **Optional: add HTTPS with Let's Encrypt**

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

After this, your backend is publicly available through your domain via Nginx.

