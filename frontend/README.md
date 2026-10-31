# FEVERScope — Frontend

React 19 + Vite 8 single-page application for the FEVERScope evidence retrieval benchmark.

## Tech Stack

- **React 19** — UI framework
- **Vite 8** — dev server and bundler
- **Vanilla CSS** — custom design system (`App.css`, `index.css`)
- **Oxlint** — fast JavaScript linter

## Structure

```
frontend/
├── src/
│   ├── App.jsx       # Main application component — search, results, benchmarks
│   ├── App.css       # Component styles and layout
│   ├── index.css     # Global CSS reset and design tokens
│   └── main.jsx      # React root entry point
├── public/           # Static assets
├── index.html        # HTML template
├── vite.config.js    # Vite configuration
└── package.json
```

## Development

```bash
# Install dependencies
npm install

# Start dev server (http://localhost:5173)
npm run dev

# Lint
npm run lint

# Production build
npm run build

# Preview production build
npm run preview
```

> **Note:** The frontend expects the FEVERScope FastAPI backend to be running on
> `http://localhost:8000`. Use `../start.sh` (macOS/Linux) or `../start.bat` (Windows)
> from the repo root to launch both services together.

## Backend API

The app communicates with the FastAPI backend via these endpoints:

| Endpoint | Usage |
|----------|-------|
| `GET /api/health` | Service status |
| `GET /api/status` | Retriever readiness |
| `GET /api/benchmarks` | Global benchmark table |
| `GET /api/sample-claims` | Pre-loaded example claims |
| `POST /api/search` | Multi-method retrieval + metrics |
| `POST /api/pinecone-config` | Update Pinecone credentials |

See the full API reference at **http://localhost:8000/docs** (Swagger UI).
