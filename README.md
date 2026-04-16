# Chess Similarity Capstone Demo (Streamlit)

Presentation-ready Streamlit demo for:

`Similarity Search at Scale: Benchmarking Algorithms with Real-World Decision Spaces`

This repo uses a chess-position similarity workflow to demonstrate:

- interactive chess analysis with branching variations
- Stockfish evaluation overlays (eval bar, numeric eval, best move arrow, PV line)
- exact and FAISS-based similarity search over encoded chess positions
- a lightweight benchmark runner for latency/build-time/recall comparisons
- a deterministic Demo Lab with curated scenarios and vector explainability

## Project Structure

- `app.py`: Main Streamlit app entry point (UI + wiring).
- `components/cg_board/__init__.py`: Streamlit custom component wrapper (`declare_component`).
- `components/cg_board/frontend/index.html`: Chessground + chess.js frontend that returns move/FEN events via `Streamlit.setComponentValue`.
- `chess_state.py`: Branch-aware analysis tree (mainline + variations) and PGN export with variations.
- `pgn_loader.py`: PGN streaming/parsing utilities and in-memory combined corpus builder (Option B pipeline).
- `encoder.py`: Deterministic chess position embedding (base 781 + configurable feature extensions).
- `search.py`: Brute-force and FAISS search backends with a clean build/query API.
- `eval_engine.py`: Stockfish wrapper, score normalization, and eval-bar helpers.
- `benchmarks.py`: Build/query latency + memory + recall benchmark scaffolding.
- `data/sample_games/`: Curated PGNs for demos.
- `data/presets.json`: Curated presentation FEN positions.
- `data/demo/demo_games.pgn`: Bundled custom demo games.
- `data/demo/demo_positions.json`: Named demo positions (FEN library + optional demo eval annotations).
- `data/demo/demo_scenarios.json`: Fixed curated scenarios (query/candidate refs, expected bands, narrative).
- `scripts/validate_demo_pack.py`: Self-check for scenario integrity, determinism, band checks, and ranking-flip intent.
- `CREDITS.md`: Third-party attribution and license notes.

## UI Features (Current Implementation)

- Chessground board with drag/drop + click moves
- Legal move enforcement (frontend + backend validation)
- Move highlight and check highlight
- Promotion chooser (Q/R/B/N)
- Undo/redo stepping, jump to start/end
- Branching analysis mode vs strict game mode
- Sample PGN loading and curated FEN preset loading
- Similar results list (click to load full game and jump to matched move)
- Separate similar-game viewer with independent stepping controls
- Stockfish eval bar, numeric eval, best-move arrow, PV line toggles
- Demo Lab tab with:
  - fixed scenario selector and score-band editor
  - 0-100 normalized similarity score + raw distance
  - explainability panel (feature-group contributions + interpretable delta table)
  - A/B/C profile comparison (base-only, base+domain, full+eval)
  - ranking-shift demonstrator for curated comparison candidates
- Search method selection (`brute`, `faiss_flat`, `faiss_hnsw`, `faiss_ivf`)
- Benchmark tab with build time, query latency p50/p95/p99, estimated index footprint, and optional recall@k

## Stockfish Notes

- The app now checks bundled local executables first, then Linux host paths (`/usr/games/stockfish`, `/usr/bin/stockfish`), then the Windows fallback path.
- The app caches the engine process with Streamlit resource caching to avoid relaunching on every rerun.
- If Stockfish is missing or inaccessible, the app degrades gracefully and continues without engine overlays.

## Public Streamlit Deployment

This project is prepared for **GitHub + Streamlit Community Cloud** deployment.

Recommended deployment shape:

- publish from a **slim deployment branch/repo**
- use repo root `app.py` as the Streamlit entrypoint
- include runtime app files, `components/cg_board`, `data/sample_games`, `data/demo`, `data/presets.json`, and the bundled `data/bulk_pgn`
- exclude local/non-runtime artifacts such as `outputs/`, `.venv/`, IDE folders, poster/submission folders, and other workspace-only files
- do not include Windows-only Stockfish binaries in the hosted Linux deploy branch

Deployment manifests included at repo root:

- `requirements.txt`
- `packages.txt`
- `runtime.txt`
- `.streamlit/config.toml`

Hosted app behavior differences:

- public corpus selection is limited to **Curated sample PGNs** and **Bundled bulk PGNs**
- arbitrary server filesystem PGN paths are not exposed in the hosted flow
- expensive controls are capped for cloud stability:
  - Stockfish depth defaults to `8` and caps at `12`
  - corpus positions default to `1500` and cap at `10000`
  - benchmark sample queries default to `20` and cap at `75`
- the in-app benchmark tab is intended as a lightweight public comparison view; heavy offline benchmark/report scripts remain available separately

## PGN Pipeline Choice (Implemented)

This repo implements **Option B** from the design spec:

- scan multiple PGNs and build a combined in-memory corpus without writing a merged PGN file

Why this choice:

- avoids expensive large-file rewrite/merge steps on Windows
- keeps ingestion streaming-friendly
- simplifies reproducibility for small/medium demo corpora
- still supports large PGN inputs safely by parsing one game at a time

## Notes for Open-Sourcing

- This project intentionally uses Chessground (GPL-compatible workflow for this repo).
- Keep `CREDITS.md` and upstream license texts when redistributing vendor assets.
- Verify exact third-party versions/license files included in `components/cg_board/frontend/vendor/`.
