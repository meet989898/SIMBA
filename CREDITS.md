# Credits and Licensing Notes

This project includes or depends on the following third-party software/components.

## Chessground

- Project: Chessground (Lichess board UI component)
- Use here: interactive board in `components/cg_board/frontend/index.html`
- Attribution: Lichess / Chessground contributors
- License: GPL (commonly GPL-3.0-or-later for Chessground distributions; verify vendor snapshot license file)

## chess.js

- Project: chess.js
- Use here: legal move generation and frontend move validation in the custom Streamlit component
- Attribution: chess.js contributors
- License: BSD-style (commonly BSD-2-Clause; verify vendor snapshot license file)

## Stockfish

- Project: Stockfish chess engine
- Use here: backend UCI engine evaluation via `python-chess`
- Attribution: Stockfish developers
- License: GPL-3.0-or-later

## python-chess

- Project: python-chess
- Use here: board state, legal moves, PGN parsing, and UCI engine wrapper
- Attribution: Niklas Fiekas and contributors
- License: GPL-3.0-or-later

## Streamlit

- Project: Streamlit
- Use here: application UI runtime + custom component integration
- Attribution: Streamlit, Inc. and contributors
- License: Apache-2.0

## FAISS (Optional)

- Project: FAISS (Facebook AI Similarity Search)
- Use here: exact/ANN index backends (`faiss_flat`, `faiss_hnsw`, `faiss_ivf`) when installed
- Attribution: Meta / FAISS contributors
- License: MIT (verify installed package/distribution)

## Vendor Asset Notes

Vendor frontend assets are expected in:

- `components/cg_board/frontend/vendor/chessground.*`
- `components/cg_board/frontend/vendor/chess.min.js`

Keep upstream LICENSE files and attribution when packaging or publishing the repository.

