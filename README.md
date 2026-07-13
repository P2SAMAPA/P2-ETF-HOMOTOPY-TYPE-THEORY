# Homotopy Type Theory for ETFs

Treats portfolios as terms in a dependent type theory. Equality is homotopy path equivalence. Higher inductive types encode trading constraints. The univalence axiom allows transport of strategies across isomorphic market structures. The per‑ETF score is the homotopy level.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Path equivalence via correlation of return paths
- Higher inductive types via hierarchical clustering
- Univalence axiom via Kolmogorov-Smirnov isomorphism
- Score = homotopy level (higher = more structure)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-homotopy-type-theory-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High homotopy level → more type-theoretic structure → potential alpha.
- Low homotopy level → structure is simple.

## Requirements

See `requirements.txt`.
