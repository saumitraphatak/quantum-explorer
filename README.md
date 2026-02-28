# Quantum Physics Explorer ⚛️

An interactive web app for exploring quantum computing concepts visually — no physics background required.

## Features

- **Bloch Sphere** — drag θ and φ sliders to move a qubit state vector in 3-D space
- **Quantum Gates** — apply X, Y, Z, H, S, T, and rotation gates and see the Bloch sphere update
- **Superposition & Interference** — control amplitudes and watch a Mach-Zehnder style demo
- **Measurement** — simulate the Born rule with up to 10 000 shots
- **Entanglement** — explore all four Bell states and their correlations

## Run locally

```bash
pip install -r requirements.txt
streamlit run quantum_explorer.py
```

## Deploy for free

[![Deploy on Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

Push this repo to GitHub, then go to [share.streamlit.io](https://share.streamlit.io) and connect the repo — it deploys automatically.
