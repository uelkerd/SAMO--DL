# Notebooks

This folder contains Jupyter notebooks for model training, experiments, and demos.

- training/: End-to-end training workflows (GPU-friendly)
- experiments/: Exploratory work and prototypes
- demos/: Minimal examples
- legacy/: Older reference notebooks

Contribution guidelines:
- Keep notebooks focused and small; prefer modular reusable code in `src/`.
- Avoid committing large outputs; prefer text logs and save models/artifacts outside the repo.
- Ensure any new dependencies are added to the appropriate requirements files and verified for security.