
# Documentation

Install the documentation dependencies and build from the repository root:

```bash
python -m pip install -e ".[docs]"
make -C docs html
```

The generated site is written to `docs/_build/html`.
