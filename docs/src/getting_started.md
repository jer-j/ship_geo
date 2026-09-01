# Getting started

The current upstream Cython 0.29.28 build supports Python 3.10.

```bash
git clone https://github.com/jer-j/ship_geo.git
cd ship_geo
python -m pip install -e ".[test]"
```

Run the verification suite and ship-section demonstration:

```bash
pytest
python examples/f_spline_ship_section.py
```

The public Python namespace remains `lsdo_geo` so existing LSDO code continues
to import unchanged.
