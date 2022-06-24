# MCNP Conversion Tools for OpenMC

[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

This repository provides tools for parsing/converting MCNP models to OpenMC
classes and/or XML files. To install these tools, run:

    python -m pip install git+https://github.com/openmc-dev/openmc_mcnp_adapter.git

This makes the `openmc_mcnp_adapter` Python module and `mcnp_to_openmc` console
script available. To convert an MCNP model, run:

    mcnp_to_openmc mcnp_input

## Disclaimer

There has been no methodical V&V this converter, so use at your own risk!

## Known Limitations

The converter currently only handles geometry and material information; source
definition (SDEF) and tally specifications are ignored. Some geometry features
are also not currently supported:

- `LIKE n BUT` form for cell cards
- Periodic boundary conditions
- `X`, `Y`, and `Z` surfaces with 1 or 3 coordinate pairs
- `RHP`, `REC`, `TRC`, `ELL`, `WED`, and `ARB` macrobodies
- Hexagonal lattices
- One-dimensional lattices
- Two-dimensional lattices with basis other than x-y
