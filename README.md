# MACE-JAX

JAX implementations of MACE atomistic models, including the pretrained
MACE-POLAR-1-M model, with automatic energy, force, and stress evaluation.
This fork adds MACE-POLAR inference and electrostatic coupling to fixed
molecular mechanics charges. It is a library, not a paper reproduction archive.

The original MACE-JAX port was developed by Philipp Benner, Abhijeet Gangan,
Mario Geiger, and Ilyes Batatia. The MACE-POLAR JAX port and ML/MM additions
are maintained in this fork by Venkat Kapil.

## Installation

Use Python 3.11 or newer. Install this fork from its repository:

```sh
git clone https://github.com/venkatkapil24/mace-jax.git
cd mace-jax
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For NVIDIA CUDA 12, install `'.[jax-cuda12]'` instead of `.`; for CUDA 13,
use `'.[jax-cuda13]'`. Install only the extra matching your CUDA environment.
The base installation uses CPU JAX unless a GPU-enabled JAX is already installed.

Checkpoint conversion also needs `mace-torch`. The `torch-cuda12` and
`torch-cuda13` extras install the corresponding optional PyTorch cuEq stack.
Model weights are distributed by the original model authors, not this repository.

## MACE-POLAR

- Retains the MACE-POLAR-1-M pretrained weights and polarization architecture.
- Supports charge, spin multiplicity, external electric fields, and periodic cells.
- Supplies energy, forces, charges, dipoles, and periodic stress.
- Supports direct reciprocal electrostatics and generalized PPPM for bulk cells.
- Supports fixed-charge electrostatic ML/MM embedding in bulk periodic cells.
- Uses fixed-shape graphs for JIT compilation and supports cuEq acceleration.

MACE-POLAR is the architecture; MACE-POLAR-1-M is the pretrained model.
MACE-POLAR-1-M-JAX denotes its implementation in this fork. Conversion does
not retrain the model or change its electronic structure reference level.

## Tests

```sh
python -m pip install -e '.[test]'
JAX_PLATFORMS=cpu python -m pytest
```

The default suite excludes tests marked `slow`. Run `python -m pytest -m slow`
to select them. CUDA tests require a GPU and the appropriate CUDA extras;
optional integration tests skip when their external software is unavailable.
GitHub Actions runs the focused MACE-POLAR CPU suite defined in
[the workflow](.github/workflows/tests.yml).
`scripts/verify_polar_backbone.py` compares a trusted pretrained checkpoint
against the original PyTorch model; use `--help` for its options.

## Citation and license

See [CITATION.cff](CITATION.cff) for software attribution.
Please also cite the [MACE](https://arxiv.org/abs/2206.07697) and
[MACE-POLAR](https://arxiv.org/abs/2602.19411) papers when using these models.
The source is distributed under the [MIT license](MIT.md).
Model weights and optional dependencies retain their own licenses.
