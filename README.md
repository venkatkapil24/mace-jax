# MACE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :rocket: [JAX](https://github.com/google/jax)

This repository contains a **porting** of MACE in **jax** developed by [**Philipp Benner**](https://bamescience.github.io/), **Abhijeet Gangan**, **Mario Geiger** and **Ilyes Batatia**.

## TorchAX verifier for MACE-OFF23 small

`scripts/verify_mace_torchax.py` compares a trusted local MACE checkpoint on
native eager Torch and eager TorchAX, using the same ASE structure and graph.
The first target is [MACE-OFF23 small](https://github.com/ACEsuit/mace-off/tree/main/mace_off23),
a standard `ScaleShiftMACE` model. The script also accepts a PolarMACE
checkpoint. No `torch.compile` or `jax.jit` is used. The native
Torch result is the reference; the report records module outputs, bitwise
equality, and configurable numerical tolerances. `--compute-force` additionally
compares native Torch forces with `jax.grad` of the TorchAX energy. Stress
gradients are not yet included.

The current local test environment is
`/Users/venkatkapil24/scratch/codex/jax/.venv` with Torch 2.11.0, TorchAX
0.0.13, JAX 0.8.0, graph_longrange 0.4.4, and MACE source commit
`c5c65fe77aa6fdc0e2c587a95abc52734d9ddb8e` (reported as mace-torch
0.3.17). TorchAX 0.0.13 did not import with Torch 2.14.0 in this environment.
Download [the small checkpoint](https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_small.model)
and supply its local path. The checkpoint used for the results below has SHA256
`165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`.

```sh
curl -L --fail \
  https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_small.model \
  -o /tmp/MACE-OFF23_small.model
/Users/venkatkapil24/scratch/codex/jax/.venv/bin/python \
  scripts/verify_mace_torchax.py \
  --checkpoint /tmp/MACE-OFF23_small.model \
  --compute-force \
  --report /tmp/mace-off23-torchax-report.json
```

By default, the test structure is water. Use `--xyz` to change the structure.
For PolarMACE, `--charge`, `--spin`, `--field`, and `--pbc-handling` set the
electrostatic case. The JSON report
includes a checkpoint hash, runtime versions, tensor shapes, individual
comparison results, and the first TorchAX execution error. The process exits
nonzero for a mismatch or runtime error. `--require-bitwise` also makes any
byte difference a failure; numerical agreement alone is the default pass
criterion because the two backends may use different floating point reductions.

On the current versions, MACE-OFF23 small passes energy and force comparison
for water and methanol: all 13 captured tensors agree within tolerance in each
case. The final energies are byte identical; maximum force differences are
`1.1e-14` and `1.3e-15`, respectively, in float64. Five tensors are byte
identical in each case. These are small smoke cases, not a general parity
claim.

### MACE-POLAR-1-M comparison

The same verifier accepts the [released MACE-POLAR-1-M checkpoint](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_polar_1).
The checkpoint used here has SHA256
`fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a`.
Two TorchAX 0.0.13 issues require explicit diagnostic workarounds:

- Indexing a TorchAX `View` raises `AttributeError: 'View' object has no attribute '_elem'`.
- e3nn `Extract` copies into narrowed views, but TorchAX does not update the parent tensor. The verifier replaces this extraction in serialized e3nn `Gate` modules with equivalent functional slices after the native Torch run.

```sh
curl -L --fail \
  https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model \
  -o /tmp/MACE-POLAR-1-M.model
/Users/venkatkapil24/scratch/codex/jax/.venv/bin/python \
  scripts/verify_mace_torchax.py \
  --checkpoint /tmp/MACE-POLAR-1-M.model \
  --compute-force \
  --workaround-view-getitem \
  --workaround-e3nn-extract \
  --report /tmp/polar-force-torchax-report.json
```

With both workarounds, neutral water and water with charge `+1`, spin
multiplicity `2`, and an external field of `[0.01, 0, 0]` pass the forward and
force comparisons. The neutral water comparison covers 54 tensors, including
charges, dipole, electrostatic energy, total energy, and forces. Maximum energy
and force differences are `1.4e-12` and `1.3e-12` in float64. In the charged
field case, the maximum force difference is `6.9e-12`. A periodic water box
also passes energy and force comparison with `--pbc-handling pbc`; its energy
is byte identical and its maximum force difference is `1.8e-15`.

The workarounds are marked in the JSON report. These runs verify TorchAX
execution of the Torch model. The native JAX port has its own comparison below.

## Native eager MACE-POLAR-1-M

`mace_jax.modules.polar_model.PolarMACE` implements the released POLAR-1-M
backbone, spin charge response, local electron energy, and Gaussian multipole
electrostatics in JAX. Its electrostatic modes cover open boundaries, bulk
periodic cells, slabs, molecules in boxes, and mixed periodic batches. Graph
inputs carry total charge, spin multiplicity, external field, and PBC flags.
Energy, forces, and stress are evaluated with JAX autodiff. The port
and its verifier run eagerly; neither calls `jax.jit` or `torch.compile`.

Convert the released Torch checkpoint with the shared float64 environment:

```sh
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /Users/venkatkapil24/scratch/codex/jax/.venv/bin/python \
  -m mace_jax.cli.mace_jax_from_torch \
  --torch-model /tmp/MACE-POLAR-1-M.model \
  --dtype float64 \
  --output /tmp/MACE-POLAR-1-M-jax.msgpack
```

The converter stores the checkpoint's reciprocal cutoff explicitly. Rebuilding
it from the recorded cutoff factor changes the number of Fourier vectors in
this release. It also preserves Torch's repeated spin irreps layout, Bessel
coefficients and prefactor, and fits the full CG weight transform in float64.
Bundles exported before these corrections still load; re-export the checkpoint
to obtain the improved agreement.

Compare native JAX directly against Torch on water:

```sh
PYTHONPATH=. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /Users/venkatkapil24/scratch/codex/jax/.venv/bin/python \
  scripts/verify_polar_backbone.py /tmp/MACE-POLAR-1-M.model \
  --pbc-handling realspace --compute-force \
  --atol 1e-5 --rtol 1e-5
PYTHONPATH=. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /Users/venkatkapil24/scratch/codex/jax/.venv/bin/python \
  scripts/verify_polar_backbone.py /tmp/MACE-POLAR-1-M.model \
  --pbc-handling pbc --compute-force --compute-stress \
  --atol 1e-5 --rtol 1e-5
```

On these water cases with a newly converted float64 bundle, the maximum energy
difference is below `2.7e-9` eV and the maximum force component difference is
below `1.5e-8` eV/Å. The periodic stress differs by at most `2.0e-11` eV/Å³.
These are smoke tests; broader structures and training behavior still need
validation.

Slab water also passes at `1e-5` relative and absolute tolerance. Charged water
(`charge=+1`, spin multiplicity `2`) under an external field of `[0.01, 0, 0]`
passes energy and force comparison; maximum differences are `1.9e-8` eV and
`2.4e-8` eV/Å. The reference Torch forward reads a Fermi level input but uses
zero scalar potential, and the JAX port follows that behavior. The Fourier
evaluator also has a Torch parity test for a mixed bulk and slab batch.

To verify the serialized artifact instead of an in-memory import, add
`--jax-bundle /tmp/MACE-POLAR-1-M-jax.msgpack` to the comparison command.

### Small static evaluation panel

For a laptop-sized check across boundary conditions, the verifier also accepts
`--s22 NAME` (an ASE S22 dimer) or `--structure FILE` (an ASE-readable geometry).
The [MACE-POLAR-1 paper](https://arxiv.org/html/2602.19411) evaluates S22
noncovalent interactions and X23-DMC molecular-crystal lattice energies.
[ASE includes S22](https://ase.gitlab.io/ase/ase/collections.html); useful small
cases are `Water_dimer` (6 atoms), `Formic_acid_dimer` (10 atoms), and
`Benzene-water_complex` (15 atoms). For bulk, the
[X23 structure archive](https://zenodo.org/records/8379098) provides CIFs;
`CO2.cif` has 12 atoms and `NH3.cif` has 16 atoms per cell. These are PBE0+MBD
geometries from an X23 study, so they are suitable for Torch–JAX parity, but
should not be treated as the exact structures used for the paper's X23-DMC
error table without checking the benchmark protocol.

The paper specifies a slab dipole correction but reports no slab benchmark.
For a boundary-condition check, read `NH3.cif` with ASE, call
`atoms.center(vacuum=8.0, axis=2)`, set `atoms.pbc=[True, True, False]`, and
write an `extxyz` file. This derived one-layer structure has 16
atoms and is only a Torch–JAX parity case. Run one structure at a time, with
no JIT or MD. In float64 energy-only runs, the maximum Torch–JAX total-energy
differences were `2.12e-9` eV for the S22 water dimer, `3.19e-8` eV for bulk
ammonia, and `3.19e-8` eV for the derived ammonia slab.

## Package overview

MACE-JAX provides a JAX/Flax training stack for atomistic models, including
streaming HDF5 data loading, gin-driven configuration, distributed training,
and utilities for preprocessing and model conversion. It includes CLI entry
points for training, preprocessing, plotting metrics, and importing Torch MACE
checkpoints. Torch-to-JAX conversion is a core feature: it lets users bring in
pre-trained foundation models and fine-tune them directly in JAX. Fixed-shape
batching is supported for efficient compilation in JAX/XLA, but it is only one
part of a broader end-to-end training workflow. The HDF5 dataset format matches
Torch MACE, so the same HDF5 files can be shared across both implementations.

### Key features

- Streaming HDF5 loader with cached padding caps per split (train/valid/test)
- Fixed-shape batching driven by `n_edge` with derived node/graph padding caps
- Gin-configured model/loss/optimizer stack with CLI overrides
- Training CLI with EMA, SWA, checkpointing (including best-checkpoint), and W&B
- Optimizer schedulers (constant, exponential, piecewise, plateau) integrated via gin
- Torch-to-JAX conversion utilities and corresponding CLI
- Multi-head training/finetuning support with per-head dataset configs
- Distributed training via `jax.distributed` with per-process dataset sharding
- Preprocessing CLI to convert XYZ to streaming HDF5 plus statistics.json output

## Test without installing

```sh
pip install nox
nox
```

## Installation

From github:

```sh
pip install git+https://github.com/ACEsuit/mace-jax
```

Or locally:

```sh
pip install -e .
```

### Optional extras

MACE-JAX defines a few optional dependency groups:

- `jax-cuda13`: JAX CUDA 13 build + cueq CUDA 13 kernels
- `jax-cuda12`: JAX CUDA 12 build + cueq CUDA 12 kernels
- `torch-cuda13`: Torch + cueq CUDA 13 torch kernels
- `torch-cuda12`: Torch + cueq CUDA 12 torch kernels
- `test`: pytest tooling
- `plot`: matplotlib + pandas
- `wandb`: Weights & Biases logging
- `data`: pymatgen (Materials Project download helper)
- `lammps`: LAMMPS Python bindings (MLIAP export helper)

Example:

```sh
pip install -e .[jax-cuda13,torch-cuda13,test]
```

If you need to stay on CUDA 12:

```sh
pip install -e .[jax-cuda12]
```

## Quick start

```sh
mkdir -p data/hdf5
mace-jax-preprocess \
  --train_file data/rmd17_aspirin_train.xyz \
  --valid_file data/rmd17_aspirin_test.xyz \
  --h5_prefix data/hdf5/ \
  --r_max 5.0 \
  --compute_statistics \
  --atomic_numbers "[1, 6, 7, 8]" \
  --E0s "average"

JAX_PLATFORMS=cpu mace-jax-train configs/aspirin_small.gin --print-config
```

This prints the operative gin configuration before training starts. On CPU-only
machines the `JAX_PLATFORMS=cpu` override avoids CUDA plugin warnings.

## Usage

### Command-line tools

After installation, the following convenience commands are available:

- `mace-jax-train`
- `mace-jax-preprocess`
- `mace-jax-predict`
- `mace-jax-train-plot`
- `mace-jax-from-torch`
- `mace-jax-hdf5-benchmark`
- `mace-jax-hdf5-info`
- `mace-create-lammps-model`

#### `mace-jax-train`

Runs training driven by gin-config files. Example:

```sh
mace-jax-train configs/aspirin_small.gin \
  --seed 0 \
  --print-config
```

The operative configuration is saved alongside the run logs.

A repository helper mirrors this example and runs a one-interval smoke test on
the bundled 3BPA dataset:

```sh
python scripts/run_aspirin_smoke_test.py --print-config
```

The script exposes knobs for the dataset paths, seed, device, log directory, and
epoch length, making it convenient for CI or quick local validation without
editing gin files.

Additional convenience flags let you adjust common gin settings directly from the CLI:

- `--torch-checkpoint PATH`: import parameters from a Torch checkpoint (converted on the fly via `mace-jax-from-torch` utilities).
- `--torch-head NAME`: select a specific head from the imported Torch model.
- `--torch-param-dtype {float32,float64}`: override the dtype used for imported parameters.
- `--train-path/--valid-path/--test-path`: point datasets to new files without editing the gin config.
- `--statistics-file PATH`: reuse `statistics.json` from preprocessing (sets atomic numbers,
  E0s, scaling mean/std, and average neighbor count).
- `--r-max VALUE`: synchronise the cutoff used in both dataset construction and model definition.
  For streaming datasets, `--batch-max-edges` (or `n_edge` in gin) sets the edge cap.

##### Equivariance acceleration backend

MACE-JAX exposes a single model-wide equivariance acceleration backend. The
backend is a preferred accelerator: unsupported operations stay on the standard
JAX implementation instead of requiring a second backend config.

```sh
# Enable cueq + conv_fusion when CUDA is detected.
mace-jax-train configs/aspirin_small.gin --enable_cueq

# cueq-only style (similar to mace --only_cueq): force cueq layout/group defaults.
mace-jax-train configs/aspirin_small.gin --only_cueq

# Enable OpenEquivariance for the fused channel-wise convolution path.
mace-jax-train configs/aspirin_small.gin --enable-openeq
```

Notes:
- `--enable-cueq` binds `EquivarianceConfig(backend="cueq")` with cueq optimization defaults.
- `--enable-openeq` binds `EquivarianceConfig(backend="openeq")` for the supported fused channel-wise convolution path.
- `--equivariance-layout {mul_ir,ir_mul}` selects the model-wide representation layout. `--cueq-layout` and `--openeq-layout` are deprecated aliases.
- `--cueq-group {O3,O3_e3nn}` and `--openeq-group O3_e3nn` provide backend group control.
- `--cueq-optimize-all/--no-cueq-optimize-all`, `--cueq-conv-fusion/--no-cueq-conv-fusion`, `--openeq-optimize-all/--no-openeq-optimize-all`, and `--openeq-conv-fusion/--no-openeq-conv-fusion` provide direct CLI control over these settings.
- Legacy serialized `cueq_config`/`openeq_config` dictionaries and Torch `cueq_config` objects are migrated as deprecated compatibility inputs.

##### OpenEquivariance fused convolution

OpenEquivariance is an opt-in CUDA backend for the fused channel-wise tensor
product convolution. It does not accelerate linear, symmetric contraction, or
fully connected tensor-product kernels. Install the pinned packages in this order
(the second build must not use build isolation):

```sh
pip install "openequivariance[jax]==0.6.8"
pip install "openequivariance_extjax==0.6.8" --no-build-isolation
```

Enable it with `--enable-openeq`. The explicit equivalents are
`--openeq-optimize-all`, `--openeq-conv-fusion`,
`--equivariance-layout=mul_ir`, and `--openeq-group=O3_e3nn`. V1 uses CUDA atomic
aggregation, supports float32/float64 and force-training derivatives, and is not
deterministic. ROCm is not qualified. Unsupported OpenEquivariance operation
requests and missing or failed OpenEquivariance kernels raise an error.

For instance, fine-tuning a Torch foundation model against a new dataset can be done with:

```sh
mace-jax-train configs/finetune.gin \
  --torch-checkpoint checkpoints/foundation.pt \
  --torch-head Surface \
  --train-path data/surface.xyz \
  --valid-path None \
  --print-config
```

See the **Foundation models** note below for how to download or select the
pre-trained checkpoint used in this example.

##### Optional logging & averaging

- **Weights & Biases** logging is enabled via CLI flags: `--wandb` launches a
  run with the automatically generated tag/operative gin, and you can supply
  `--wandb-project`, `--wandb-entity`, `--wandb-tag`, etc. No manual gin edits
  are necessary. Metrics for each evaluation split plus per-interval timing are
  streamed to the configured run.

- **Stochastic Weight Averaging (SWA)** is available through `--swa` (with
  `--swa-start`, `--swa-every`, `--swa-min-snapshots`, … mirroring the Torch
  CLI). Once the requested number of snapshots has been accumulated,
  evaluations use the SWA parameters while the raw/EMA weights continue to be
  optimised.

- **Gradient clipping & EMA** can be toggled directly from the CLI using
  `--clip-grad VALUE` and `--ema-decay VALUE`. These map to the same behaviour
  as the Torch `--clip_grad`/`--ema` flags and remove the need for explicit gin
  bindings.
- **Optimizer & scheduler** settings can be specified via `--optimizer`
  (`adam`, `adamw`, `amsgrad`, `sgd`, `schedulefree`), `--lr`, `--weight-decay`,
  `--schedule-free-b1`, `--schedule-free-weight-lr-power`, `--scheduler`
  (`constant`, `exponential`, `piecewise_constant`, `plateau`), and
  `--lr_scheduler_gamma`.
  These bind directly into the gin optimizer helper, mirroring the Torch CLI.
- **Foundation models** can be pulled directly via the same interface as Torch
  MACE. Use `--foundation_model small` (or `medium`, `large`, `small_off`, …) to
  download and initialize from the released checkpoints, or pass a custom path.
  The CLI adjusts the cutoff, learning-rate defaults, and multihead finetuning
  knobs just like `mace run_train`.
- **Multihead finetuning** mirrors the Torch defaults via `--multiheads_finetuning`
  (optionally with `--force_mh_ft_lr`): learning rate/EMA defaults are adjusted
  automatically when combined with `--foundation_model`, so migrating scripts
  can keep the same behaviour.
- **Distributed training** can run multi-process on a single node or across hosts.
  For single-node multi-GPU runs, the default `--launcher auto` will start one process
  per visible GPU and enable `jax.distributed` automatically when more than one GPU is
  visible. Use `CUDA_VISIBLE_DEVICES` to restrict which GPUs are used, or pass
  `--launcher none` to force single-process mode.

  ```sh
  # Single node, two GPUs (auto spawns 2 processes).
  CUDA_VISIBLE_DEVICES=0,1 \
  mace-jax-train configs/aspirin_small.gin \
    --device cuda \
    --launcher auto
  ```

  For multi-host launches you still need to provide `--distributed` plus the
  process topology (`--process-count`, `--process-index`, and optional
  `--coordinator-address/--coordinator-port`). When distributed is enabled the CLI
  initialises `jax.distributed`, shards the training/validation/test datasets per
  process deterministically, and only writes logs/checkpoints from rank 0.
  Environment variables such as `JAX_PROCESS_COUNT`, `JAX_PROCESS_INDEX` or
  Slurm launch variables can also be used; they override the CLI defaults automatically.
A typical 2-host launch (one process per host) looks like:

  ```sh
  JAX_PROCESS_COUNT=2 \
  JAX_PROCESS_INDEX=${SLURM_PROCID} \
  mace-jax-train configs/aspirin_small.gin \
    --device cuda \
    --distributed \
    --coordinator-address host0 \
    --coordinator-port 12345
  ```

#### `mace-jax-predict`

Runs inference on XYZ or HDF5 datasets and writes `.npz` outputs (one per input).
For streaming HDF5 inputs, predictions are ordered by the stable `graph_id` and
written as NumPy arrays:

```sh
# XYZ inference.
mace-jax-predict checkpoints/model.ckpt data/structures.xyz

# HDF5 inference (uses cached streaming stats if present).
mace-jax-predict checkpoints/model.ckpt data/hdf5/train_0.h5 \
  --batch-max-edges 6000 \
  --num-workers 8
```

Notes:
- `--batch-max-edges` is required for streaming HDF5 predictions unless cached
  streaming stats are present next to the file.
- `--batch-max-nodes` is optional for predictions if you want to override the
  cached node cap.
- XYZ predictions use the same fixed-shape batching path as HDF5 (pmap + padding);
  you can pass `--batch-max-edges` to force the same caps used during training.

#### `mace-jax-preprocess`

Converts one or more XYZ files into MACE-style streaming HDF5 files and (optionally)
computes dataset statistics. The outputs are written to `<h5_prefix>/train`,
`<h5_prefix>/val`, and `<h5_prefix>/test`, plus `statistics.json` when requested.

```sh
mace-jax-preprocess \
  --train_file data/train.xyz \
  --valid_file data/valid.xyz \
  --h5_prefix data/hdf5/ \
  --r_max 5.0 \
  --compute_statistics \
  --E0s "average"
```

Pass the resulting `statistics.json` to training via `mace-jax-train --statistics-file`
to reuse the computed scaling and average neighbor counts.

Key preprocessing arguments (most commonly used):
- `--train_file/--valid_file/--test_file`: input XYZ files. If `--valid_file` is omitted,
  `--valid_fraction` is used to split the training file.
- `--h5_prefix`: output root for HDF5 shards (`<prefix>/train`, `<prefix>/val`, `<prefix>/test`).
- `--num_process`: number of worker processes **and** output shards per split
  (e.g., `train_0.h5`, `train_1.h5`). Higher values increase parallel writes
  and create more shards for streaming; keep it in line with disk throughput.
- `--r_max`: neighbor cutoff used for graph construction and statistics.
- `--atomic_numbers` / `--E0s`: explicitly set atomic numbers or atomic energies; otherwise
  they are inferred from the dataset (energies default to `average` when stats are computed).
- `--compute_statistics`: write `statistics.json` (scaling + neighbor stats) for reuse in training.
- `--shuffle`: randomize configs before splitting/sharding (affects train/valid ordering).

#### `mace-jax-train-plot`

Produces loss/metric curves from `.metrics` logs generated during training (JSON-lines logs in the run directory):

```sh
mace-jax-train-plot --path results --keys rmse_e_per_atom,rmse_f --output-format pdf
```

The command accepts either a directory (all `.metrics` files are processed) or a single metrics file. It plots
`valid` and `train` loss curves (with per-head suffixes like `valid:Surface` when present), plus the
extra metric keys requested via `--keys`.

#### `mace-jax-hdf5-info`

Prints a quick summary of MACE-style HDF5 datasets (batch counts, graph counts,
property keys, etc.). If a streaming stats cache is present, it also reports the
cached padding caps:

```sh
mace-jax-hdf5-info data/hdf5/train_*.h5
```

#### `mace-jax-hdf5-benchmark`

Measures HDF5 read throughput for MACE-style datasets (sequential and optional
shuffled access patterns):

```sh
mace-jax-hdf5-benchmark data/hdf5/train_0.h5 --count 1000 --repeat 3 --shuffle
```

#### `mace-create-lammps-model`

Converts a Torch MACE checkpoint to the JAX MLIAP format:

```sh
mace-create-lammps-model checkpoints/model.pt --dtype float32 --output exported_model.pkl
```

Add `--head NAME` when exporting a multi-head model.

#### `mace-jax-from-torch`

Performs Torch→JAX parameter conversion. It can also
download and import pre-trained foundation models when `--torch-model` is not
provided.

If `--output` is omitted the converted parameters are written to `<checkpoint>-jax.npz`.

You can try this with one of the released foundation models (downloaded automatically):

```sh
mace-jax-from-torch --foundation mp --model-name small
```

All commands can be invoked via `python -m mace_jax.<module>` if preferred.

### Expert usage (internal APIs)

If you want to call the CLI helpers directly from Python (for notebooks or
custom pipelines), the internal entrypoints are exposed in the CLI modules.
These helpers are not part of the stable public API (they use leading
underscores), so prefer the CLI for long-term scripts.

#### Predictions (`_predict_xyz` / `_predict_hdf5`)

Both helpers live in `mace_jax.cli.mace_jax_predict` and return
`(graph_ids, outputs)`. For HDF5 inputs the outputs are **sorted by graph_id**
within the local process, so they can be merged across processes later if
needed.

```py
from pathlib import Path

from mace_jax.cli import mace_jax_predict
from mace_jax.tools import bundle as bundle_tools

bundle = bundle_tools.load_model_bundle("checkpoints/model-jax.npz", "")
head_name, head_to_index = mace_jax_predict._resolve_head_mapping(
    bundle.config, head=None
)
predictor = mace_jax_predict._build_predictor(
    bundle.module, compute_forces=True, compute_stress=True
)

# XYZ prediction (equivalent to CLI on .xyz input).
graph_ids, outputs = mace_jax_predict._predict_xyz(
    Path("data/structures.xyz"),
    predictor=predictor,
    params=bundle.params,
    model_config=bundle.config,
    head_name=head_name,
    head_to_index=head_to_index,
    batch_max_edges=None,
    batch_max_nodes=None,
    batch_max_graphs=None,
    prefetch_batches=2,
    progress_bar=True,
)

# Streaming HDF5 prediction (requires batch_max_edges).
graph_ids, outputs = mace_jax_predict._predict_hdf5(
    Path("data/train_0.h5"),
    predictor=predictor,
    params=bundle.params,
    model_config=bundle.config,
    head_name=head_name,
    head_to_index=head_to_index,
    batch_max_edges=6000,
    batch_max_nodes=None,
    prefetch_batches=2,
    num_workers=8,
    progress_bar=True,
)
```

#### Training (`run_training`)

The CLI entrypoint in `mace_jax.cli.mace_jax_train` is also reusable in-process
as long as gin is configured first:

```py
import gin

from mace_jax.cli import mace_jax_train

gin.parse_config_files_and_bindings(
    ["configs/aspirin_small.gin"],
    bindings=[],
)
mace_jax_train.run_training()
```

For even finer control you can call the same building blocks used by the CLI:
`mace_jax.tools.gin_datasets.datasets`, `mace_jax.tools.gin_model.model`,
`mace_jax.tools.gin_functions.loss/optimizer/train`, and the helper functions
inside `mace_jax.cli.mace_jax_train` (such as `_build_predictor`).

### Streaming HDF5 loader (mace-jax specific)

Fixed-shape batches let JAX/XLA compile once and reuse the same executable, so
the streaming loader is designed to keep batch shapes stable across epochs.
MACE‑JAX compiles the model with **fixed batch shapes**, so it cannot accept
variable‑sized batches on the fly. To make this efficient, the streaming loader
runs a stats scan to derive fixed padding caps and then builds batches on the fly
within those caps. This avoids repeated recompilations and keeps throughput stable
on large HDF5 datasets.

The only sizing knob you must provide is `n_edge` (also exposed as `--batch-max-edges`
or `n_edge` in gin). `n_node` is **not** supported for streaming datasets; it is
derived automatically from the streaming stats scan (with an optional percentile
cap for tighter node padding).

How `n_edge` is used:

- **Edge cap / graph feasibility:** Each graph’s edge count is checked against the
  configured `n_edge` limit. If a graph exceeds the limit, the loader raises the cap
  and logs a warning so all graphs fit.
- **Batch packing (knapsack‑like):** The stats pass uses greedy packing to estimate
  typical batch sizes under the `n_edge` cap.
- **Derived node/graph caps:** The streaming stats compute node/graph caps from the
  edge-limited batches. `n_node` defaults to the 90th percentile of batch node
  totals, but is never lower than the largest single-graph node count. `n_graph`
  uses the maximum observed graph count per batch. These become the fixed
  `n_node`/`n_graph` pads used for JAX compilation.
- **On‑the‑fly batching:** During training/eval, worker processes read HDF5 entries
  directly, pack batches until a cap is hit, pad, and emit the batch. There are
  no separate reader processes. Shuffling is optional and, when enabled, uses a
  deterministic per-epoch shuffle of global graph indices.
- **Graph IDs:** Each graph is tagged with a stable global `graph_id` so streaming
  prediction outputs can be reordered to the original HDF5 order when needed.
- **Multi-process sharding:** For distributed runs, global graph indices are
  sharded round-robin by `process_index` across processes; each process reads only
  its own slice of indices but can span all HDF5 files.
- **Persisted caps:** The derived `n_nodes/n_edges/n_graphs` caps are stored in the
  streaming stats. Train/valid/test each get their own stats so loaders can be reused
  independently.

In short, `n_edge` controls memory and compilation shape: it is the fixed edge budget
per batch. The loader packs graphs to stay under this budget, computes the node/graph
padding needed, and then reuses these fixed shapes for every epoch.

Streaming datasets accept a single HDF5 file, a directory of `.h5/.hdf5` files, or a
glob pattern. The loader expands all matching files and derives padding caps across
the combined dataset.

Streaming loader knobs (CLI or gin):
- `--batch-max-edges` / `n_edge`: edge cap for fixed-shape batches.
- `--suffle`: enable/disable per-epoch shuffling for training (default: true).
  Validation/test loaders never shuffle.
- `--batch-node-precentile`: percentile for node padding cap (default: 90);
  caps never drop below the largest single graph.
- `--stream-train-max-batches`: optional cap on batches per epoch.

### Configuration

Links to the files containing the functions configured by the gin config file.

- [`flags`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`logs`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`datasets`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_datasets.py)
- [`model`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_model.py) and [here](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/modules/models.py)
- [`loss`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/modules/loss.py)
    if `loss.energy_weight`, `loss.forces_weight` or `loss.stress_weight` is nonzero the loss will be applied
- [`optimizer`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)
- [`train`](https://github.com/ACEsuit/mace-jax/blob/main/mace_jax/tools/gin_functions.py)

## Contributions

We are happy to accept pull requests under an [MIT license](https://choosealicense.com/licenses/mit/). Please copy/paste the license text as a comment into your pull request.

## References

If you use this code, please cite our papers:
```text
@misc{Batatia2022MACE,
  title = {MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2206.07697},
  eprint = {2206.07697},
  eprinttype = {arxiv},
  doi = {10.48550/ARXIV.2206.07697},
  archiveprefix = {arXiv}
}
@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have any questions, please contact us at ilyes.batatia@ens-paris-saclay.fr, geiger.mario@gmail.com, or philipp.benner@gmail.com.

## License

MACE is published and distributed under the [MIT](MIT.md).
