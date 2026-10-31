# MACE-POLAR examples

Examples using the released MACE-POLAR-1-M model and the native JAX port.

The [HCl in water example](../HCl-MACE-POLAR/README.md) contains a fixed-cell
geometry optimization and short JIT-compiled NVE trajectory. The
[POLAR/MM example](../HCl-MLMM/README.md) uses MACE-POLAR for the first shell
and AMBER TIP3P for the remaining waters.

`minimal_forward.py` runs a single energy/forces/stress evaluation with float64,
cuEq, generalized PPPM (1.2 Å spacing, order 4), and fixed-weight inference.
Neighbor rebuilding and graph compaction are inside the same JIT calculation.
It takes a float64 cuEq-converted bundle and a single-system MACE input dictionary
saved with `np.savez`; the input preserves the charge, spin and external field.
Set `MODEL` and `INPUT` at the top of the file, then run:

```sh
python examples/MACE-POLAR/minimal_forward.py
```

The example is for an orthorhombic periodic box with sides larger than twice
the model cutoff. Its 120-neighbor and 93,000-edge capacities match the
999-atom water benchmark. It checks for overflow rather than truncating silently.
