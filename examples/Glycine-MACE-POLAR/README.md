# Neutral glycine in water: MACE-POLAR and MACE-POLAR/MM

`initial-neutral.xyz` contains a 10-atom neutral glycine and 52 complete
waters in an 11.76 Å cubic periodic cell. Glycine starts from the
[PubChem CID 750 3D conformer](https://pubchem.ncbi.nlm.nih.gov/compound/750).
The carboxyl group is rotated toward the amine to construct the IIp-like
reactive conformer used for direct intramolecular proton transfer by
[Leung and Rempe](https://doi.org/10.1063/1.1885445). The 52 waters are
selected from the published equilibrated HCl water snapshot described in
`structure.json`; hydronium and chloride are excluded. This combined cell is
newly constructed and is **not** a published equilibrated aqueous glycine
configuration.

On Boltzmann, `run-on-boltzmann.sh` first performs fixed-cell ASE BFGS using
MACE-POLAR-1-M until the largest projected force is below 0.1 eV/Å. The
glycine acid O-H length is constrained during optimization to keep the
starting structure neutral. The constraint is removed for both 330 K MD runs.
Both trajectories start from `bfgs/optimized-neutral.xyz` with identical
initial velocities and seed. They use 0.5 fs BAOAB Langevin integration,
1 ps thermostat relaxation time, and 200,000 steps = 100 ps. The second
trajectory has glycine in the POLAR region and the 52 waters in flexible
AMBER14 TIP3P. Both run on **CPU** in detached Boltzmann processes.

Each MD output directory writes `progress.json`, `run.log`, atomic
`checkpoint.npz`, and numbered trajectory chunks under `chunks/`. Re-running
the script resumes at the last completed checkpoint. `final.xyz` appears
after 100 ps. If glycine transfers its proton to an MM water, the fixed
POLAR/MM partition cannot describe that reaction; the all-POLAR trajectory
can describe either direct or water-mediated transfer.

The POLAR/MM electrostatics are the same subtractive construction as in the
prior prototype: the model's periodic long-range term sees all MM TIP3P point
charges, and MM-MM interactions use Ewald. Flexible TIP3P bonded and O-O LJ
terms plus glycine-water cross LJ terms complete the short-range energy.
The cross LJ parameters are mapped to corresponding atom types in
[AMBER ff14SB](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/data/amber14/protein.ff14SB.xml).
This mapping is an approximation for standalone glycine, which is not a
standard protein residue.
