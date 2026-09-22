# Hydrogen maleate in water

The starting cell contains one hydrogen maleate anion (`C4H3O4-`) and 52 waters
in an 11.76 Å periodic cube. The solute is based on PubChem CID 11966254 and
starts in the internally hydrogen-bonded conformation. Initial donor O-H and
acceptor O...H distances are 0.981 and 1.510 Å. The QM charge is -1 and spin
multiplicity is 1; periodic electrostatics use a uniform neutralizing background.

The detached CPU pipeline performs fixed-cell MACE-POLAR BFGS to 0.1 eV/Å while
holding the initial O-H length, then removes the constraint for two 330 K,
0.5 fs, 100 ps trajectories from identical coordinates and velocities. One is
all MACE-POLAR. The other uses MACE-POLAR for the anion and first-shell waters
(water O within 3.8 Å of a solute O), with remaining water described by flexible
AMBER14 TIP3P. A flat-bottom restraint keeps fixed QM waters within 4.2 Å.

Saved frames record the proton distance to both carboxylates and
`proton_coordinate_angstrom = right - left`; a sign change marks transfer.
Both jobs are restartable from atomic checkpoints.
