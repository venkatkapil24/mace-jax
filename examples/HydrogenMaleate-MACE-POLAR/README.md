# Hydrogen maleate in water

The starting cell contains one hydrogen maleate anion (`C4H3O4-`) and 52 waters
in an 11.76 Å periodic cube. The solute is based on PubChem CID 11966254 and
starts in the internally hydrogen-bonded conformation. Initial donor O-H and
acceptor O...H distances are 0.981 and 1.510 Å. The QM charge is -1 and spin
multiplicity is 1; periodic electrostatics use a uniform neutralizing background.

The detached CPU pipeline performs fixed-cell MACE-POLAR BFGS to 0.1 eV/Å while
holding the initial O-H length, then removes the constraint for 330 K, 0.5 fs,
100 ps trajectories from identical coordinates and velocities. The reference is
full MACE-POLAR. Electrostatic and mechanical embedding controls use MACE-POLAR
for the anion and first-shell waters (water O within 3.8 Å of a solute O), with
remaining water described by flexible AMBER14 TIP3P. A flat-bottom restraint
keeps the fixed QM waters within 4.2 Å.

In electrostatic embedding, the MM point charges enter MACE-POLAR's long-range
field. In OpenMM-style mechanical embedding, MACE-POLAR receives no MM field;
fixed-charge QM-MM Ewald electrostatics and QM-MM Lennard-Jones interactions are
added classically. QM water uses TIP3P charges. Hydrogen maleate uses the fixed
charges published in Supplementary Section S2 of Vener et al., *Int. J. Mol.
Sci.* **23**, 6302 (2022), DOI 10.3390/ijms23116302.

Saved frames record the proton distance to both carboxylates and
`proton_coordinate_angstrom = right - left`; a sign change marks transfer.
Both jobs are restartable from atomic checkpoints.
