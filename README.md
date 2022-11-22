# pdb
A simple script for the extraction of ligands from the pdb structure files.
Implements rudimentary cli, can fetch the desired structures from the local copy on your machine. 

**IMPORTANT, PLEASE READ!**

Current version of the script assumes the local (partial) copy of the pdb,
saved in the pdb "tree" fashion, i.e., the root directory containing 
multiple other directories, each named with two middle characters of the 
structure IDs. Structures themselves should be stores with the .ent extension
and a pdb prefix, e.g. a file for the structure 7VYP should be named `pdb7vyp.ent`.

As an example, the structure with the PDB ID `7VYP` should be located in 
`.../pdb_dir/vy/pdb7VYP.ent`.

However, since the `get_structure()` function returns a `Bio.PDB.Structure.Structure`
object, it would be trivial to implement a different function with the API provided by
the BioPython library to fetch a requested structure on demand from the PDB.

# process.py
process.py file contains all the main logic associated with the extraction of ligands from the structure files.

# test_structures.py
Involves simple multiprocessing, works way way faster
