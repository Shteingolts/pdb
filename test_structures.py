import os
import time
from multiprocessing import Pool
import importlib
import process


DESKTOP = '/mnt/c/users/serge/desktop'
OUT_DIR = os.path.join(DESKTOP, 'test_structures')
NUM_ATOMS = 10
MOL_WEIGHT = 150
DISTANCE_THRESHOLD = 1.7


with open(os.path.join(DESKTOP, 'test_structs.txt'), 'r') as f:
    test_structures = f.read().split()
test_structures = [structure_id.lower() for structure_id in test_structures]
test_structures.remove('2vrv') # structure 2vrv is obsolete for some reason


def extract_ligands(
    name,
    num_atoms=NUM_ATOMS,
    mol_weight=MOL_WEIGHT,
    distance_threshold=DISTANCE_THRESHOLD,
    test_structures=test_structures,
    debug=False,
    ) -> tuple[str, float]:
    
    """
    The same extract as in process, but for multiprocessing.
    
    """
    
    start_job = time.perf_counter()

    original_structure = process.get_structure(name, process.PDB_PATH)
    hetero_residues = process.get_hetero_residues(original_structure)
    if len(hetero_residues) == 0:
        print(f'Structure {name} has 0 hetero residues!')
        return

    structure = process.StructureGraph(hetero_residues)
    structure.create_graph(distance_threshold=distance_threshold)
    clusters = structure.get_clusters()
    residues = list()
    for cluster in clusters:
        residues.append(process.combine(cluster))
    
    ligands = process.filter_ligands(residues, num_atoms=num_atoms, mol_weight=mol_weight, rmsd=2.0)
    if len(ligands) == 0:
        print(f'Structure {name} has 0 zero ligands after filtering with current parameters:\n  NUM_ATOMS: {NUM_ATOMS}, MOL_WEIGHT: {MOL_WEIGHT}, RMSD: {2.0}')
        return
    
    ligand_structures = [process.resi_to_struct(ligand, original_structure=original_structure) for ligand in ligands]
    process.save_ligand_to_file(ligand_structures, OUT_DIR, original_structure)
        # print(f'Structure {original_structure.id} raised an error during ligand filtering!')

    end_job = time.perf_counter()
    job_duration = end_job - start_job

    if debug == True:
        print(f'Structure {name} completed in {job_duration:.2f}s')

    return name, job_duration

start_time = time.perf_counter()

with Pool() as pool:
    pool.map(extract_ligands, test_structures)

end_time = time.perf_counter()
total_duration = end_time - start_time
print(f"{len(test_structures)} structures took {total_duration:.2f}s total")
