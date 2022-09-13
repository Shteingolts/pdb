import os
import time
from multiprocessing import Pool
import process

DESKTOP = '/mnt/c/users/serge/desktop'
OUT_DIR = os.path.join(DESKTOP, 'test_structures')

NUM_ATOMS = 15
MOL_WEIGHT = 200
DISTANCE_THRESHOLD = 1.7

with open(os.path.join(DESKTOP, 'test_structs.txt'), 'r') as f:
    test_structures = f.read().split()
test_structures = [sid.lower() for sid in test_structures]
test_structures.remove('2vrv')




def extract_ligands(
    name,
    num_atoms=NUM_ATOMS,
    mol_weight=MOL_WEIGHT,
    distance_threshold=DISTANCE_THRESHOLD) -> tuple[str, float]:
    """
    The same as extract in process, but for multiprocessing
    """
    start_job = time.perf_counter()

    original_structure = process.get_structure(name, process.PDB_PATH)
    hetero_residues = process.get_hetero_residues(original_structure)
    structure = process.StructureGraph(hetero_residues)
    structure.create_graph(distance_threshold=distance_threshold)
    clusters = structure.get_clusters()
    residues = list()
    for cluster in clusters:
        residues.append(process.combine(cluster))
    ligands = process.filter_ligands(residues, num_atoms=num_atoms, mol_weight=mol_weight, rmsd=2.0)
    ligand_structures = [process.resi_to_struct(ligand, original_structure=original_structure) for ligand in ligands]
    process.save_ligand_to_file(ligand_structures, OUT_DIR, original_structure, debug=True)

    end_job = time.perf_counter()
    job_duration = end_job - start_job
    print(f'Structure {name} completed in {job_duration:.2f}s')

    return name, job_duration

start_time = time.perf_counter()

with Pool() as pool:
    pool.map(extract_ligands, test_structures)

end_time = time.perf_counter()
total_duration = end_time - start_time
print(f"{len(test_structures)} structures took {total_duration:.2f}s total")
