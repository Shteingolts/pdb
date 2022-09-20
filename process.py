from collections import namedtuple
import os
import copy
import platform
import warnings
import Bio.PDB
from Bio.PDB.PDBIO import PDBIO
from scipy.spatial import distance
import numpy as np
import pandas as pd
import pocket_search as ps

ATOM_TABLE = pd.read_csv("atoms.csv")

# got tired of multiple warnings
from Bio.PDB.Atom import PDBConstructionWarning

warnings.simplefilter("ignore", PDBConstructionWarning)

if platform.system() == "Windows":
    print("Running on Windows.\n")
    PDB_PATH = os.path.join("C:", "Users", "serge", "Desktop", "pdb_download", "pdb")
    OUT_PATH = os.path.join("C:", "Users", "serge", "Desktop")
elif platform.system() == "Linux":
    print("Running on Linux.\n")
    PDB_PATH = os.path.join("/mnt", "c", "users", "serge", "desktop", "pdb_download", "pdb")
    OUT_PATH = os.path.join("/mnt", "c", "users", "serge", "desktop")


def get_structure(structure_id: str, pdb_path: str, *, debug=False) -> Bio.PDB.Structure.Structure:
    """
    Takes the structure ID and the path to the PDB directory as inputs.
    Returns the `Bio.PDB.Structure.Structure` object.

    Arguments:

    - `structure_ID`: str - structure ID
    - `pdb_path`: str - path to the PDB directory
    - `debug`: bool - prints debug information to the console.
    """
    struct_path = os.path.join(pdb_path, structure_id[1:-1], "pdb" + structure_id + ".ent")
    try:
        structure = Bio.PDB.PDBParser(QUIET=True).get_structure(structure_id, struct_path)
    except Exception as exc:
        raise Exception(f"Could not find structure {structure_id} in {struct_path}") from exc
    if debug:
        print(f"Structure {structure.get_id()} was loaded from {os.path.abspath(struct_path)}")
    return structure


def get_sequence(structure: Bio.PDB.Structure.Structure) -> list:
    """
    Takes the Bio.PDB.Structure.Structure object as input.
    Returns the sequence string of the protein as list of amino acid codes.

    Arguments:

    - `structure`: Bio.PDB.Structure.Structure object
    """

    sequence: list = []
    for model in structure.get_list():
        for chain in model.get_list():
            for residue in chain.get_list():
                if residue.get_id()[0] == " ":
                    sequence.append(residue.resname)

    return sequence


def describe_structure_residues(
    structure: Bio.PDB.Structure.Structure, *, log=False
) -> None:
    """
    Takes the Bio.PDB.Structure.Structure object as input.
    Extracts the contents of a PDB file, prints the number
    and the type of non-polymer residues to the console.

    Arguments:

    - `structure`: Bio.PDB.Structure.Structure object
    - `log`: bool - if True, function returns the structure describtion
                as a string, otherwise prints to the screen.
    """
    num_waters: int = 0
    num_amino_acids: int = 0
    num_hetero_residues: int = 0
    list_hetero_residues: list = []
    for residue in structure.get_residues():
        if residue.get_id()[0] == "W":
            num_waters += 1
        elif residue.get_id()[0] == " ":
            num_amino_acids += 1
        elif "H_" in residue.get_id()[0]:
            num_hetero_residues += 1
            list_hetero_residues.append(residue)

    structure_description = (
        f"There are {num_waters} water molecules in this structure.\n"
        f"There are {num_amino_acids} standart amino and nucleic acid residues.\n"
        f"There are {num_hetero_residues} hetero residues in this structure:\n"
    )
    if num_hetero_residues > 0:
        for residue in list_hetero_residues:
            structure_description = (
                structure_description
                + f"\n  Residue {residue.get_id()} with {len(residue.get_list())} atoms.\n"
            )
            for atom in residue.get_list():
                structure_description = (
                    structure_description
                    + f"    {atom.get_name()}, {atom.get_coord()}\n"
                )

    if log == True:
        return structure_description
    else:
        print(structure_description)


def get_hetero_residues(structure: Bio.PDB.Structure.Structure, *, debug=False) -> list:
    """
    Takes the Bio.PDB.Structure.Structure object as input.
    Extracts the heteroaromic residues (H_*) from the structure.

    Returns the `list` of `Bio.PDB.Residue.Residue` objects.

    Arguments:

    - `structure`: `Bio.PDB.Structure.Structure` object
    - `debug`: bool - if True, prints the number of atoms and their labels for each ligand.
    """
    h_residues: list = []
    num_h_residues: int = 0
    # for the rare cases of there being more than 1 model, only taking the 1st one
    for chain in structure.get_list()[0].get_list():
        for residue in chain:
            if residue.get_id()[0].startswith("H"):
                if not part_of_protein(residue, structure):
                    num_h_residues += 1
                    h_residues.append(residue)

    if debug is True:
        print(f"Found {num_h_residues} hetero residues in structure {structure.get_id()} .")
        for ligand in h_residues:
            print(f"Residue {ligand.get_id()} has {len(ligand.get_list())} atoms.")
    if len(h_residues) == 0:
        print(f"No hetero residues were found in structure {structure.get_id()}.")

    return h_residues


def count_data_files(pdb_path: str, *, debug=False) -> int:
    """
    Takes the path to the PDB directory as input.
    Counts the number of data files in a directory.
    Returns `int` - number of data files.

    Arguments:

    - `pdb_path`: str - path to the PDB directory
    - `debug`: bool - if True, prints the number of data files.
    """
    file_count = 0
    directory_count = 0
    for directory in os.listdir(pdb_path):
        directory_count += 1
        for data_file in os.listdir(os.path.join(pdb_path, directory)):
            if data_file.endswith(".ent"):
                file_count += 1
    if debug == True:
        print(
            f"Found {file_count} data files in {pdb_path}, {directory_count} data directories."
        )
    return file_count


def resi_to_struct(
    residue: Bio.PDB.Residue.Residue,
    original_structure: Bio.PDB.Structure.Structure = None,
    struct_id="ASBL",
    model_id=1,
    chain_id="A",
    seg_id="ARTF",
) -> Bio.PDB.Structure.Structure:
    """
    Takes the `Bio.PDB.Residue.Residue` object as input.

    If the original structure is not provided, new residues is created from scratch.

    If the original structure is provided, the new residue is created using the information
    from the original structure.

    Returns the `Bio.PDB.Structure.Structure` object.

    Arguments:

    - `residue`: `Bio.PDB.Residue.Residue` object

    Optional arguments:

    - `original_structure`: `Bio.PDB.Structure.Structure` object -

                            optional argument for inheritance
                            of the following attributes:

                          - `struct_id`,
                          - `model_id`,
                          - `chain_id`,
                          - `segid`
    - `struct_id`: str - PDB structure ID
    - `model_id`: int - PDB model ID
    - `chain_id`: str - PDB chain ID
    - `seg_id`: str - PDB segment ID
    """
    if original_structure == None:
        struct_id = residue.get_id()[0]
        model_id = 1
        chain_id = "A"
        seg_id = "ARTF"
    else:
        struct_id = residue.get_id()[0]
        model_id = original_structure.get_list()[0].get_id()
        chain_id = original_structure.get_list()[0].get_list()[0].get_id()
        seg_id = (
            original_structure.get_list()[0].get_list()[0].get_list()[0].get_segid()
        )
    assembled_structure = Bio.PDB.StructureBuilder.StructureBuilder()
    assembled_structure.init_structure(struct_id)
    assembled_structure.init_model(model_id)
    assembled_structure.init_chain(chain_id)
    assembled_structure.init_seg(seg_id)
    assembled_structure.init_residue(
        residue.get_resname(), "H", residue.get_id()[1], residue.get_id()[2]
    )
    for atom in residue.get_list():
        assembled_structure.init_atom(
            atom.get_name(),
            atom.get_coord(),
            atom.get_bfactor(),
            atom.get_occupancy(),
            atom.get_altloc(),
            atom.get_fullname(),
            atom.element,
        )

    return assembled_structure.get_structure()


def get_random_structures(pdb_path: str, num_structs: int, *, debug=False) -> list:
    """
    Takes the path to the PDB directory as input.
    Return a `list` of `num_structs` random structures from the PDB directory.
    If `num_structs` equals to 1, returns a `Bio.PDB.Structure.Structure` object.

    Arguments:

    - `pdb_path`: str - path to the PDB directory
    - `num_structs`: number of structures to choose
    - `debug`: print debug information
    """
    file_count = count_data_files(pdb_path, debug=debug)
    if file_count < num_structs:
        print(f"Not enough data files in {pdb_path}")
        return None
    else:
        random_directories = np.random.choice(os.listdir(pdb_path), num_structs)
        random_structures = []

        while len(random_structures) < num_structs:
            random_directory = np.random.choice(random_directories, size=1)[0]
            random_structure = np.random.choice(
                os.listdir(os.path.join(pdb_path, random_directory)), size=1
            )[0]
            structure_file: str = os.path.join(
                pdb_path, random_directory, random_structure
            )
            structure_ID: str = os.path.basename(structure_file).split(".")[0][3:]
            parser = Bio.PDB.PDBParser(QUIET=True)
            structure = parser.get_structure(structure_ID, structure_file)
            random_structures.append(structure)

        if debug == True:
            print(
                f"Randomly selected {len(random_structures)} structures from {pdb_path}:"
            )
            if len(random_structures) < 10:
                for structure in random_structures:
                    print(f"  {structure.get_id()}")
            if len(random_structures) > 10:
                for structure in random_structures[:10]:
                    print(structure.get_id())
                print("...")
                print(random_structures[-1].get_id())

        if len(random_structures) == 1:
            return random_structures[0]
        else:
            return random_structures


def distance_between_structures(
    structure1: Bio.PDB.Structure.Structure,
    structure2: Bio.PDB.Structure.Structure,
    debug=False,
) -> float:
    """
    Takes two `Bio.PDB.Structure.Structure` objects as input.
    Computes the distance between them.
    Distance is defined as the minimum distance between two
    atoms of the two structures.

    Arguments:

    - `structure1`: Bio.PDB.Structure.Structure object.
    - `structure2`: Bio.PDB.Structure.Structure object.
    - `debug`: bool -  prints the names of the two structures and the distance between them to the console.
    """
    atoms1: list[Bio.PDB.Atom.Atom] = structure1.get_residues()[0].get_list()
    atoms2: list[Bio.PDB.Atom.Atom] = structure2.get_residues()[0].get_list()
    min_distance: float = 100.0

    for atom_m in atoms1:
        for atom_n in atoms2:
            if atom_m - atom_n < min_distance:
                min_distance = atom_m - atom_n

    if debug == True:
        print(
            f"Distance between {structure1.get_id()} and {structure2.get_id()} is {min_distance}"
        )
    return min_distance


def distance_between_residues(
    residue1: Bio.PDB.Residue.Residue, residue2: Bio.PDB.Residue.Residue, debug=False
) -> float:
    """
    Takes two `Bio.PDB.Residue.Residue` objects as input.
    Computes the distance between them.

    Distance is defined as the minimum distance between two
    atoms of the two residues.

    Arguments:

    - `residue1`: `Bio.PDB.Residue.Residue` object.
    - `residue2`: `Bio.PDB.Residue.Residue` object.
    - `debug`: bool -  prints the names of the two residues and the distance between them to the console.
    """

    atoms1: list[Bio.PDB.Atom.Atom] = residue1.get_list()
    atoms2: list[Bio.PDB.Atom.Atom] = residue2.get_list()
    min_distance: float = 100.0

    for atom_m in atoms1:
        for atom_n in atoms2:
            if atom_m - atom_n < min_distance:
                min_distance = atom_m - atom_n

    if debug == True:
        print(
            f"Distance between {residue1.get_resname()} and {residue2.get_resname()} is {min_distance}"
        )
    return min_distance


def distance_between_entities(entity1, entity2, debug=False) -> float:
    """
    Takes the two entities as input, either `Structure` or `Residue` objects.
    Computes distance between two entities.

    Distance is defined as the minimum distance between two
    atoms of the two residues.

    Arguments:

    - `entity1`: `Bio.PDB.Structure.Structure` or `Bio.PDB.Residue.Residue` object.
    - `entity2`: `Bio.PDB.Structure.Structure` or `Bio.PDB.Residue.Residue` object.
    - `debug`: print the names of the two entities and the distance between them.
    """

    if isinstance(entity1, Bio.PDB.Structure.Structure) and isinstance(
        entity2, Bio.PDB.Structure.Structure
    ):
        return distance_between_structures(entity1, entity2, debug=debug)
    if isinstance(entity1, Bio.PDB.Residue.Residue) and isinstance(
        entity2, Bio.PDB.Residue.Residue
    ):
        return distance_between_residues(entity1, entity2, debug=debug)
    if isinstance(entity1, Bio.PDB.Structure.Structure) and isinstance(
        entity2, Bio.PDB.Residue.Residue
    ):
        return distance_between_structures(
            entity1, resi_to_struct(entity2), debug=debug
        )
    if isinstance(entity1, Bio.PDB.Residue.Residue) and isinstance(
        entity2, Bio.PDB.Structure.Structure
    ):
        return distance_between_structures(
            resi_to_struct(entity1), entity2, debug=debug
        )


def get_molecular_mass(
    structure: Bio.PDB.Structure.Structure, atom_table: str = ATOM_TABLE
) -> float:
    """
    Takes the `Bio.PDB.Structure.Structure` object as input.
    Calculates molecular mass of the given structure.

    Atom table contains the information about the atoms.

    Returns the molecular mass of the structure.

    Arguments:

    - `structure`: `Bio.PDB.Structure.Structure` object or `Bio.PDB.Residue.Residue` object.
    - `atom_table`: str - path to the `pd.DataFrame` object.
    """
    if isinstance(structure, Bio.PDB.Structure.Structure):
        mol_weight: int = 0
        for model in structure.get_list():
            for chain in model.get_list():
                for residue in chain.get_list():
                    for atom in residue.get_list():
                        atomic_weight = (
                            atom_table.loc[
                                atom_table["Symbol"] == atom.element.capitalize()
                            ]
                            .squeeze()
                            .at["AtomicMass"]
                        )
                        # print(f"Asked for {atom.element.capitalize()}")
                        # print(f"Got {atom_table.loc[atom_table['Symbol'] == atom.element.capitalize()].squeeze().at['Symbol']}")
                        mol_weight += atomic_weight
        return mol_weight

    elif isinstance(structure, Bio.PDB.Residue.Residue):
        mol_weight: int = 0
        for atom in structure.get_list():
            atomic_weight = (
                atom_table.loc[atom_table["Symbol"] == atom.element.capitalize()]
                .squeeze()
                .at["AtomicMass"]
            )
            mol_weight += atomic_weight
        return mol_weight
    else:
        raise ValueError(
            "structure must be a Bio.PDB.Structure.Structure object or Bio.PDB.Residue.Residue object"
        )


def part_of_protein(
    residue: Bio.PDB.Residue.Residue, protein_structure: Bio.PDB.Structure.Structure
) -> bool:
    """
    Takes the `Bio.PDB.Residue.Residue` object and the `Bio.PDB.Structure.Structure` object as input.
    Checks if the given residue is part of a protein.
    Returns either True or False.

    Arguments:

    - `residue`: Bio.PDB.Residue.Residue object
    - `target_structure`: Bio.PDB.Structure.Structure object
    """
    if residue.resname in get_sequence(protein_structure):
        return True
    else:
        return False


class StructureGraph:
    """
    A grapg respresentation of a `Bio.PDB.Structure.Structure` object.

    A simple implementation involving only the `Bio.PDB.Residue.Residue` objects.
    """

    def __init__(self, residues: list[Bio.PDB.Residue.Residue]):
        self.residues = residues
        self.graph = {}

    def create_graph(self, distance_threshold: float = 2.0):
        """
        Takes the distance threshold value as input, which is treated a as cut-off of bonded state.
        Creates the graph of the given structure with the residues as nodes.

        Nodes are connected if the distance between them is less than the given threshold.

        Arguments:

        - `distance_threshold`: float - distance threshold value in Angstrom.
        """
        for residue in self.residues:
            self.graph[residue] = []

        for residue_m in self.graph:
            for residue_n in self.graph:
                if residue_m != residue_n and (distance_between_residues(residue_m, residue_n) < distance_threshold):
                    self.graph[residue_m].append(residue_n)

    def print_connectivity(self):
        """
        Prints the connectivity of the graph.

        Basically the same as calling `StructureGraph.graph`, but prettier.
        """
        for index, node in enumerate(self.graph):
            print(f"\n {index+1}: Node {node.resname} is adjacent to: ", end="")
            for neighbor in self.graph[node]:
                print(f"{neighbor.resname}, ", end="")

    def get_connected(
        self, starting_node: Bio.PDB.Residue.Residue
    ) -> list[Bio.PDB.Residue.Residue]:
        """
        Takes one node of the graph as input - `starting_node`, i.e., the Residue within the structure.

        Returns a `list` of all `Bio.PDB.Residue.Residue` that are connected to the given node.

        Arguments:

        - `starting_node`: Bio.PDB.Residue.Residue object
        """
        queue: list = []
        explored: list = []
        explored.append(starting_node)
        queue.append(starting_node)

        connected_residues: list = []
        while queue != []:
            current_node = queue.pop(0)
            for neighbor in self.graph[current_node]:
                if neighbor not in explored:
                    queue.append(neighbor)
                    explored.append(neighbor)
                    connected_residues.append(neighbor)
        return connected_residues

    def get_clusters(self) -> list[Bio.PDB.Residue.Residue]:
        """
        Takes the `list` of `Bio.PDB.Residue.Residue` objects as input.
        Organizes them into clusters based on the connectivity of the graph.

        Returns a `list` of `list`'s of `Bio.PDB.Residue.Residue` objects.
        """
        clusters: list = []
        for node in self.graph:
            candidate: list = self.get_connected(node)
            candidate.append(node)
            candidate.sort()
            if len(candidate) == 1:
                clusters.append([node])
            else:
                if candidate not in clusters:
                    clusters.append(candidate)
        return clusters

    def get_graph(self):
        return self.graph

    def get_residues(self):
        return self.residues


def combine(
    residues: list[Bio.PDB.Residue.Residue], resseq=999
) -> Bio.PDB.Residue.Residue:
    """
    Takes the `list` of `Bio.PDB.Residue.Residue` objects as input.

    Combines them into a single `Bio.PDB.Residue.Residue` object.

    Returns that single `Bio.PDB.Residue.Residue` object.

    The name and ID of the combined residue are set to those of the first residue for now.

    Arguments:
    - `residues`: `list` of `Bio.PDB.Residue.Residue` objects
    - `resseq`: internal variable needed by Biopython
    """
    if len(residues) == 1:
        return residues[0]

    else:
        combined_residue_name = "".join(
            [residue.resname[0] for residue in residues]
        ).strip()[:3]
        combined_residue = Bio.PDB.Residue.Residue(
            ("H_" + combined_residue_name, resseq, " "), combined_residue_name, " "
        )

        for index, residue in enumerate(residues):
            for atom in residue.get_list():
                atom.name = atom.name + "_" + str(index)
                atom.id = atom.id + "_" + str(index)
                combined_residue.add(atom)

        return combined_residue


def filter_ligands(
    residues: list[Bio.PDB.Residue.Residue],
    num_atoms: int = 5,
    mol_weight: int = 150,
    rmsd=2.0,
    debug=False,
) -> list[Bio.PDB.Residue.Residue]:
    """
    Takes the list of residues as input and filters them according to `num_atoms`
    and `mol_weight` criteria.

    Returns the `list` of `Bio.PDB.Residue.Residue` objects.

    Arguments:

    - `residues`: `list` of `Bio.PDB.Residue.Residue objects`.
    - `num_atoms`: `int` - the number of atoms in the ligand.
    - `mol_weight`: `int` - the molecular weight of the ligand.
    - `debug`: `bool` - if True, debug information to the console.
    """

    # Preliminary filtering based on number of atoms and molecular mass
    filtered_residues: list = []
    for residue in residues:
        if (len(residue.get_list()) > num_atoms and get_molecular_mass(residue) >= mol_weight):
            filtered_residues.append(residue)

    if debug:
        print(f"Criteria: num_atoms = {num_atoms}, mol_weight = {mol_weight}")
        print(f"Number of residues before filtering: {len(residues)}")
        print(f"Number of residues after filtering: {len(filtered_residues)}")

    # list of future ligands with unique conformations
    unique_ones = []

    # while the list of filtered ligands is not empty
    while filtered_residues:
        # taking an element from the filtered residues and adding it
        # to the unique ones
        residue = filtered_residues.pop()
        unique_ones.append(residue)
        # deleting all other residues with the similar conformations
        # from the original list of filtered residues
        for index, other_residue in enumerate(filtered_residues):
            if similar_conformation(residue, other_residue, rmsd):
                filtered_residues.pop(index)

    return unique_ones


def similar_conformation(
    residue1: Bio.PDB.Residue.Residue,
    residue2: Bio.PDB.Residue.Residue,
    rmsd_threshold=2.0,
    debug=False,
) -> bool:
    """
    Takes two `Bio.PDB.Residue.Residue` objects as input and checks if
    their conformation are similar based on the rmsd.

    If two different residues are compared, automatically returns `False`.

    Returns `True` if the conformation are similar, `False` otherwise.

    Arguments:

    - `residue1`: `Bio.PDB.Residue.Residue` object.
    - `residue2`: `Bio.PDB.Residue.Residue` object.
    - `rmsd_threshold`: `float` - the rmsd threshold.
    - `debug`: `bool` - if `True`, prints debug information to the console.
    """

    if residue1.id[0] != residue2.id[0] or len(residue1.get_list()) != len(residue2.get_list()):
        return False

    _rmsd_before, rmsd_after = get_rmsd(residue1, residue2, debug=debug)
    if rmsd_after < rmsd_threshold:
        return True
    else:
        return False


def get_rmsd(
    residue1: Bio.PDB.Residue.Residue, residue2: Bio.PDB.Residue.Residue, debug=False
) -> tuple:
    """
    Takes two `Bio.PDB.Residue.Residue` objects as input with the same chemical formala.

    Superimposes the two given structures onto each other.

    Returns the `tuple` of two rmsd values: `rms_before` and `rms_after` transformation.

    Arguments:

    - `residue1`: `Bio.PDB.Residue.Residue`
    - `residue2`: `Bio.PDB.Residue.Residue`
    - `debug`: `bool` - if True, prints debug information to the console.
    """
    atoms1 = copy.deepcopy(residue1.get_list())
    atoms2 = copy.deepcopy(residue2.get_list())
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(atoms1, atoms2)
    rms_before = sup.rms

    if debug is True:
        print(f"RMSD between {residue1.resname} and {residue2.resname} before transformation: {sup.rms}")

    sup.apply(atoms1)

    if debug is True:
        print(f"RMSD between {residue1.resname} and {residue2.resname} after transformation: {sup.rms}")

    rms_after = sup.rms

    return (rms_before, rms_after)


def save_ligand_to_file(
    ligand_list: list[Bio.PDB.Structure.Structure],
    output_directory: str,
    original_structure: Bio.PDB.Structure.Structure,
    debug=False,
) -> None:
    """
    Writes a PDB file with the ligand structure.

    - `ligand_list`: list[Bio.PDB.Structure.Structure]
    - `output_directory`: directory to which the pdb file will be written
    - `original_structure`: Bio.PDB.Structure.Structure object used for the file name
    - `log`: prints the description of the ligand to the log file
    - `debug`: prints the file path to the console.
    """
    if len(ligand_list) == 1:
        ligand_file = PDBIO()
        ligand_file.set_structure(ligand_list[0])
        structure_directory = original_structure.get_id()
        file_name = "".join([original_structure.get_id(), "_", ligand_list[0].get_id(), ".pdb"])
        file_path = os.path.join(output_directory, structure_directory, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ligand_file.save(file_path)

    else:
        structure_directory = original_structure.get_id()
        for index, structure in enumerate(ligand_list):
            ligand_file = PDBIO()
            ligand_file.set_structure(structure)
            file_name = "".join(
                [
                    str(index),
                    "_",
                    original_structure.get_id(),
                    "_",
                    structure.get_id(),
                    ".pdb",
                ]
            )
            file_path = os.path.join(output_directory, structure_directory, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            ligand_file.save(file_path)


def extract(
    original_structure: Bio.PDB.Structure.Structure,
    output_directory: str,
    distance_threshold: float,
    num_atoms: int,
    mol_weight: int,
    rmsd: float,
    debug=False
    ) -> None:

    hetero_residues = get_hetero_residues(original_structure, debug=debug)
    structure = StructureGraph(hetero_residues)
    structure.create_graph(distance_threshold=distance_threshold)
    clusters = structure.get_clusters()
    residues = list()
    for cluster in clusters:
        residues.append(combine(cluster))
    ligands = filter_ligands(residues, num_atoms=num_atoms, mol_weight=mol_weight, rmsd=rmsd, debug=debug)

    if len(ligands) > 1 or len(ligands) == 0:
        print(f"Structure {original_structure.get_id()} has {len(ligands)} suitable ligands.")
        print('using fpocket...')


    ligand_structures = [resi_to_struct(ligand, original_structure=original_structure) for ligand in ligands]
    save_ligand_to_file(ligand_structures, output_directory, original_structure, debug=True)


def get_distance_to_pocket(structure: Bio.PDB.Residue.Residue, pocket: namedtuple) -> float:
    """
    Takes `Bio.PDB.Residue.Residue` and `pocket` as inputs and calculates the minimum
    distance between the pocket center and the closest atom of the residue.

    Returns `min_distance` as float.

    Arguments:
    - `structure` : 'Bio.PDB.Residue.Residue' object.
    - `pocket` : namedtuple.
    """
    min_distance = 1000
    for atom in structure.get_list():
        dist = distance.euclidean(atom.get_coord(), pocket.center)
        if dist < min_distance:
            min_distance = dist
    return min_distance


def filter_fpocket(
    original_structure_id: str,
    ligands: list[Bio.PDB.Residue.Residue],
    distance_threshold=2.5,
    pdb_path=PDB_PATH,
) -> list:

    """
    Takes the potential ligands as a list of `Bio.PDB.Residue.Residue` objects as input,
    runs the fpocket calculation and determines the distance between each ligand
    and the supplied pockets.

    Arguments:

    - `original_structure_id` - id for the structure lookup.
    - `ligands` - list of potential ligands.
    - `distance_threshold` - minimum distance between a ligand and a pocket for the latter
    to be considered a real ligand.
    - `pdb_path` - (optional) path to the local copy of PDB. Currently defaults to my copy.
    - `debug` - prints debug information.
    """

    cleaned_pdb: str = ps.clean_pdb(
        os.path.join(pdb_path, original_structure_id[1:-1], "pdb" + original_structure_id + ".ent")
    )
    fpocket_dir: str = ps.fpocket(cleaned_pdb)
    _barycenters = ps.get_centers(fpocket_dir)
    pockets: list = ps.get_pockets(fpocket_dir, num_pockets_threshold=3)
    ps.write_bat(os.path.join(pdb_path, original_structure_id[1:-1]), fpocket_dir)

    filtered_ligands: list = []
    for ligand in ligands:
        for pocket in pockets:
            if (get_distance_to_pocket(ligand, pocket) < distance_threshold and ligand not in filtered_ligands):
                filtered_ligands.append(ligand)
                break
    return filtered_ligands


def main():
    distance_threshold = 1.6
    num_atoms = 15
    mol_weight = 200
    rmsd = 1.5
    while True:
        print("1.  find and process structure with the given id.")
        print(
            "2.  settings (filtration criteria: number of atoms, molecular mass, and rmsd)"
        )
        print("3.  process random structure. Enter for 1 random structure.")
        print("q   : quit program.")
        user_input = input("> ")
        if user_input == "q":
            print("Quitting...")
            exit()
        elif user_input == "2":
            while True:
                print(f"1. Minimal number of atoms: {num_atoms}.")
                print(f"2. Minimal molecular weight: {mol_weight}.")
                print(f"3. RMSD threshold: {rmsd}.")
                print("4. Go back.")
                user_input = input("> ")
                if user_input == "1":
                    try:
                        num_atoms = int(input("Enter the desired value: "))
                    except ValueError:
                        print("Invalid input. Try again.")
                elif user_input == "2":
                    try:
                        mol_weight = float(input("Enter the desired value: "))
                    except ValueError:
                        print("Invalid input. Try again.")
                elif user_input == "3":
                    try:
                        rmsd = float(input("Enter the desired value: "))
                    except ValueError:
                        print("Invalid input. Try again.")
                elif user_input == "4":
                    break

        elif user_input == "":
            random_structure = get_random_structures(PDB_PATH, 1, debug=False)
            print(f"Extracting ligands from 1 random structure {random_structure.id}...")
            extract(random_structure, OUT_PATH, distance_threshold,  num_atoms, mol_weight, rmsd, debug=False)

        elif user_input.startswith("3"):
            print("b   : Go back.")
            print("q   : Quit program.")
            print("Number of random structures: [1]")
            while True:
                user_input = input("> ")
                if user_input == "":
                    random_structure = get_random_structures(PDB_PATH, 1, debug=False)
                    print(
                        f"Extracting ligands from 1 random structure {random_structure.id}..."
                    )
                    extract(
                        random_structure,
                        OUT_PATH,
                        distance_threshold,
                        num_atoms,
                        mol_weight,
                        rmsd,
                        debug=False,
                    )
                elif user_input.isdigit():
                    try:
                        random_structures = get_random_structures(
                            PDB_PATH, user_input, debug=False
                        )
                        print(
                            f"Extracting ligands from {user_input} random structures..."
                        )
                        for random_structure in random_structures:
                            extract(
                                random_structure,
                                OUT_PATH,
                                distance_threshold,
                                num_atoms,
                                mol_weight,
                                rmsd,
                                debug=False,
                            )
                    except ValueError:
                        print("Invalid input. Try again.")
                elif user_input == "b":
                    break
                elif user_input == "q":
                    exit()

        elif user_input == "1":
            print("Enter the structure id: ")
            user_input = input("> ")
            try:
                print(f"Extracting ligands from structure {user_input}...")
                structure = get_structure(user_input, PDB_PATH, debug=True)
                extract(structure, OUT_PATH, distance_threshold, num_atoms, mol_weight, rmsd, debug=False)
            except Exception as exc:
                raise Exception("Invalid input. Try again.") from exc

        else:
            print("Invalid input. Try again.")


if __name__ == "__main__":
    main()
