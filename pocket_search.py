from pathlib import Path
from collections import namedtuple
import os
import re
import numpy as np
import Bio.PDB
from Bio.PDB import PDBParser, Select

PDB_PATH = "example/path/pdb"
OUT_PATH = "example/out/path"

class NonHetSelect(Select):
    """BioPython class for selecting desired structure parts."""
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue,standard=True) else 0


def clean_pdb(input_file: str, output_directory: str = None):
    """
    Takes a PDB file and returns a cleaned PDB file.

    Arguments:

    - `input_file`: the input PDB file
    - `output_directory`: the optional output directory
    """
    pdb = PDBParser().get_structure("protein", input_file)
    io = Bio.PDB.PDBIO()
    io.set_structure(pdb)
    file_name: str = os.path.basename(input_file).split('.')[0].replace('pdb', '') + '.pdb'
    if output_directory is None:
        io.save(os.path.join(OUT_PATH, file_name), NonHetSelect())
        return os.path.join(OUT_PATH, file_name)
    else:
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        io.save(os.path.join(output_directory, file_name), NonHetSelect())
        return os.path.join(output_directory, file_name)


# Code for this function was shamelessly stolen
# from DeepPocket (https://github.com/devalab/DeepPocket)
def fpocket(protein_file: str) -> str:
    """
    Launches the focket and returns the path to the resulting pockets directory.

    Arguments:

    - `protein_file`: the input PDB file
    """
    os.system(f'fpocket -f {protein_file}')
    fpocket_dir: str = os.path.join(protein_file.replace('.pdb', '_out'), 'pockets')
    return fpocket_dir

# Code for this function was shamelessly stolen
# from DeepPocket (https://github.com/devalab/DeepPocket)
def get_centers(pockets_dir):
    """
    Takes pocket files produced by fpocket as input
    and writes a file containign the barycenters of the pockets.

    Arguments:

    - `pockets_dir`: the input directory with the fpocket pocket files.
    """
    with open(os.path.join(pockets_dir, 'barycenters.txt'), 'w') as out_f:
        for d in os.listdir(pockets_dir):
            centers = []
            masses = []
            if d.endswith('vert.pqr'):
                num = int(re.search(r'\d+', d).group())
                with open(os.path.join(pockets_dir, d)) as pocket_f:
                    for line in pocket_f:
                        if line.startswith('ATOM'):
                            center=list(map(float,re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", ' '.join(line.split()[5:]))))[:3]
                            mass=float(line.split()[-1])
                            centers.append(center)
                            masses.append(mass)



                    centers = np.asarray(centers)
                    masses = np.asarray(masses)
                    xyzm = (centers.T * masses).T
                    xyzm_sum = xyzm.sum(axis=0) # find the total xyz*m for each element
                    cg = xyzm_sum / masses.sum()
                    out_f.write(str(num) + '\t' + str(cg[0]) + '\t' + str(cg[1]) + '\t' + str(cg[2]) + '\n')
    return os.path.join(pockets_dir, 'barycenters.txt')


def get_pockets(pockets_dir: str, num_pockets_threshold=3) -> dict:
    """
    Takes a directory with pocket files produced by fpocket.
    Returns a list of namedtuples with the pocket number,
    score and coordinates.

    Arguments:

    - `pockets_dir` - directory with all pockets files produced by fpocket.
    - `num_pockets` - threshold for pocket analysis
    """

    fpocket_directory = os.path.dirname(pockets_dir)
    pocket_info = []
    for file in os.listdir(fpocket_directory):
        if file.endswith('_info.txt'):
            info_file = os.path.join(fpocket_directory, file)
            print(info_file)
    num_pockets = 0
    with open(info_file) as info_file, open(os.path.join(pockets_dir, 'barycenters.txt')) as barycenters_file:
        pockets = info_file.readlines()
        # dict of pocket numbers as keys and coordinates of pocket centers as tuples of three floats
        barycenters = {line.split()[0] : tuple(float(x) for x in line.split()[1:]) for line in barycenters_file.readlines()}
        for index, line in enumerate(pockets):
            if 'Pocket' in line and num_pockets < num_pockets_threshold:
                num_pockets += 1
                pocket_number = int(line.split()[-2])
                pocket_score = float(pockets[index+1].split()[-1])
                pocket_center = barycenters[line.split()[-2]]
                Pocket = namedtuple('Pocket', ['number', 'score', 'center'])
                pocket = Pocket(pocket_number, pocket_score, pocket_center)
                pocket_info.append(pocket)
    # print(pocket_info)
    return pocket_info

def write_bat(input_file: str, pockets_dir: str):
    file_name: str = os.path.basename(input_file).split('.')[0].replace('pdb', '')
    script_name = file_name + '_VMD.bat'
    with open(os.path.join(os.path.dirname(pockets_dir), script_name), 'w') as f:
        f.write(f'@echo off\n\nvmd "{file_name}_out.pdb" "-e" "{file_name}.tcl"')






