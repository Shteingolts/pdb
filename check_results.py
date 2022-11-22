import os

OUT_DIR = os.path.join('/mnt/c/users/serge/desktop', 'test_structures')

total = 0
for struct_dir in os.listdir(OUT_DIR):
    path = os.path.join(OUT_DIR, struct_dir)
    ligands = len(os.listdir(path))
    if ligands > 1:
        total += 1
        print(f'Structure {struct_dir} has {ligands} ligands!')

print(f'{total} out of {len(os.listdir(OUT_DIR))} structures have more than 1 ligand!')