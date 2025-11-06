import argparse
import copy
import json
import pickle
parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default='data/4AA_data')
parser.add_argument('--mddir', type=str, default='data/4AA_sims')
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_batches', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--out_dir', type=str, default="tps_output")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--n_chunks', type=int, default=1)
parser.add_argument('--relax', action='store_true')  # linspace
parser.add_argument('--sample', type=str, default="linear", choices=["linear", "uniform"])
args = parser.parse_args()
args = parser.parse_args()
import tps_flow.analysis
import os, torch, mdtraj, tqdm
from tps_flow.geometry import atom14_to_atom37, atom37_to_torsions
from tps_flow.tensor_utils import tensor_tree_map
from tps_flow.utils import atom14_to_pdb

from tps_flow.residue_constants import restype_order
from tps_flow.wrapper import TPS_Flow
from tps_flow.dataset import atom14_to_frames
import pandas as pd
import contextlib
import numpy as np
from pyrosetta import *
import timeit
from Bio.PDB.PDBParser import PDBParser
import tps_flow.residue_constants as rc



def structure_to_atom14(structure):
    residue_count = len(list(structure.get_residues()))
    arr = np.zeros((1, residue_count, 14, 3), dtype=np.float16)
    for i, residue in enumerate(structure.get_residues()):
        resi_name = residue.get_resname()
        for atom in residue:
            at_name = atom.get_name()
            if at_name not in rc.restype_name_to_atom14_names[resi_name]:
                    #  print(resi_name, at_name, 'not found')
                     continue
            j = rc.restype_name_to_atom14_names[resi_name].index(at_name)
            coords = atom.get_coord()
            arr[:, i, j] = coords

    return arr


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

os.makedirs(args.out_dir, exist_ok=True)

def get_sample(arr, seqres, start_idxs, end_idxs, start_state, end_state, num_frames=1000, energy=-1000, rosetta_relax=True):
    start_idx = np.random.choice(start_idxs, 1).item()
    end_idx = np.random.choice(end_idxs, 1).item()
    
    select_energy = energy 
    if (energy > -1000).all()  and len(energy) > 10:
        start_energy = energy[start_idx]
        end_energy = energy[end_idx]
        if args.sample == 'uniform':
            select_energy = np.random.uniform(start_energy, end_energy, num_frames).astype(np.float32)
            select_energy[0] = start_energy
            select_energy[-1] = end_energy
        else:
            select_energy = np.linspace(start_energy, end_energy, num_frames, dtype=np.float32)
        # select_energy = start_energy.expand(num_frames,).clone()
        # select_energy[-1] = end_energy
        # select_energy = np.linspace(start_energy, end_energy, num_frames, dtype=np.float32)
        
    start_arr = np.copy(arr[start_idx:start_idx + 1][:, :len(seqres)]).astype(np.float32)  # (1, L, 14, 3)
    end_arr = np.copy(arr[end_idx:end_idx + 1][:, :len(seqres)]).astype(np.float32)    

    
       
        
    seqres = torch.tensor([restype_order[c] for c in seqres])
    start_frames = atom14_to_frames(torch.from_numpy(start_arr))
    start_atom37 = torch.from_numpy(atom14_to_atom37(start_arr, seqres[None])).float()
    start_torsions, start_torsion_mask = atom37_to_torsions(start_atom37, seqres[None])  #(1,4,7,2) (1,4,7)
    
    end_frames = atom14_to_frames(torch.from_numpy(end_arr))
    end_atom37 = torch.from_numpy(atom14_to_atom37(end_arr, seqres[None])).float()
    end_torsions, end_torsion_mask = atom37_to_torsions(end_atom37, seqres[None])
    L = start_frames.shape[1]
    traj_torsions = start_torsions.expand(num_frames, -1, -1, -1).clone()
    traj_torsions[-1] = end_torsions

    traj_trans = start_frames._trans.expand(num_frames, -1, -1).clone()
    traj_trans[-1] = end_frames._trans

    traj_rots = start_frames._rots._rot_mats.expand(num_frames, -1, -1, -1).clone()
    traj_rots[-1] = end_frames._rots._rot_mats

    mask = torch.ones(L)
    return {
        'torsions': traj_torsions,    # (num_frames, 4, 7, 2)
        'torsion_mask': start_torsion_mask[0], # (4, 7)
        'trans': traj_trans,    # (num_frames, 4, 3)
        'rots': traj_rots,    # (num_frames, 4, 3, 3)
        'seqres': seqres,    # (4,)
        'start_idx': start_idx,    # int
        'end_idx': end_idx,    # int
        'start_state': start_state,    # int
        'end_state': end_state,    # int
        'mask': mask,  # (L,)  (4,)
        'energy': select_energy,   # (num_frames,)
    }

def do(model, name, seqres):
    print('doing', name)
    name = name.split('_')[0]
    # xtc_name = f'split_{int(split_name[0])*10}_{int(split_name[1])*10}'
    if os.path.exists(f'{args.out_dir}/{name}_metadata.json'):
        pass
        # return
    if os.path.exists(f'{args.out_dir}/{name}_metadata.pkl'):
        pkl_metadata = pickle.load(open(f'{args.out_dir}/{name}_metadata.pkl', 'rb'))
        msm = pkl_metadata['msm']
        cmsm = pkl_metadata['cmsm']
        ref_kmeans = pkl_metadata['ref_kmeans']
    # if os.path.exists(f'{args.mddir}/{name}_metadata.pkl'):
    #     pkl_metadata = pickle.load(open(f'{args.mddir}/{name}_metadata.pkl', 'rb'))
    #     msm = pkl_metadata['msm']
    #     cmsm = pkl_metadata['cmsm']
    #     ref_kmeans = pkl_metadata['ref_kmeans']
    else:
        with temp_seed(137):
            top_file = os.path.join(args.mddir, 'apo_sol_strip_wat_lipid.prmtop')
            # feats, ref = tps_flow.analysis.get_featurized_traj(f'{args.mddir}/{name}/{name}', sidechains=True)
            # feats, ref = tps_flow.analysis.get_featurized_traj_rMD(top_file, f'{args.mddir}/{xtc_name}.xtc',sidechains=True)

            feats, ref = tps_flow.analysis.get_featurized_traj_nc_apo(f'{args.mddir}/{name}.nc', top_file, sidechains=True) # /home/xk/mount/md_file
            #
            tica, _ = tps_flow.analysis.get_tica(ref, lag=40)
            kmeans, ref_kmeans = tps_flow.analysis.get_kmeans(tica.transform(ref), k=4)
            try:
                msm, pcca, cmsm = tps_flow.analysis.get_msm(ref_kmeans, lag=40, nstates=4)

            # tica, _ = tps_flow.analysis.get_tica(ref)
            # kmeans, ref_kmeans = tps_flow.analysis.get_kmeans(tica.transform(ref))
            # try:
            #     msm, pcca, cmsm = tps_flow.analysis.get_msm(ref_kmeans)
            except Exception as e:
                print('ERROR', e, name, flush=True)
                return
        pickle.dump({
            'msm': msm,
            'cmsm': cmsm,
            'tica': tica,
            'pcca': pcca,
            'kmeans': kmeans,
            'ref_kmeans': ref_kmeans,
        }, open(f'{args.out_dir}/{name}_metadata.pkl', 'wb'))

    flux_mat = cmsm.transition_matrix * cmsm.pi[None, :]
    flux_mat[flux_mat < 0.0000001] = np.inf  # set 0 flux to inf so we do not choose that as the argmin
    start_state, end_state = np.unravel_index(np.argmin(flux_mat, axis=None), flux_mat.shape) # 索引转换为一个元组
    ref_discrete = msm.metastable_assignments[ref_kmeans]
    start_idxs = np.where(ref_discrete == start_state)[0]
    end_idxs = np.where(ref_discrete == end_state)[0]  #([])->[]
    if (ref_discrete == start_state).sum() == 0 or (ref_discrete == end_state).sum() == 0:
        print('No start or end state found for ', name, 'skipping...')
        return

    # arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r') #(999999, 4, 14, 3)
    # arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}.npy', 'r') #(999999, 4, 14, 3)
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}_.npy', 'r')
    energy = -1000
    if hasattr(model.args, 'energy'):
            engergy_path = os.path.dirname(args.data_dir)
            energy_csv = os.path.join(engergy_path, f'{name}_processed', 'traj_info.csv')
            if not os.path.exists(energy_csv):
                raise ValueError('Energy file does not exist')
            df = pd.read_csv(energy_csv)
            energy = df['energy'].values
    

    metadata = []
    for i in tqdm.tqdm(range(args.num_batches), desc='num batch'):
        batch_list = []
        for _ in range(args.batch_size):
            batch_list.append(
                get_sample(arr, seqres, copy.deepcopy(start_idxs), end_idxs, start_state, end_state, num_frames=args.num_frames, energy=energy, rosetta_relax=args.relax))
        batch = next(iter(torch.utils.data.DataLoader(batch_list, batch_size=args.batch_size)))
        batch = tensor_tree_map(lambda x: x.cuda(), batch)
        print('Start tps for ', name, 'with start coords', batch['trans'][0, 0, 0])
        print('Start tps for ', name, 'with end coords', batch['trans'][0, -1, 0])
        atom14s, _ = model.inference(batch)
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            path = os.path.join(args.out_dir, f'{name}_{idx}.pdb')
            atom14_to_pdb(atom14s[j].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

            traj = mdtraj.load(path)
            traj.superpose(traj)
            traj.save(os.path.join(args.out_dir, f'{name}_{idx}.xtc'))
            traj[0].save(os.path.join(args.out_dir, f'{name}_{idx}.pdb'))
            metadata.append({
                'name': name,
                'start_idx': batch['start_idx'][j].cpu().item(),
                'end_idx': batch['end_idx'][j].cpu().item(),
                'start_state': batch['start_state'][j].cpu().item(),
                'end_state': batch['end_state'][j].cpu().item(),
                'path': path,
            })
    json.dump(metadata, open(f'{args.out_dir}/{name}_metadata.json', 'w'))


@torch.no_grad()
def main():

    model = TPS_Flow.load_from_checkpoint(args.sim_ckpt)
    model.eval().to('cuda')
    add_ckpt = os.path.splitext(os.path.basename(args.sim_ckpt))[0]
    args.out_dir = os.path.join(args.out_dir, add_ckpt)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    df = pd.read_csv(args.split, index_col='name')
    for name in df.index:
        if args.pdb_id and name not in args.pdb_id:
            continue
        do(model, name, df.seqres[name][:model.args.crop])

 


main()

# --sim_ckpt interpolation.ckpt --data_dir data/4AA_data --num_frames 100 --split splits/4AA_test_litter.csv --suffix _i100 --mddir data/4AA_sims --out_dir tps/results/0506_tps_1ns --batch_size 1

