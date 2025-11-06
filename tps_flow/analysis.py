import json
import os

import mdtraj
import numpy as np
import pyemma
from tqdm import tqdm

def get_featurizer(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    return feat
    
def get_featurized_traj(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'.xtc', features=feat)
    return feat, traj


def get_featurized_traj_nc_GaMD(path, top_file, sidechains=False, cossin=True):
    # true_name = name.split('_')[0]
    # top_file = os.path.join(path, 'comp_wat.prmtop')
    # md_file = os.path.join(path, true_name+'.nc')
    traj = mdtraj.load(path, top=top_file)
    nc_name = os.path.basename(path).split('.')[0]
    # u = mda.Universe(topology, trajectory)
    # traj.superpose(traj)
    selection_string = "protein"
    selected_atoms = traj.topology.select(selection_string)
    # protein_indices = traj.topology.select(selected_atoms)
    protein_traj = traj.atom_slice(selected_atoms)
    # protein_traj.superpose(protein_traj)
    # 去除氢原子
    non_hydrogen_indices = protein_traj.topology.select('not element H')
    # 创建一个新的仅包含非氢原子的轨迹
    non_hydrogen_traj = protein_traj.atom_slice(non_hydrogen_indices)
    non_hydrogen_traj.superpose(non_hydrogen_traj)

    output_filename = os.path.join(os.path.dirname(path), f'{nc_name}_processed.xtc')
    non_hydrogen_traj.save_xtc(output_filename)

    top_file_res = os.path.join(os.path.dirname(path), 'res_out.pdb')
    feat = pyemma.coordinates.featurizer(top_file_res)
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(output_filename, features=feat)

    # c_alpha_indices = traj.topology.select('name CA')
    # c_alpha_traj = traj.xyz[:, c_alpha_indices, :]
    # c_alpha_traj_flat = c_alpha_traj.reshape(c_alpha_traj.shape[0], -1)

    return None, traj

def get_featurized_traj_nc_apo(path, top_file, sidechains=False, cossin=True):
    # true_name = name.split('_')[0]
    # top_file = os.path.join(path, 'comp_wat.prmtop')
    # md_file = os.path.join(path, true_name+'.nc')
    traj = mdtraj.load(path, top=top_file)
    nc_name = os.path.basename(path).split('.')[0]
    # u = mda.Universe(topology, trajectory)
    # traj.superpose(traj)
    selection_string = "protein"
    selected_atoms = traj.topology.select(selection_string)
    # protein_indices = traj.topology.select(selected_atoms)
    protein_traj = traj.atom_slice(selected_atoms)
    # protein_traj.superpose(protein_traj)
    # 去除氢原子
    non_hydrogen_indices = protein_traj.topology.select('not element H')
    # 创建一个新的仅包含非氢原子的轨迹
    non_hydrogen_traj = protein_traj.atom_slice(non_hydrogen_indices)
    non_hydrogen_traj.superpose(non_hydrogen_traj)
     
    output_filename = os.path.join(os.path.dirname(path), f'{nc_name}_processed.xtc')
    if not os.path.exists(output_filename):
        print("Saving processed trajectory to", output_filename)
        non_hydrogen_traj.save_xtc(output_filename)

    top_file_res = os.path.join(os.path.dirname(path), 'apo_sol_noH_fixed.pdb')
    feat = pyemma.coordinates.featurizer(top_file_res)

    key_pairs = [(90, 473), (40, 440), (50, 450), (60, 460), (70, 468), (80, 485), (100, 493), (120, 510), (130, 520),  (142, 530), (150, 536)]
    ca_pairs = []
    for r1, r2 in key_pairs:
        idx1 = feat.topology.select(f"resid {r1} and name CA")[0]
        idx2 = feat.topology.select(f"resid {r2} and name CA")[0]
        ca_pairs.append([idx1, idx2])
    feat.add_distances(ca_pairs)

    traj = pyemma.coordinates.load(output_filename, features=feat)

    # c_alpha_indices = traj.topology.select('name CA')
    # c_alpha_traj = traj.xyz[:, c_alpha_indices, :]
    # c_alpha_traj_flat = c_alpha_traj.reshape(c_alpha_traj.shape[0], -1)

    return None, traj

def get_featurized_traj_rMD(pdb_file, xtc_file, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(pdb_file)
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(xtc_file, features=feat)
    return feat, traj

def get_featurized_atlas_traj(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'_prod_R1_fit.xtc', features=feat)
    return feat, traj

def get_tica(traj, lag=1000):
    tica = pyemma.coordinates.tica(traj, lag=lag, kinetic_map=True)
    # lag time 100 ps = 0.1 ns
    return tica, tica.transform(traj)

def get_kmeans(traj, k=100):
    kmeans = pyemma.coordinates.cluster_kmeans(traj, k=k, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj)[:,0]

def get_msm(traj, lag=1000, nstates=10):
    msm = pyemma.msm.estimate_markov_model(traj, lag=lag)
    pcca = msm.pcca(nstates)
    # assert len(msm.metastable_assignments) == 100
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj], lag=lag)
    return msm, pcca, cmsm

def discretize(traj, kmeans, msm):
    return msm.metastable_assignments[kmeans.transform(traj)[:,0]]

def load_tps_ensemble(name, directory):
    metadata = json.load(open(os.path.join(directory, f'{name}_metadata.json'),'rb'))
    all_feats = []
    all_traj = []
    for i, meta_dict in tqdm(enumerate(metadata)):
        feats, traj = get_featurized_traj(f'{directory}/{name}_{i}', sidechains=True)
        all_feats.append(feats)
        all_traj.append(traj)
    return all_feats, all_traj


def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)


def get_tp_likelihood(tp, trans):
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[0, -1]
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs


def get_state_probs(tp, num_states=10):
    stationary = np.bincount(tp.reshape(-1), minlength=num_states)
    return stationary / stationary.sum()