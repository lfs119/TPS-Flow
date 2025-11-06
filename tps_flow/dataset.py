import os
import torch
from .rigid_utils import Rigid
from .residue_constants import restype_order, restype_order_with_x
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
       
class tps_flowDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        if self.args.atlas:
            i = np.random.randint(1, 4)
            full_name = f"{name}_R{i}"
        else:
            full_name = name
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r') #(10000, 4, 14, 3)
        # arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}.npy', 'r') #(10000, 4, 14, 3)   

        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        
        frame_start = np.random.choice(np.arange(arr.shape[0] - self.args.num_frames))
        if self.args.overfit_frame:
            frame_start = 0
        end = frame_start + self.args.num_frames
        # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
        arr = np.copy(arr[frame_start:end]).astype(np.float32) # / 10.0 # convert to nm
        if self.args.copy_frames:
            arr[1:] = arr[0]

        if self.args.replace_frames:
            out_bin = np.lib.format.open_memmap(os.path.join(self.args.data_dir, 'res_out_fixed.npy'), 'r').astype(np.float32)  # (1, 472, 14, 3)
            in_bin = np.lib.format.open_memmap(os.path.join(self.args.data_dir, 'res_in_fixed.npy'), 'r').astype(np.float32) 
            arr[0] = out_bin
            arr[-1] = in_bin
        
        if self.args.energy:
            engergy_path = os.path.dirname(self.args.data_dir)
            energy_csv = os.path.join(engergy_path, f'{full_name}processed', 'traj_info.csv')
            if not os.path.exists(energy_csv):
                raise ValueError('Energy file does not exist')
            df = pd.read_csv(energy_csv)
            suffix = self.args.suffix.split('i')[-1]
            energy = df['energy'].values[::int(suffix)]
            energy = np.copy(energy[frame_start:end]).astype(np.float32)
            # noise_energy = energy + np.random.normal(0, 0.001, energy.shape) 
            noise_energy = np.random.uniform(energy[0], energy[-1], energy.shape) + energy
            noise_energy = np.copy(noise_energy).astype(np.float32)
        else:
            energy = -1000 * np.ones(self.args.num_frames, dtype=np.float32)
            noise_energy =  -1000 * np.ones(self.args.num_frames, dtype=np.float32)

        

        # arr should be in ANGSTROMS
        frames = atom14_to_frames(torch.from_numpy(arr))
        seqres = np.array([restype_order[c] for c in seqres])
        # seqres = np.array([restype_order_with_x[c] for c in seqres])
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float() #(100,4,37,3)
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        if self.args.no_frames:
            return {
                'name': full_name,
                'frame_start': frame_start,
                'atom37': atom37,
                'seqres': seqres,
                'mask': restype_atom37_mask[seqres], # (L,)
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)  #(100,4,7,2) (100,4,7)
        
        torsion_mask = torsion_mask[0]

    
        
        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)
                torsions = torsions[:,start:start+self.args.crop]
                frames = frames[:,start:start+self.args.crop]
                seqres = seqres[start:start+self.args.crop]
                mask = mask[start:start+self.args.crop]
                torsion_mask = torsion_mask[start:start+self.args.crop]
                
            
            elif L < self.args.crop:
                pad = self.args.crop - L
                frames = Rigid.cat([
                    frames, 
                    Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
                ], 1)
                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
                torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,
            'mask': mask, # (L,)
            'energy': noise_energy,
            
        }

