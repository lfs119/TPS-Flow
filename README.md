# TPS-FLOW

Implementation of[Enhanced Hybrid Physical and Flow-Based Generative Modeling for Transition Path Sampling of Membrane Protein Dynamics toward Tuberculosis Drug Discovery]


## Abstract

A hybrid physics-based and data-driven flow-based generation method for sampling of MP transition pathways. Our method combines structural and energy constraints with a normalized flow model trained on MP conformational data, enabling the generation of physically plausible trajectories. A control energy module embedded as model input allows for the use of multiple sampling strategies during inference to enhance coverage of the sampling space. Validation on representative MPs, such as the MTB MmpL3 protein, demonstrates that our method generates pathway conformations with an average Tm score of 0.83 and can generate novel MP conformations not observed in the original MD trajectories, thereby expanding the MP conformational space beyond traditional sampling limits and improving sampling efficiency by 14-fold. The model's performance in two tasks, generating mutant conformations and generating predefined intermediate conformations, was analyzed simultaneously. Cross-docking against the four predefined PD-domain states yielded Vina scores < −10 kcal mol⁻¹ for all conformers. Single- and double-point mutants constructed in the PN sub-domain were accurately rebuilt (TM-score > 0.82), retaining native-like flexibility profiles (Pearson ρ > 0.74). The results suggest that our method can be used to generate biologically plausible MP conformations and intermediate states, which could facilitate the discovery of new drugs for TB.


## Installation

```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 
pip install pytorch_lightning==2.0.4 mdtraj==1.9.9 biopython==1.79
pip install wandb dm-tree einops torchdiffeq pyEMMA torchtyping transformer
pip install matplotlib==3.7.2 numpy==1.23.5  pandas==1.5.3
pip install -U vina meeko

```


## MD topology and coordinates

```
https://zenodo.org/records/17233825  and  https://zenodo.org/records/15687037
```


## Model weights
The model weights used in the paper may be downloaded here

```
https://zenodo.org/records/17223963

```

## Inference
```
python tps_inference_apo.py --sim_ckpt path_to_ckpt --data_dir path_to_data --split path_to_split --suffix _i100  --num_frames 120 --mddir path_to_mddir --outdir path_to_output  --batch_size batch_size   --num_batches num_batches 
```

## Train
```
python train.py --tps_condition  --train_split splits/apo_train.csv --val_split splits/apo_val.csv  --data_dir path_to_data --num_frames 20 --prepend_ipa --abs_pos_emb --crop 730 --ckpt_freq 1000 --val_repeat 5   --suffix _i100  --epochs 10000 --run_name  run_name  --batch_size 2 --toker_ecoder  --energy  --recon

```

## Acknowledgement
```
The code is based on the following repositories:
https://github.com/bjing2016/mdgen.git

```

## Lincense
```
Code is released under MIT LICENSE.
```


