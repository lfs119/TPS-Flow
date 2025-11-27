# TPS-FLOW

Implementation of[TPS-Flow: physics-guided flow-based generative modeling of protein transition paths ]


## Abstract

Membrane proteins (MPs) mediate critical cellular processes, yet sampling their dynamic transition pathways—critical for understanding function and guiding drug design—remains challenging due to experimental limitations and the inefficiency of traditional computational methods. Here, we propose a hybrid physics-based and data-driven flow-based generation method for sampling of MP transition pathways. Our method combines structural and energy constraints with a normalized flow model trained on MP conformational data, enabling the generation of physically plausible trajectories. A control energy module embedded as model input allows for the use of multiple sampling strategies during inference to enhance coverage of the sampling space. Validation on representative MPs, such as the MTB MmpL3 protein, demonstrates robust generalizability across additional systems: 1HPV achieved a TM-Score of 0.90, while 1BRS reached a DockQ of 0.81. The method generates pathway conformations with an average TM-Score of 0.83 and can generate novel MP conformations not observed in the original MD trajectories, thereby expanding the MP conformational space beyond traditional sampling limits and improving sampling efficiency by 14-fold. Cross-docking against the four predefined PD-domain states yielded Vina scores < −10 kcal•mol⁻¹ for all conformers. Single- and double-point mutants constructed in the PN sub-domain were accurately rebuilt (TM-Score > 0.82), retaining native-like flexibility profiles (Pearson ρ > 0.74). In addition, a case study on the canonical closed–open transition of the soluble enzyme adenylate kinase shows that TPS-Flow can generate endpoint-conditioned transition paths that traverse the expected closed, half-open–half-closed, and open basins in domain-level collective variables. This work proposes the first MP transition path sampling framework based on hybrid physics data flow, which provides a powerful tool for elucidating MP functional dynamics and accelerating membrane structure-based drug discovery, especially for rare or unobserved conformational states.

## Installation

```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 
pip install pytorch_lightning==2.0.4 mdtraj==1.9.9 biopython==1.79
pip install wandb dm-tree einops torchdiffeq pyEMMA torchtyping transformers
pip install matplotlib==3.7.2 numpy==1.23.5  pandas==1.5.3
pip install -U vina meeko

```


## MD topology and coordinates

```
https://doi.org/10.5281/zenodo.17233825  and   https://doi.org/10.5281/zenodo.15687037
```


## Model weights
The model weights used in the paper may be downloaded here

```
https://doi.org/10.5281/zenodo.17731628

```

## Inference
### MTB MmpL3 protein
```
python tps_inference_apo.py --sim_ckpt path_to_ckpt --data_dir path_to_data --split path_to_split --suffix _i100  --num_frames 120 --mddir path_to_mddir --outdir path_to_output  --batch_size batch_size   --num_batches num_batches 
```
### 1BRS
```
python tps_inference_1brs_fixed.py --sim_ckpt path_to_ckpt --data_dir path_to_data --num_frames 120 --split path_to_split --suffix _i10  --out_dir  path_to_output --batch_size batch_size  --num_batches num_batches
```
### 1HPV
```
python tps_inference_1hpv_fixed.py --sim_ckpt path_to_ckpt --data_dir path_to_data --num_frames 120 --split path_to_split --suffix _i10  --out_dir  path_to_output --batch_size batch_size  --num_batches num_batches
```
### adk
```
python tps_inference_adk_fixed.py --sim_ckpt path_to_ckpt --data_dir path_to_data --num_frames 50 --split path_to_split  --out_dir  path_to_output --batch_size batch_size  --num_batches num_batches
```

## Train
### MTB MmpL3 protein
```
python train.py --tps_condition  --train_split splits/apo_train.csv --val_split splits/apo_val.csv  --data_dir path_to_data --num_frames 20 --prepend_ipa --abs_pos_emb --crop 730 --ckpt_freq 1000 --val_repeat 5   --suffix _i100  --epochs 10000 --run_name  run_name  --batch_size 2 --toker_ecoder  --energy  --recon

```
### 1BRS
```
python train.py --tps_condition --train_split splits/ppmid_train.csv --val_split splits/ppmid_val.csv --data_dir path_to_data --num_frames 60 --prepend_ipa --abs_pos_emb --crop 197 --ckpt_freq 1000 --val_repeat 1 --suffix _i10 --epochs 10000 --run_name run_name --batch_size 3  --toker_ecoder

```
### 1HPV
```
python train.py --tps_condition --train_split splits/chain_train_split.csv --val_split splits/chain_test_split.csv --data_dir path_to_data --num_frames 60 --prepend_ipa --abs_pos_emb --crop 99 --ckpt_freq 1000 --val_repeat 1 --suffix _i10 --epochs 10000 --run_name run_name --batch_size 3  --toker_ecoder

```
### adk
```
python train.py --tps_condition --train_split splits/adk_train.csv --val_split splits/adk_val.csv --data_dirpath_to_data --num_frames 50 --prepend_ipa --abs_pos_emb --crop 214 --ckpt_freq 1000 --val_repeat 5  --epochs 10000 --run_name run_name --batch_size 3  --toker_ecoder
```

## Model Metrics

### DockQ
DockQ: A Quality Measure for Protein-Protein Docking Models

https://github.com/bjornwallner/DockQ

*  Basu, S. and Wallner, B., 2016. DockQ: a quality measure for protein-protein docking models. PloS one, 11(8), p.e0161879.
*  Mirabello, C. and Wallner, B., 2024. DockQ v2: Improved automatic quality measure for protein multimers, nucleic acids, and small molecules. bioRxiv, pp.2024-05.  

```python
pip install DockQ
DockQ examples/1A2K_r_l_b.model.pdb examples/1A2K_r_l_b.pdb
```


### TM-Score

```c++
wget https://zhanggroup.org/TM-score/TMscore.cpp
g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp
TMscore model.pdb native.pdb
```

## Acknowledgement
```
The code is based on the following repositories:
https://github.com/bjing2016/mdgen.git

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

## Citation

*  Kai Xu, Jianmin Wang, Mingquan Liu, Kewei Zhou, Shaolong Lin, Weihong Li, Lin Shi, Peng Zhou, Huanxiang Liu, and Xiaojun Yao. **"EEfficient Generation of Protein and Protein–Protein Complex Dynamics via SE(3)-Parameterized Diffusion Models."** Journal of Chemical Information and Modeling; doi: https://doi.org/10.1021/acs.jcim.5c01971
