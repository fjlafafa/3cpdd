# 3CPDD

## Code base
our code is based on [ABdockgen](github.com/wengong-jin/abdockgen)

## Dependencies
 
* PyTorch 
* tqdm
* sidechainnet
* prody
* [torchsparse](https://github.com/mit-han-lab/torchsparse/tree/v2.0.0)

## 3D convolution


## Data
Data for testing the basic functions are in folder data

To download the pretraining data set please run, this will take about 1 hour and generate a jsonl file with size 5G
```
python pdbfile/pretrain_data
```

## Training

training script can be launched by

```
python dock_train.py --L_target 20 --save_dir ckpts/HERN-dock
python dock_train.py --L_target 20 --sparse_encoder --save_dir ckpts/HERN-dock
```

```
mkdir outputs
python predict.py ckpts/HERN-dock-sparse/model.best data/rabd/test_data.jsonl
```

It will produce a PDB file for each epitope in the test set.

## Context given paratope Design

The training script can be launched by

```
python lm_train.py --L_target 20 --save_dir ckpts/HERN-gen
python lm_train.py --L_target 20 --sparse_encoder --save_dir ckpts/HERN-gen
```

At test time, we can generate new CDR-H3 paratopes specific to a given epitope:

```
python generate.py ckpts/HERN_gen.ckpt data/rabd/test_data.jsonl 1 > results/HERN.txt
```

The above script will generate one CDR-H3 sequence per epitope. You can sample more candidates by changing this parameter.
