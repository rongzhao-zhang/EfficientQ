# EfficientQ
Official implementation of the paper "EfficientQ: An efficient and accurate post-training neural network quantization method for
medical image segmentation" in Medical Image Analysis.

# Dataset preparation
Specify the dataset path by passing the `--data_dir` argument to the script. The `data_dir` should be in the following format:
```
data_dir
    ├── seg
    ├── ct
```
for LiTS or 
```
data_dir
    ├── seg
    ├── flair
    ├── t1
    ├── t1ce
    ├── t2
```
for BRATS.

Pass `--split_dir` argument to the script to specify the split of the dataset. The `split_dir` should be in the following format:
```
split_dir
    ├── round1
    │   ├── train.txt
    │   ├── val.txt
    ├── round2
    │   ├── train.txt
    │   ├── val.txt
    ...
```
The data should be preprocessed to zero-mean and unit variance. The data should be in the format of `.npy` files.

Or you can modify the `get_data_cube()` function in `definer.py` and files in `dataloader` to load the dataset in your own way.

The `data_dir` and `split` can also be specified in the configuration file under `config`.

# Training
To train the FP model, run the following command (for BraTS):
```
cd src
python entrance.py train_fp --round 1 --config ../config/brats_fp.yaml --data_dir /path/to/your/data --split_dir /path/to/your/split --round X
```
Once the FP model is trained, you can train the 2-bit PTQ model by running the following command:
```
cd src
python entrance.py ptq --qlvl_w 4 --qlvl_a 4 --round 1 --config ../config/brats_ptq.yaml  --pretrain /path/to/pretrain --data_dir /path/to/your/data --split_dir /path/to/your/split
```
Other bitwidths can be specified by changing the `qlvl_w` and `qlvl_a` arguments.

# Citation
If you find this code useful, please consider citing:
```
@article{zhang2024efficientq,
  title={EfficientQ: An efficient and accurate post-training neural network quantization method for medical image segmentation},
  author={Zhang, Rongzhao and Chung, Albert CS},
  journal={Medical Image Analysis},
  pages={103277},
  year={2024},
  publisher={Elsevier}
}
```

