 

## MPI-INF-3DHP

We heavily borrowed code from [P-STMO](https://github.com/paTRICK-swk/P-STMO) to train and evaluate our model on MPI-INF-3DHP.

### Data Preparation

Please follow the instructions from the [original repo](https://github.com/paTRICK-swk/P-STMO?tab=readme-ov-file#mpi-inf-3dhp) to prepare the data.

### Train

Use the following command to train our 81-frame model:

```
python run_3dhp.py -f 81 -frame-kept 9 -coeff-kept 9 -b 512 --train 1 --lr 0.0007 -lrd 0.97 -c CKPT_NAME --gpu 0
```

### Evaluation

A simple evaluation can be done by running:

```
python run_3dhp.py -f 81 -frame-kept 9 -coeff-kept 9 --reload 1 --previous_dir checkpoint/no_refine_39_2758.pth
```

Our checkpoint is released [here,](https://drive.google.com/file/d/1b-CGzqhiKICL_fNEgE7CKgSu0vW3SyQL/view?usp=drive_link) and we assume you placed this file under `checkpoint/no_refine_39_2758.pth`. For more metrics (e.g., PCK), please follow the instructions in the [original repo](https://github.com/paTRICK-swk/P-STMO?tab=readme-ov-file#mpi-inf-3dhp).
