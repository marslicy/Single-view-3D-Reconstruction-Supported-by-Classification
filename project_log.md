### 10. 06

- First meeting: decided our goals, requirements engineering, API design, distributed tasks of coding structures. (Chenyang Li, Jiaping Zhang, Jiaye Yang, Yihao Wang)

### 11. 06

- Initialized gitlab respository. (Chenyang Li)
- Finished basic structures of *inference.py*, *model.py*, *visualization.py*. (Chenyang Li, Yihao Wang)

### 13. 06

- Add *path.py* file for specifying the dataset. (Jiaye Yang)

### 14. 06

- Implemented the *split.txt* for proper split the dataset. (Jiaye Yang)
- Added the required code for loading vox and img data. (Jiaye Yang)

### 15. 06

- Finished implementing dataset and dataloader. (Jiaye Yang)

### 16. 06

- Implemented the model, but with different ConvTranspose3d parameters compared with the baseline model. The baseline model gives the output shape with dimension (53,53,53) (Chenyang Li)
  - Baseline version:reshape to (128, 4, 4, 4), (5, 2), (5, 2), (5, 2) in format (kernel_size, stride)
  - version 1: reshape to (128, 4, 4, 4), (1, 2), (3, 2), (4, 2) in format (kernel_size, stride)
  - version 2: reshape to (8192, 1, 1, 1), (5, 2), (5, 2, 1), (5, 2, 1) in format (kernel_size, stride, (padding))


### 13. 07 

- Implement the new dataset and dataloader for baseline model (Jiaye Yang)
