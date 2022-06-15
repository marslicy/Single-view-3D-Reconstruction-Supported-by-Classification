import os

import numpy as np

from configs.path import voxroot

"""
primary goal: save all the cateid/shapeid into a txt

1. randomly choose some in the list as the val&test set
"""


# we first split the shape to train and test
train_list = []
test_list = []
val_list = []
overfit_list = []
for cat in os.listdir(voxroot):
    sub_train_list = []
    sub_test_list = []
    sub_val_list = []
    sub_overfit_list = []
    if os.path.isdir(os.path.join(voxroot, cat)):
        # mylist.append(file)
        for shape in os.listdir(os.path.join(voxroot, cat)):
            dice = np.random.choice(np.arange(0, 3), p=[0.6, 0.2, 0.2])
            if not dice:
                sub_train_list.append(f"{cat}/{shape}")
            elif dice == 1:
                sub_val_list.append(f"{cat}/{shape}")
            else:
                sub_test_list.append(f"{cat}/{shape}")
            # randomly sample 10 from current category for the overfitting

        sub_overfit_list = np.random.choice(sub_train_list, 10)

        train_list.append(sub_train_list)
        val_list.append(sub_val_list)
        test_list.append(sub_test_list)
        overfit_list.append(sub_overfit_list)


with open("split/train_shape.txt", "w") as file:
    for ls in train_list:
        for item in ls:
            file.write(item)
            file.write("\n")

with open("split/test_shape.txt", "w") as file:
    for ls in test_list:
        for item in ls:
            file.write(item)
            file.write("\n")

with open("split/val_shape.txt", "w") as file:
    for ls in val_list:
        for item in ls:
            file.write(item)
            file.write("\n")

with open("split/overfit_shape.txt", "w") as file:
    for ls in overfit_list:
        for item in ls:
            file.write(item)
            file.write("\n")

# %%
"""
goal:
1. select random shape for val&test. Portion should be 0.4 at each category
2. from the training set, select random view for val&test
"""


# randint = np.random.choice(np.arange(0, 2), p=[0.4, 0.6])
#
# ls = []
# for i in range(10):
#    randint = np.random.choice(np.arange(0, 2), p=[0.4, 0.6])
#    ls.append(randint)
#
# print(ls)
#
# for i in ls:
#    if not i:
#        print("not")

# %%


# %%
