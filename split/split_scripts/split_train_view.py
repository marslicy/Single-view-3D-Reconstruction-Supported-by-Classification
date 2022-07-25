import imageio

from configs.path import imgroot

with open("split/train_shape_v2.txt") as file:
    lines = []
    for line in file:
        lines.append(line.rstrip())


def get_image(shapenetid, num):
    imgpath = f"{imgroot}/{shapenetid}/rendering/{num}.png"
    im = imageio.imread(imgpath)
    return im


with open("split/train.txt", "w") as file:
    for line in lines:
        for i in range(20):
            n = str(i).zfill(2)
            file.write(f"{line}/{n}.png")
            file.write("\n")

with open("split/view_val.txt", "w") as file:
    for line in lines:
        for i in range(20, 22):
            n = str(i).zfill(2)
            file.write(f"{line}/{n}.png")
            file.write("\n")

with open("split/view_test.txt", "w") as file:
    for line in lines:
        for i in range(22, 24):
            n = str(i).zfill(2)
            file.write(f"{line}/{n}.png")
            file.write("\n")


# num137 = 0
# num128 = 0
# unknown = 0
# for shapenetid in lines:
#    num = np.random.randint(24)
#    num = str(num).zfill(2)
#    image = get_image(shapenetid, num)
#    if np.shape(image)[0] == 137:
#        num137 += 1
#    elif np.shape(image)[0] == 128:
#        num128 += 1
#    else:
#        unknown += 1
# print(num137, num128, unknown)


# %%
# %%
