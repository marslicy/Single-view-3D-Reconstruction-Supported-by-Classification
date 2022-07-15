import os
from pathlib import Path

"""
Goals
    1. load the old train.txt split, read it as dictionary, exclude 3 categories.
    2. Do the same for the shape_val and shape_test as well.
    3. Merge the excluded part as a new split called new_categories.txt
"""

# list out all the object
train_txt = "split/train_shape_v2.txt"

shape_val_txt = "split/val_shape_v2.txt"

shape_test_txt = "split/test_shape_v2.txt"


def seperate_split(file):

    split = Path(file).read_text().splitlines()

    shape_dict = {}  # if the category not exists, create key and empty list, add the

    for line in split:
        category, shape = line.split("/")
        if str(category) not in shape_dict.keys():
            shape_dict[str(category)] = []
            shape_dict[str(category)].append(shape)
        else:
            shape_dict[str(category)].append(shape)

    # manually selected 3 category to exclude
    exclude_categories = [
        "02828884",
        "02933112",
        "03636649",
        "04090263",
        "04256520",
        "04530566",
    ]

    for category in shape_dict.keys():
        if category not in exclude_categories:
            pass

    old_dict = shape_dict

    new_dict = {}

    # add new_dict
    for cat in exclude_categories:
        new_dict[cat] = shape_dict[cat]
        old_dict.pop(cat)

    return old_dict, new_dict


train, new1 = seperate_split(train_txt)
shape_val, new2 = seperate_split(shape_val_txt)
shape_test, new3 = seperate_split(shape_test_txt)


# define a function to write split txt from dictionary
def write_split(split_dict, name):

    # define the directory to save the txt
    save_dir = "."

    file = f"{save_dir}/{name}"

    # create the file if not exists
    os.system(f"touch {file}")

    # write in the format of category_id/shape_id for each line
    for key in split_dict.keys():
        with open(Path(file), "a") as f:
            for item in split_dict[key]:
                f.write(f"{key}/{item}")
                f.write("\n")


write_split(train, "split/train.txt")

write_split(shape_val, "split/shape_val.txt")

write_split(shape_test, "split/shape_test.txt")

# merge the new category from previous splits new1 new2 new3

new_categories = {}

news = [new1, new2, new3]

# iteratively append
for new in news:
    for key in new.keys():
        # if the key not exists, add the key and value
        if key not in new_categories.keys():
            new_categories[key] = new[key]
        # else append the value
        else:
            new_categories[key] = new_categories[key] + new[key]

write_split(new_categories, "split/new_categories.txt")
