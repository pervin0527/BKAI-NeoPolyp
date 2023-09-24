import random

def write_file(file_name, file_list):
    with open(f"{dir}/{file_name}.txt", "w") as f:
        for idx, file in enumerate(file_list):
            f.write(file)

            if idx != len(file_list) - 1:
                f.write("\n")


def split_file_list(spliters):
    train_files = []
    valid_files = []

    file_names = ["red", "green", "rng"]
    for idx, spliter in enumerate(spliters):
        with open(f"{dir}/{file_names[idx]}.txt", "r") as f:
            file_list = f.readlines()
        
        file_list = [x.strip() for x in file_list]
        random.shuffle(file_list)
        
        train_files.extend(file_list[: spliter[0]])
        valid_files.extend(file_list[spliter[0] :])

    return train_files, valid_files

if __name__ == "__main__":
    dir = "/home/pervinco/Datasets/BKAI_IGH_NeoPolyp"

    # red_split = [555, 139] ## 694
    # green_split = [206, 51] ## 257
    # rng_split = [39, 10] ## 49

    red_split = [200, 494]
    green_split = [200, 57]
    rng_split = [20, 29]

    train, valid = split_file_list([red_split, green_split, rng_split])
    print(len(train), len(valid))

    write_file("train1", train)
    write_file("valid1", valid)