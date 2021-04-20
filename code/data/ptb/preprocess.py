import glob

with open("./train.conllx", "w") as train_out, open("./dev.conllx", "w") as dev_out:
    for file in glob.glob("./UD_English/*.conllu"):
        with open(file) as data_file:
            for line in data_file:
                if not line.strip().startswith("#") and len(line.strip()) > 0:
                    word_list = line.strip().split()
                    if "-" not in word_list[0]:
                        if "dev" in file:
                            dev_out.write(line)
                        else:
                            train_out.write(line)
                elif len(line.strip()) == 0:
                    if "dev" in file:
                        dev_out.write("\n")
                    else:
                        train_out.write('\n')

with open("./test.conllx", "w") as test_out:
    for file in glob.glob("./UD_Vietnamese/*.conllu"):
        with open(file) as data_file:
            for line in data_file:
                if not line.strip().startswith("#") and len(line.strip()) > 0:
                    word_list = line.strip().split()
                    if "-" not in word_list[0]:
                        test_out.write(line)
                elif len(line.strip()) == 0:
                    test_out.write("\n")
