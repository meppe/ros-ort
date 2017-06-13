import os
import random

def generate_train_val_test_split():
    train_val_test_split = [0.25, 0.25, 0.5]
    assert(sum(train_val_test_split) == 1.0)
    data_dir = '/storage/data/nico2017/nico2017'
    image_sets_dir = data_dir + os.sep + 'ImageSets'
    annotations_dir = data_dir + os.sep + 'Annotations'

    for subdir, dirs, files in os.walk(annotations_dir):
        num_samples = len(files)
        all_samples = range(1, num_samples+1)
        random.shuffle(all_samples)
        num_train = int(num_samples * train_val_test_split[0])
        num_val = int(num_samples * train_val_test_split[1])
        num_test = int(num_samples * train_val_test_split[2])

        train_samples = sorted(all_samples[:num_train])
        val_samples = sorted(all_samples[num_train:num_train+num_val])
        test_samples = sorted(all_samples[-num_test:])
        trainval_samples = sorted(train_samples+val_samples)

        f = open(image_sets_dir+'/Main/train.txt', "w")
        for sam in train_samples:
            f.write(str(sam).zfill(6)+'\n')
        f.close()
        f = open(image_sets_dir + '/Main/val.txt', "w")
        for sam in val_samples:
            f.write(str(sam).zfill(6) + '\n')
        f.close()
        f = open(image_sets_dir + '/Main/test.txt', "w")
        for sam in test_samples:
            f.write(str(sam).zfill(6) + '\n')
        f.close()
        f = open(image_sets_dir + '/Main/trainval.txt', "w")
        for sam in trainval_samples:
            f.write(str(sam).zfill(6) + '\n')
        f.close()
        f = open(image_sets_dir + '/Main/all.txt', "w")
        for sam in all_samples:
            f.write(str(sam).zfill(6) + '\n')
        f.close()
        print("done")
