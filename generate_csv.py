"""Generate train and test csv for human dataset."""

import os
from random import sample
from random import shuffle

train_dirs = [1, 2, 6, 8, 9, 13, 14, 15, 42, 43, 55, 81, 83, 86, 115, 117, 118]
# test_dirs = [3, 4, 30, 33]

base_dir = '/mnt/sdb1/shared'
target_dirs = train_dirs  # change to test_dirs for test.

# count number of files in each target_dir.
lens = [len([y for y in os.listdir(os.path.join(base_dir, 'depth', str(x),
        'aligned')) if y.endswith('.png')]) for x in target_dirs]

# randomly sample 'k' from each target_dir.
# sampling depths, hence starting from 0.
k = 600
try:
    train_files = [sample(xrange(0, x), k) for x in lens]
except ValueError:
    print('Sample size exceeded population size.')

# create [(depth_path, rgb_path)] only if correctly sampled.
# depths start from 0.png and corresponding rgb from 1.jpg.
trains = [(os.path.join(base_dir, 'depth', str(d), 'aligned', str(f) + '.png'),
          os.path.join(base_dir, 'RGB', str(d), 'crops', str(f+1) + '.jpg'))
          for (d, files) in zip(target_dirs, train_files) for f in files
          if os.path.exists(os.path.join(base_dir, 'depth', str(d), 'aligned',
                            str(f) + '.png')) and
          os.path.exists(os.path.join(base_dir, 'RGB', str(d), 'crops',
                         str(f+1) + '.jpg'))]

# shuffle the bad boys.
shuffle(trains)

with open('train_{}.csv'.format(k), 'w') as output:
    for (depth_name, image_name) in trains:
        output.write("%s,%s" % (image_name, depth_name))
        output.write("\n")
