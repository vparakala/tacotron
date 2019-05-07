import os
from fnmatch import fnmatch

metadata = open('metadata.csv', 'w')
root = '/mnt/disks/data/VCTK-Corpus/txt'
file_type = '*.txt'
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, file_type):
            with open(os.path.join(path, name), 'r') as f:
                utterance = f.readlines()
            metadata.write('{}|{}|{}\n'.format(name[:-4].strip(), utterance[0].strip(), utterance[0].strip()))
metadata.close()        


