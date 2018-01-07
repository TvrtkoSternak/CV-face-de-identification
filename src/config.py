"""Module contains global flags and data directories."""
import os


# source directory, the directory of this configuration file
SRC_DIR = os.path.dirname(os.path.realpath(__file__))

# project base directory, the parent dir of the source dir
BASE_DIR = os.path.dirname(SRC_DIR)

# data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

# face metadata directory (files containing face locations and dimensions)
FACES_DIR = os.path.join(BASE_DIR, 'faces')

# classifier directory
CLF_DIR = os.path.join(BASE_DIR, 'Classifiers')

# frontal haar classifier
FRONTAL_CLF = os.path.join(CLF_DIR, 'haarcascade_frontalface_default.xml')

# profile haar classifier
PROFILE_CLF = os.path.join(CLF_DIR, 'haarcascade_profileface.xml')
