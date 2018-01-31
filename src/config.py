"""Module contains global configuration flags and data directories."""
import os


# source directory, the directory of this configuration file
SRC_DIR = os.path.dirname(os.path.realpath(__file__))

# project base directory, the parent dir of the source dir
BASE_DIR = os.path.dirname(SRC_DIR)

# data directory
DATA_DIR = os.path.join(BASE_DIR, 'data')

# deidentified data directory
DEIDENTIFIED_DIR = os.path.join(BASE_DIR, 'deidentified')

# reidentified data directory
REIDENTIFIED_DIR = os.path.join(BASE_DIR, 'reidentified')

# face metadata directory (files containing face locations and dimensions)
FACES_DIR = os.path.join(BASE_DIR, 'faces')

# face metadata path
FACES_PICKLE = os.path.join(FACES_DIR, 'faces.pickle')

# classifier directory
CLF_DIR = os.path.join(BASE_DIR, 'Classifiers')

# frontal haar classifier
FRONTAL_CLF = os.path.join(CLF_DIR, 'haarcascade_frontalface_default.xml')

# profile haar classifier
PROFILE_CLF = os.path.join(CLF_DIR, 'haarcascade_profileface.xml')
