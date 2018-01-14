"""Module contains functions for face detection."""
import os
import pickle
import argparse

import numpy as np
import cv2
from tqdm import tqdm

from config import DATA_DIR, BASE_DIR, FACES_DIR, FRONTAL_CLF, PROFILE_CLF


# text file containing face metadata (locations and dimensions)
FACES_TXT = 'faces.txt'

# binary pickle file containing face metadata
FACES_PICKLE = 'faces.pickle'


def detect_faces(data_dir, output_dir):
    """
    Runs face detection on every image found in the
    data directory and stores the face location and dimensions
    to the specified file. 
    
    Args:
        data_dir: The data directory.
        output_file: Directory where the files which contain
                     face locations and dimensions are generated.

    Returns:
        faces_dict: Dictionary mapping image file to the list 
                    of face locations and dimensions.
                    e.g. {'Harrison_Ford_0005.jpg': [66 68 119 119]}
    """
    frontal_face_cascade = cv2.CascadeClassifier(FRONTAL_CLF)
    profile_face_cascade = cv2.CascadeClassifier(PROFILE_CLF)
    faces_dict = {}

    # create the output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, FACES_TXT), 'w') as fout:
        for root, dirs, files in tqdm(list(os.walk(data_dir))):
            for image_file in files:
                img = cv2.imread(os.path.join(root, image_file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                frontals = frontal_face_cascade.detectMultiScale(gray, 1.3, 5)
                profiles = profile_face_cascade.detectMultiScale(gray, 1.3, 5)

                faces = frontals # + profiles
                faces_dict[image_file] = faces

                for x, y, w, h in faces:
                    print(image_file, x, y, w, h, file=fout)

    with open(os.path.join(output_dir, FACES_PICKLE), 'wb') as fout:
        pickle.dump(faces_dict, fout)

    return faces_dict
   

if __name__ == '__main__':
    """Parses arguments and runs face detection."""
    parser = argparse.ArgumentParser(description='Run face detection.') 

    parser.add_argument('-d', '--data_dir', default=DATA_DIR,
                        help='Path to the dataset.')

    parser.add_argument('-o', '--output_dir', default=FACES_DIR,
                        help='Path to the faces metadata (X, Y, W, H).')

    args = vars(parser.parse_args())

    detect_faces(**args)
