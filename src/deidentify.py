"""
Performs image deidentification - finds all faces
in an image and blurs them to hide the identity.
"""
import os
import argparse
import pickle
from time import time
from datetime import timedelta

import cv2
from tqdm import tqdm

from deidentifier2 import Deidentifier
from config import DATA_DIR, DEIDENTIFIED_DIR, FACES_PICKLE
from config import FRONTAL_CLF, PROFILE_CLF


def deidentify_data(data_dir, output_dir, save_faces=True):
    """
    Runs face deidentification on every image found in the
    data directory and stores the deidentified faces
    to the specified file.

    Args:
        data_dir: The data directory.
        output_dir: Directory where the deidentified images will
                    be stored.
        save_faces: Save face metadata.
                    (dict[image_file]: location and dimensions)
                    This is used for face reidentification.
    """
    frontal_face_cascade = cv2.CascadeClassifier(FRONTAL_CLF)
    profile_face_cascade = cv2.CascadeClassifier(PROFILE_CLF)
    deidentifier = Deidentifier()
    n_errors = 0
    start = time()
    faces_dict = {}

    for root, _, files in tqdm(list(os.walk(data_dir))):
        for image_file in files:
            img = cv2.imread(os.path.join(root, image_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frontals = frontal_face_cascade.detectMultiScale(img, 1.3, 5)
            profiles = profile_face_cascade.detectMultiScale(img, 1.3, 5)

            faces = []
            faces.extend(frontals)
            faces.extend(profiles)
            faces_dict[image_file] = faces

            old_path = os.path.join(root, image_file)
            new_path = old_path.replace(data_dir, output_dir)
            # create the output dir if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # deidentify
            blurred_img = img
            for x, y, w, h in faces:
                try:
                    blurred_img[y:y+h, x:x+w] = \
                        deidentifier.deidentify(blurred_img[y:y+h, x:x+w])
                except ValueError:
                    n_errors += 1
                    print('\nError while deidentifying:', image_file)

            cv2.imwrite(new_path, blurred_img)

    os.makedirs(os.path.dirname(FACES_PICKLE), exist_ok=True)
    with open(FACES_PICKLE, 'wb') as fout:
        pickle.dump(faces_dict, fout)

    print('Finished. Total time:', timedelta(seconds=time() - start),
          ' Total errors:', n_errors)


if __name__ == '__main__':
    """Parses arguments and runs face detection."""
    parser = argparse.ArgumentParser(description='Run face detection.')

    parser.add_argument('-d', '--data_dir', default=DATA_DIR,
                        help='Path to the dataset.')

    parser.add_argument('-o', '--output_dir', default=DEIDENTIFIED_DIR,
                        help='Path to the faces metadata (X, Y, W, H).')

    args = vars(parser.parse_args())

    deidentify_data(**args)
