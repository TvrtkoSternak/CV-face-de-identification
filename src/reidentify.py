"""
Performs re-identification on deidentified images.
(reverse process of the default deidentification functionality)
"""
import os
import argparse
import pickle
from time import time
from datetime import timedelta

import cv2
from tqdm import tqdm

from deidentifier2 import Deidentifier
from config import DEIDENTIFIED_DIR, REIDENTIFIED_DIR, FACES_PICKLE
from config import FRONTAL_CLF, PROFILE_CLF


def reidentify_data(data_dir, output_dir):
    """
    Runs face re-identification on every image found in the
    data directory and stores the re-identified faces
    to the specified file.

    Args:
        data_dir: The deidentified image data directory.
        output_dir: Directory where the re-identified images will
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
    found_faces = False

    if os.path.exists(FACES_PICKLE):
        with open(FACES_PICKLE, 'rb') as f:
            faces_dict = pickle.load(f)
            found_faces = True
    else:
        print("Didn't find face metadata. (face locations and dimensions)")

    for root, _, files in tqdm(list(os.walk(data_dir))):
        for image_file in files:
            img = cv2.imread(os.path.join(root, image_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if found_faces:
                faces = faces_dict[image_file]
            else:
                frontals = frontal_face_cascade.detectMultiScale(img, 1.3, 5)
                profiles = profile_face_cascade.detectMultiScale(img, 1.3, 5)

                faces = []
                faces.extend(frontals)
                faces.extend(profiles)

            old_path = os.path.join(root, image_file)
            new_path = old_path.replace(data_dir, output_dir)
            # create the output dir if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # reidentify image
            unblurred_img = img
            for x, y, w, h in faces:
                try:
                    unblurred_img[y:y+h, x:x+w] = \
                        deidentifier.identify(unblurred_img[y:y+h, x:x+w])
                except ValueError:
                    n_errors += 1
                    print('\nError while reidentifying:', image_file)

            cv2.imwrite(new_path, unblurred_img)

    print('Finished. Total time:', timedelta(seconds=time() - start),
          ' Total errors:', n_errors)


if __name__ == '__main__':
    """Parses arguments and runs face detection."""
    parser = argparse.ArgumentParser(description='Run face detection.')

    parser.add_argument('-d', '--data_dir', default=DEIDENTIFIED_DIR,
                        help='Path to the deidentified (input) dataset.')

    parser.add_argument('-o', '--output_dir', default=REIDENTIFIED_DIR,
                        help='Path to the reidentified (output) dataset.')

    args = vars(parser.parse_args())

    reidentify_data(**args)
