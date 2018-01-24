"""Module contains functions for face detection."""
import os
import argparse

import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from deidentifier2 import Deidentifier
from config import DATA_DIR, DEIDENTIFIED_DIR
from config import FRONTAL_CLF, PROFILE_CLF

def deidentify_data(data_dir, output_dir):
    """
    Runs face deidentification on every image found in the
    data directory and stores the deidentified faces
    to the specified file.

    Args:
        data_dir: The data directory.
        output_dir: Directory where the deidentified images will
                    be stored.
    """
    frontal_face_cascade = cv2.CascadeClassifier(FRONTAL_CLF)
    profile_face_cascade = cv2.CascadeClassifier(PROFILE_CLF)
    deidentifier = Deidentifier()

    for root, _, files in tqdm(list(os.walk(data_dir))):
        for image_file in files:
            img = cv2.imread(os.path.join(root, image_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frontals = frontal_face_cascade.detectMultiScale(img, 1.3, 5)
            profiles = profile_face_cascade.detectMultiScale(img, 1.3, 5)

            faces = []
            faces.extend(frontals)
            faces.extend(profiles)

            old_path = os.path.join(root, image_file)
            new_path = old_path.replace(data_dir, output_dir)
            # create the output dir if it doesn't exist
            #os.makedirs(os.path.dirname(new_path), exist_ok=True)

            # deidentify image
            img_blurred = img
            for x, y, w, h in faces:
                img_blurred[y:y+h, x:x+w] = deidentifier.deidentify(img_blurred[y:y+h, x:x+w])
                #img_unblurred = img_blurred
                #img_unblurred[y:y+h, x:x+w] = deidentifier.identify(img_unblurred[y:y+h, x:x+w])
            
            cv2.imwrite(new_path, img_blurred)
            #cv2.imwrite(new_path, img_unblurred)


if __name__ == '__main__':
    """Parses arguments and runs face detection."""
    parser = argparse.ArgumentParser(description='Run face detection.')

    parser.add_argument('-d', '--data_dir', default=DATA_DIR,
                        help='Path to the dataset.')

    parser.add_argument('-o', '--output_dir', default=DEIDENTIFIED_DIR,
                        help='Path to the faces metadata (X, Y, W, H).')

    args = vars(parser.parse_args())

    deidentify_data(**args)

    # TODO detect faces in deidentified image and call identify function upon them
