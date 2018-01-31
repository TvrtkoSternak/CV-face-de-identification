"""Module contains functions for data loading."""
import os
from collections import Counter

import numpy as np
import cv2


def get_image_paths(data_dir, min_faces=50):
    """Finds all images in the specified directory of people
    which have at least the minimum number of images specified.

    Args:
        data_dir: Data directory.
        min_faces: Minimum number of images per person.

    Returns:
        image_files ([str]): Image paths
        labels ([int]):
        people ([str]):
    """
    image_files = []
    labels = []
    people = set()
    counter = Counter()

    # find face counts of every person
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                person = '_'.join(file.split('_')[:-1])
                counter[person] += 1

    for person, count in counter.items():
        if count >= min_faces:
            people.add(person)
    people = list(people)

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jpg"):
                person = '_'.join(file.split('_')[:-1])
                if counter[person] >= min_faces:
                    image_files.append(os.path.join(root, file))
                    labels.append(people.index(person))

    return image_files, labels, people

def load_faces(image_files, w, h, greyscale=True):
    """Loads the cropped images from the specified image paths.

    Args:
        image_files: Paths to images.
        w: Crop width.
        h: Crop height.
        greyscale: Loads as greyscale.
    """
    X = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if greyscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        y1 = int(img.shape[0]/2 - h/2)
        y2 = int(img.shape[0]/2 + h/2)
        x1 = int(img.shape[1]/2 - w/2)
        x2 = int(img.shape[1]/2 + w/2)
        crop = img[y1:y2, x1:x2]
        X.append(crop)

    return np.array(X)

def get_blurred(data_dir, blurred_dir, image_files):
    """Returns the blurred versions of the specified image_files.

    If the blurred equivalent of the original image does not exist,
    the original image is placed at that index instead.

    Args:
        data_dir: Original (untouched) data directory.
        blurred_dir: Directory to the deidentified images.
        image_files: The paths to the original image files.
    """
    out = []
    blurred_images = [old_path.replace(data_dir, blurred_dir)
                      for old_path in image_files]
    for i, file_path in enumerate(blurred_images):
        if os.path.exists(file_path):
            out.append(file_path)
        else:
            out.append(image_files[i])

    return out
