# CV-face-de-identification

Description
---

Setup
----
Use pip to install the project requirements:

```pip3 install -r requirements.txt```

To download the data simply run the ```get_data.sh``` bash script:

```bash get_data.sh```

or download the data from http://vis-www.cs.umass.edu/lfw/lfw.tgz and unzip it into the ./data directory.
The data path should look like this: ```.../CV-face-de-identification/data/lfw/...```

Usage
----
To **deidentify** the images from the data directory, run the following command:

```python3 src/deidentify.py```

This will create a deidentified directory in the project directory containing the deidentified images.
The script uses defaults specified in the ```src/config.py``` file.
For more details, run:

```python3 src/deidentify.py --help```

To **evaluate** the deidentification process, run the ```src/face_recognition.ipynb``` notebook.

To **re-identify** the deidentified images, run the following script:

```python3 src/reidentify.py```

This will create a reidentified directory in the project directory containing the reidentified images.
For more details, run:

```python3 src/reidentify.py --help```

To check if everything went well, run the ```src/Playground.ipynb``` notebook.

Authors and contributors
---

Mirela Gospodinović

Karmela Slačanac

Lukas Stolcman

Tvrtko Stenak

David Lozić

Licensing information: READ LICENCE
---

Credits
---

Databases used for benchmarking the de-identification algorithm:

http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html

http://conradsanderson.id.au/vidtimit/

