# CV-face-de-identification

Adaptive deidentification in images and image sequences

Description
---

Use two Viola-Jones detectors (front-side and profile) to detect faces in images and video sequences and apply adaptable deidentification that uses Gaussian mask whose parameters depend on face resolution

The filter should be as reversible as possible so that the original video can be reconstructed when the parameters of the convolution filter are known.

Collaborate with the other project groups in order to propose criteria to evaluate privacy protection level, natural look level and usability level of the video obtained after filter application. Determine the criteria using large number of participants (crowdsourcing). Determine the speed of deidentification (sec/frame). Determine FP and FN for the face detection step.

Authors and contributors
---

Mirela Gospodinović

Karmela Slačanac

Łukasz Stolcman

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

