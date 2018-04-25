# CV-face-de-identification

Adaptive deidentification in images and image sequences


## Description

1. Detect faces in images/video sequences. Use two Viola-Jones detectors: 
   - one for front-side faces
   - other for face profiles
2. Apply adaptive deidentification using Gaussian masks whose parameters depend on the face resolution.
   - make the process reversible by storing the parameters of the filter
   - make it possible to reconstruct original image using stored parameters
3. Evaluate the system:
   - privacy protection level
   - natural look level
   - usability level
   - determine speed of deidentification (sec/frame)
   - determine FP (false positives) and FN (false negatives) in face detection

We should be able to change the first three depending on the criteria we set. The criteria should be determined using a large number of participants and it should give optimal results.


### Viola-Jones face detector

- solves problem of detecting faces in images
- robust - very high detection rate (true-positive rate) & very low false-positive rate
- real time - for practical applications at least 2 frames per second must be processed
- Face detection only (not recognition)
- Four stages of algorithm:
  - Haar Feature Selection
    - Haar features - similar to regularities of face properties 
  - Creating an Integral Image
    - simplifies calculation of sum of pixel
  - Adaboost Training
    - learning algorithm
  - Cascading Classifiers


### Gaussian filter

- implementation using SciPy function [gaussian_filter]( https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter)


## Databases for testing

- [MIT-CBCL Face Recognition Database](http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html)
  - contains images of 10 subjects with test sets of 200 images per subject
  - variations in illumination, pose and background
- [AT&T Database of faces](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
  - images of 40 subjects with 10 images per subject
  - variations in lighting, facial expressions and facial details
  - PGM format
- [VidTIMIT Audio-Video Dataset](http://conradsanderson.id.au/vidtimit/)
  - videos of 43 people reciting short sentences and moving their heads (to the left, right, back to the center, up, then down and finally return to center)
  - broadcast quality digital video camera, each video is a numbered sequence of JPEG images with a resolution of 512 x 384 pixels


## Tools

Python with:
- SciPy
- NumPy
- OpenCV


## Existing solutions
- https://github.com/Simon-Hohberg/Viola-Jones
- https://github.com/btuan/ViolaJones


## References

- [Robust Real-Time Face Detection](http://www.vision.caltech.edu/html-files/EE148-2005-Spring/pprs/viola04ijcv.pdf)
- [Rapid Object Detection using a Boosted Cascade of Simple Features](http://wearables.cc.gatech.edu/paper_of_week/viola01rapid.pdf)
- [Illumination invariant face detection using viola jones algorithm](http://ieeexplore.ieee.org/document/8014571/)
- https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html
- [An Analysis of the Viola-Jones Face Detection Algorithm](http://www.ipol.im/pub/art/2014/104/article.pdf)
- S. Ribaric, N. Pavesic, An Overview of  Face De-identification in Still Images and Videos
- S. Ribaric, A. Ariyaeeinia, N. Pavesic, De-identification for privacy protection in multimedia content: Asurvey


## Authors and contributors

Mirela Gospodinović

Karmela Slačanac

Lukas Stolcman

Tvrtko Stenak

David Lozić


## Licensing information:

MIT


## Credits

Databases used for benchmarking the de-identification algorithm:

http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html

http://conradsanderson.id.au/vidtimit/

