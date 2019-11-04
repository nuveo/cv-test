# Description

The collection contains offline signature samples. The images were scanned at 600dpi resolution and cropped with a controlled image region where the signature verification is done.

The signature collection for training contains 209 images. The signatures comprise 9 reference signatures by the same writer “A” and 200 questioned signatures. The 200 questioned signatures comprise 76 genuine signatures written by the reference writer in his/her normal signature style; 104 simulated/forged signatures (written by 27 forgers freehand copying the signature characteristics of the reference writer); 20 disguised signatures written by the reference writer. The disguise process comprises an attempt by the reference writer to purposefully alter his/her signature to avoid being identified or for him/her to deny writing the signature. The simulation/forgery process comprises an attempt by a writer to imitate the reference signature characteristics of a genuine authentic author.

The signature collection for testing contains 125 signatures. The signatures comprise 25 reference signatures by the same writer “B” and 100 questioned signatures. The 100 questioned signatures comprise 3 genuine signatures written by the reference writer in his/her normal signature style; 90 simulated signatures (written by 34 forgers freehand copying the signature characteristics of the reference writer); 7 disguised signatures written by the reference writer. All writings were made using the same make of ballpoint pen and using the same make of paper.

# Training Set

## Folder Structure and File Naming

The signatures of the training set are arranged according to the following folder structure:

- **Disguised:** Contains all the disguised signatures of specimen author ‘A’
- **Genuine:** Contains all the genuine signatures of specimen author ‘A’
- **Reference:** Contains the reference signatures of specimen author ‘A’ which can be used for training the classifiers for author ‘A’.
- **Simulated:** Contains all the skilled forgeries for specimen author ‘A’

Note that the naming convention also reveals the type of signature for the training set. For example, D023: disguised signatures, G003: a genuine signature, S001: a forged/simulated signature, and similarly R001: a reference signature. The signatures of the test set are arranged according to the following folder structure:

- **Reference:** Contains the reference signatures of another specimen author ‘B’. These signatures are used to train the classifier for verifying authorship of the specimen author ‘B’.
- **Questioned:** Contains all the signatures for which the task is to verify authorship of the specimen author ‘B’.

# Objective

The candidate should provide a classification method that correctly states whether a signature can be a fraud/genuine/disguised type.

# Important details

- The candidate will provide to us the compiled model (e.g. a H5 file or similar) that can be loaded and used for prediction. Any instruction on how to load/use the provided model is important to our reviewer make the ground-truth comparison.
- We are aware that the task is difficult due to the lack of data. But this is a challenging situation that we evaluate the candidate inventiveness.
- The training process is adviced to be used on the Reference folder only, due to the real-case scenario does not provide a long set of real signatures from an individual. Although this is the preferential procedure to be done, we are flexible to receive other insights that the candidate may have.
- The trend of thoughts used to build the fraud detection model will be also a piece of important information to the reviewers. Therefore make the code and ideas behind it very clear (e.g. in commentaries or even documentation files)
- Please, set your model output to show up the following pattern (as string characters): F -> forged, D -> diguised, G -> genuine
