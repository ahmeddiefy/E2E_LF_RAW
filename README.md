# E2E_LF_RAW
End-to-End Residual Network for Light Field Reconstruction on Raw Images and View Image stacks

Light Field (LF) technology has become a focus of great interest for its use in many applications, especially since the introduction of the consumer LF camera, 
which facilitated the acquisition of dense LF images. Obtaining dense LF images is costly due to the trade-off that exists between spatial resolution and angular resolution.
Accordingly, in this research, we suggest a learn-ing-based solution to this challenging problem and reconstruct dense, high-quality LF images.
Rather than using several images of the same scene, we trained our model using raw LF images (lenslet images).
The raw LF format enables the encoding of several images of the same scene into a single image.
As a consequence, it allows the network to comprehend and model the connection between various images, resulting in higher quality.
Our model is divided into two sequential modules: LF Reconstruction (LFR) and LF Augmentation (LFA).
Each module is modeled using a residual network based on a convolutional neural network (CNN).
End-to-end, our network was trained to reduce the total of absolute errors between the reconstructed and ground truth images.
Experimental findings on three real-world datasets show that our suggested method has great performance and superiority over state-of-the-art approaches.

For training:
1- Download training data from: https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/
 and save the training data at ./data/TrainingData/Training/{Dataset Folder}

2- Create these directories:
  a) checkpoints: for saving the trained weights
  b) results/numpy: to save the reconstructed LFs
  c) data/train: for saving the training patches
  d) data/test/Set x: for saving the testing patches
  
3- run one of these Matlab files to generate training patches
  a) aug_train_D2S_2_8_Extra_0.m for task1: 2 × 2 - 8 × 8 Extrapolation 0
  b) aug_train_D2S_2_8_Extra_1.m for task2: 2 × 2 - 8 × 8 Extrapolation 1
  c) aug_train_D2S_2_8_Extra_1.m for task3: 2 × 2 - 8 × 8 Extrapolation 2

4- Choose one of the three models available for training and rename it to --> MODEL.py and then run the LF_E2E.py to start the training
  a) MODEL_task1.py for task1: 2 × 2 - 8 × 8 Extrapolation 0
  b) MODEL_task2.py for task2: 2 × 2 - 8 × 8 Extrapolation 1
  c) MODEL_task3.py for task3: 2 × 2 - 8 × 8 Extrapolation 2

 
For testing:
1- Download testing data from: 
  a) 30 Scenes dataset: https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/
  b) Reflective and occlusion datasets: http://lightfields.stanford.edu/LF2016.html
and save the test data at ./data/TrainingData/Test/{Dataset Folder}

2- run one of these Matlab files to generate testing patches
  a) aug_test_D2S_Extra_0.m: task1: 2 × 2 - 8 × 8 Extrapolation 0
  b) aug_test_D2S_Extra_1.m: task2: 2 × 2 - 8 × 8 Extrapolation 1
  c) aug_test_D2S_Extra_2.m: task3: 2 × 2 - 8 × 8 Extrapolation 2
  
3- Then, run TEST.py to test the model and he reconstructed LFs will be saved at results/numpy

4- run one of these Matlab files to compare the reconstructed LFs with the ground-truth:
  a) calculate_PSNR_2_8_D2S_Extra_0.m: task1: 2 × 2 - 8 × 8 Extrapolation 0
  b) calculate_PSNR_2_8_D2S_Extra_1.m: task2: 2 × 2 - 8 × 8 Extrapolation 1
  c) calculate_PSNR_2_8_D2S_Extra_2.m: task3: 2 × 2 - 8 × 8 Extrapolation 2
  
5- Finally, to generate RGB images from the luminance images run this Matlab file: save_rgb_image.m

