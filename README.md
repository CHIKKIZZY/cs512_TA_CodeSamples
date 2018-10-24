# cs512_TA_CodeSamples
Code Samples for Computer Vision (cs512) Assignments

# AS4: TensorFlow Estimator Code Sample
You will find in AS4 directory a python script tf_sample_cnn.py. 
This script is a simple implement of a 2 layer CNN using TensorFlow Estimator API.
The CNN model is trained on the MNIST dataset for 5 epochs. 
Evaluation of the model is done every epoch.

There are few other things that the script does such as:
1. Distributed training on all available GPUs. See line 317 and 204
2. Uses TensorBoard naming convention for visualizing the model's graph
3. Uses tf.summary() to save and visualize input images, histograms, and scalar plots of metrics such as accuracy and loss

# Run Script Instruction
Dependencies: You should have the latest version(v11.1.0) of tensorflow/tensorflow-gpu installed.

To run the script use command: python tf_sample_cnn.py

Depending on your system's resources this script should run any where from 5 minutes to 1 hr. 
Training on my laptop with 16GB memory, Core i7 8th generation 2.20 GHz, and NVIDIA GeForce GTX 1050 Ti GPU takes about 4 minutes on average.

# Note
This sample code is intended as a bootstrap for AS4. 
You may and should modify the code to perform the task required of you in AS4. 
But you have to understand this sample code and know exactly what each line of code is doing before you alter it.
Hence, I strongly suggest you study the code and read TensorFlows' documentation if you want to better understand a particular function.


Happy Learning!