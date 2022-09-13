# Deep Learning on Google Colab 

#### Update (Sept. 2022)
There are a few ways to train on Colab and the steps you follow will depend on what you want to do. 
Here I will detail the following steps:

    A. Setup a GPU Colab environment

    B. Connect your Google drive to Colab

    C. Train your model

    D. Visualize your results using TensorBoard

**Ok, let's get to it!**


## A. Setup Colab GPU Environment 
1. [Go to Colab's welcome page](https://colab.research.google.com/notebooks/welcome.ipynb)
2. Click on: 
    > File > New notebook
3. With your mouse, click on '_Untitled..._' at the top-right and enter an appropriate name for the script.
4. On the menu-bar select: 
    > Runtime > Change runtime type >
    
    select GPU in the **Hardware accelerator** dropdown
5. Click **SAVE**
6. On the top-right menu bar click **CONNECT** (connection is indicated by a green check)
7. Colab already has the latest tensorflow-gpu v2.8.2, keras v2.8.0, and torch v1.12.1 preinstalled.
    + To verify tensorflow installation, for example, enter command: ```!pip show tensorflow ```


## B. Connect your Google Drive to Colab
1. Follow this [video guide](https://youtu.be/PA1WPr0o_DY) to mount your IIT Google Drive on Colab


## C. Training and Saving Your Model
1. Assuming your python script is in ```drive/MyDrive/AS0/tf_sample_cnn.py```, the following command runs the script:
```!python drive/MyDrive/AS0/tf_sample_cnn.py```
2. When you save models, model weights, pickle files (etc.) on Colab, they will be written to your mounted Google Drive.


## Here are some useful resources:
1. [YouTube Video for Colab Bignner](https://www.youtube.com/watch?v=4BVpzY6prJ0)
2. [Use TensorBoard on Colab](
    https://colab.research.google.com/drive/1afN2SALDooZIHbBGmWZMT6cZ8ccVElWk#scrollTo=b0wdo5o8dyzm)
3. [Another Guide to Run TensorBoard in Colab](
    https://www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/)
