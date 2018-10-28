# Deep Learning on Google Colab 
There are a couple of ways to train on Colab and the steps you follow will depend on what you want to do. 
Here I will detail the steps following:

    A. Setup a GPU Colab environment

    B. Connect your Google drive to Colab

    C. Train your model

    D. Visualize your results using TensorBoard

**Ok, let's get to it!**


## A. Setup Colab GPU Environment 
1. [Go to Colab's welcome page](https://colab.research.google.com/notebooks/welcome.ipynb)
2. Click on: 
    > File > New Python 3 Notebook
3. When the notebook loads up click on: 
    > Runtime > Change runtime type >
    
    select Python 3 for **Runtime type** and GPU for **Hardware accelerator**
4. Click **SAVE**
5. On the top-right menu bar click **CONNECT** (connection is indicated by a green check)
6. Colab already has tensorflow-gpu v1.12.0rc1 preinstalled.
    My sample code can run on this version so you don't have to downgrade to v1.11
    + To verify tensorflow installation enter command: ```!pip show tensorflow ```
    + You may install keras using commands: ```!pip install keras```


## B. Connect your Google Drive to Colab
1. In your google drive create folder ```/apps```
2. Copy over the sample code to the folder
3. Copy, Paste, and Run the following code in the notebook to install all needed packages to link Colab to your Drive 
   (For first time users only)
    ```
    !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
    !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
    !apt-get update -qq 2>&1 > /dev/null
    !apt-get -y install -qq google-drive-ocamlfuse fuse
    from google.colab import auth
    auth.authenticate_user()
    from oauth2client.client import GoogleCredentials
    creds = GoogleCredentials.get_application_default()
    import getpass
    !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
    vcode = getpass.getpass()
    !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
    ```
4. You will be shown a google link. Click on the link to grant colab access to your Google drive
5. You will be redirected to a page which displays a key, copy the key into the box and press ```Shift + Enter```
6. You may have to repeat Steps B-4 and B-5 if the server times out
7. Create a folder in the colab environment and link it to Google Drive using the commands:
   ```!mkdir -p drive !google-drive-ocamlfuse drive```
8. You can verify Colab is linked to your drive using the command: ```!ls drive/apps/```
9. Step B-8 should display the sample code you copied over to your ```/apps``` folder in Step B-2


## C. Train Your Model
Assuming the sample code is named tf_sample_cnn.py, the following command runs the script:
```!python drive/apps/tf_sample_cnn.py```


## D. Visualize Your Results with TensorBoard

You could either download the saved model to your local drive and run TensorBoard directly on your computer 
or you could do it directly on Colab. I will illustrate both options.

### i. Download Saved Model
By default Colab saves the model in ```/content```. In my sample code I specified the model directory 
as ```'cnn_model'```. Hence it will be located at ```/content/cnn_model```
1. To locate where your model is saved run the command: ```!find / -type d -name "name_of_model"```
2. Use the following command to move the model to your drive: ```!mv /content/cnn_model drive/apps```
3. Step D-i-2 assumes your model directory is named ```'cnn_model'``` 
   and moves the model to your Drive's ```/apps``` folder
4. In you Drive, navigate to the ```/apps``` folder and download the model: 
   >right-click on the model's folder > Download
5. Unzip the downloaded file
6. Open the terminal and activate the environment where you have installed TensorFlow/TensorBoard on your computer
7. Use the command to load your model: ```tensorboard --logdir=/path/to/unzipped/model/dir```
8. Open your browser(preferably google chrome) and go to: ```localhost:6006```

### ii. Run TensorBoard Directly from Colab
1. Download and unzip ngrok using the following commands:
    ```
    !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
    !unzip ngrok-stable-linux-amd64.zip
    ```
2. Get TensorBoard to run in the background and tell ngok where to log your model using the command:
    ```
    LOG_DIR = './log'             
    get_ipython().system_raw(   
        'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'   
        .format(LOG_DIR)
    )
    ```
3. Launch ngrok background process with the command: ```get_ipython().system_raw('./ngrok http 6006 &')```
4. Retrieve the public url where ngrok pushes TensorBoard outputs using the command:
    ```
    ! curl -shttp://localhost:4040/api/tunnels | python3 -c \   
        "import sys,json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
    ```
5. Retrain the model again as was done in Step C 
6. After the training is done click on the link that was displayed when you ran Step D-ii-4


**That's it!**

## Here are some useful resources:
1. [YouTube Video for Colab Bignner](https://www.youtube.com/watch?v=4BVpzY6prJ0)
2. [Use TensorBoard on Colab](
    https://colab.research.google.com/drive/1afN2SALDooZIHbBGmWZMT6cZ8ccVElWk#scrollTo=b0wdo5o8dyzm)
3. [Another Guide to Run TensorBoard in Colab](
    https://www.dlology.com/blog/quick-guide-to-run-tensorboard-in-google-colab/)
