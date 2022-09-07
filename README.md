Tasks:
1. Explore the CPU & GPU-based installation method of the Keras library.
2. Import the inbuilt MNIST dataset through the Keras library & perform the following operation on classification:
a) Prepare sequential & functional model for above dataset.
b) Perform different operations on the above model like compile, fit, evaluate, predict, etc.
c) Save the model & weight and load the model & weight for prediction.


Observations:

1.	Explore the CPU & GPU-based installation method of the Keras library..

•	Firstly, check the system requirements.

•	Install and setup anaconda. Then, create an conda environment suppose tf
 conda create --name keras python=3.9

•	GPU setup
First install the
[NVIDIA GPU driver](https://www.nvidia.com/Download/index.aspx){:.external} 
if you have not. You can use the following command to verify it is
installed

	     nvidia-smi


•	Then install CUDA and cuDNN with conda.
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

•	Then install keras
pip install keras


•	Then verify the CPU and GPU setup.

               Now, you can start working with keras

2.	Import the inbuilt MNIST dataset through the Keras library & perform the following operation on classification:

MNIST is ‘Modified National Institute of Standards and Technology. This dataset consists of handwritten digits from 0 to 9 and it provides a pavement for testing image processing systems.

Loading MNIST Dataset
Using Python and its vast inbuilt modules it has the MNIST Data in the keras.datasets module. So, we don’t need to externally download and store the data. So, from keras.datasets module we import the mnist function which contains the dataset.
Then the data set is stored in the variable data using the mnist.load_data() function which loads the dataset into the variable data. We know that the mnist dataset contains handwritten digit images, stored in the form of tuples.

Splitting the dataset into train and test
We directly split the dataset into train and test. So for that, we initialize four variables X_train, y_train, X_test, y_test to sore the train and test data of dependent and independent values respectively.

Reshaping the data
Now, we have to reshape in such a way that we have we can access every pixel of the image. The reason to access every pixel is that only then we can apply deep learning ideas and can assign color code to every pixel. Then we store the reshaped array in X_train, X_test respectively.

Training the model
To perform Model building we have to import the required functions i.e. Sequential and Dense to execute Deep Learning which is available under the Keras library.

But this is not directly available for which we need to understand this simple line chart: 
1) Keras -> Models -> Sequential 
2) Keras -> Layers -> Dense

Then we store the function in the variable model as it makes it easier to access the function every time instead of typing the function every time, we can use the variable and call the function. Then convert the image into a dense pool of layers and stack each layer one above the other and we use ‘relu’ as our activation function. Then again, we stack a few more layers with ‘softmax’ as our activation function.

Compiling
Then finally we compile the entire model and use cross-entropy as our loss function, to optimize our model use adam as our optimizer and use accuracy as our metrics to evaluate our model.

Summary
To get an overview of our model we use ‘model.summary()’, which provides brief details about our model.

Fitting the model
This is the penultimate step where we are going to train the model with just a single line of code. So for that, we are using the .fit() function which takes the train set of the dependent and the independent and dependent variable as the input, and set epochs = 10, and set batch_size as 100. 

Train set - X_train; y_train 

Epochs - An epoch means training the neural network with all the training data for one cycle. An epoch is made up of one or more batches, where we use a part of the dataset to train the neural network. Meaning we send the model to train 10 times to get high accuracy. You could also change the number of epochs depending on how the model performs. 

Batch_size - Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration. So basically, we send 100 images to train as a batch per iteration.

Predicting Accuracy
So to know how well the model works in the testing dataset I use the scores variable to store the value and use the .evaluate() function which takes the test set of the dependent and the independent variables as the input. This computes the loss and the accuracy of the model in the test set. As we are focused on accuracy we print only the accuracy.

Saving model and weights
We use the model.save() function to save the model and model.save_weights() to save the weights. We can also use the model.save.load() function in order to load the saved model.
