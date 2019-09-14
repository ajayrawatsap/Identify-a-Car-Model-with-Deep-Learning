# Identify-a-Car-Model-with-Deep-Learning
Explore how to practice real world Data Science by collecting data, curating it and apply advance Deep Learning techniques to
create high quality models which can be deployed in Production.

# Data
- There are 4000 images of two of the popular cars (Swift and Wagonr) in India of make Maruti Suzuki with 2000 pictures belonging to
each model. The data is divided into training set with 2400 images , validation set with 800 images and test set with 800 images. 
- The data was randomized before splitting into training, test and validation set.
- The data was collected using a web scraper written in python. Selenium Library was used to load the full HTML page and then
Beautifulsoup library was used to extract and download images from HTML tags
- The data is hosted at [Kaggle](https://www.kaggle.com/ajaykgp12/cars-wagonr-swift) which can be downloaded or you can 
also create a notebook directly on Kaggle and access data in your notebook.
![data_image](https://github.com/ajayrawatsap/Identify-a-Car-Model-with-Deep-Learning/blob/master/resources/cover_image.PNG)

# Context
- Data science beginners often start with curated set of data, but it's a well-known fact that in a real Data Science Project, major time is spent on collecting, cleaning and organizing data.  Also, domain expertise is considered as important aspect of creating good ML models. 
- Being an automobile enthusiast, I took up this challenge to collect images of two of the popular car models from a used car website, where users upload the images of the car they want to sell and then train a Deep Neural Network to identify model of a car from car images. 
- In my search for images I found that approximately 10 percent of the car pictures did not represent the intended car correctly and those pictures must be deleted from final data.

# Topics that would be covered 
We will explore all Major Deep Learning framework starting from Keras, and moving to Pytorch and TensorFlow 2.0. 
- Keras is a high-level deep learning framework and is very well suited for fast prototyping and for beginners. 
- Pytorch is gaining popularity due to its use in Research and is considered pythonic, it provides flexibility to experiment with different Neural Network Architectures. 
- TensorFlow 2.0 is a revamp of old TensorFlow 1.xxx version which was not very user friendly with steep learning curve, static graphs and lots of boilerplate code to get things done. TensorFlow 2.0 solves this problem by adopting Ideas from Keras and Pytorch. TensorFlow is most suited to deployment in production with support to multiple production environments.

As we are dealing with Images, we will be focusing on Convolutional Neural Networks (CNNs / ConvNets), and start with simple CNN and train it from scratch.
We will then move to advance techniques like Data Augmentation, Dropout, Batch Normalization and using Pre-Trained networks like VGG16 trained on ImageNet.
<br> As we progress in our journey we will also explore some key aspects below, which are important for any Data Science Project.
1.	How much data is enough to get reliable results from Deep Neural Networks and is more data is always good?
2.	How do I deal with Bias and Variance tradeoff, and how to select best model which can generalize better without sacrificing too much of performance?
3.	For Image recognition tasks what works best, custom CNN model or a Pre-Trained network?
4.	What strategy to choose to validate the model performance. Hint: Trust your validation score and do not touch test set till end.
5. What Deep Learning Framework to choose and which one is best suitable for the task.

<br>There are no straightforward answers to some of the questions and it depends on context and the type of problem or data that we are dealing with, and it will be my endeavor to answers some of the difficult questions with experimentation and analysis.

# Pre-Requisites, Resources and Acknowledgments
- It is assumed that you have some experience in Python and Basic Deep Learning Concepts.

- Many of the ideas presented here are based on book [Deep Learning with Python]( https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438), written by Francois Chollet who also happens to be author of Keras Library

- Another excellent resource to learn theoretical concepts around deep learning is [Deep Learning Specialization]( https://www.coursera.org/specializations/deep-learning) taught by Andrew Ng on Coursera. Even though its paid course the Videos are freely available on YouTube. Initially it was not easy to grasp the convolutions, but thanks to Andrew Ng, Convolution Neural Networks no longer appear to be convoluted

- The book  focuses on the practical application while  Andrew Ng gravitates towards theoretical concepts with easy to understand mathematics. I have personally benefit from both, and even though both differ in their approaches they complement each other well just like Ensemble Machine learning models.

- Its recommended that that code be run on machine with GPU, there are many ways to achieve it without owning a high-end machine.  You can use Kaggle Notebook with GPU which is what I would be doing throughout the project. Links to Kaggle Notebooks will be shared and it can be forked, modified and results can be easily reproduced without any extra set up
<br>Another way is to use Google Colab or Google Cloud Platform with free credits of 300 USD.

# A note on the motivation and challenges
- When I started with this project my goal was to achieve a reasonable performance with my Image Classification model with more focus on building a robust model which can generalize well on unseen data.
- The data is user generated and images are taken by many users from all over India, from different angles, under different lighting conditions and mostly using mobile devices with varying image quality. 
- This presents an interesting challenge and I am very much curious as well as anxious as to how things would turn out. 
  I had heard multiple stories on how some models which performed well during training and validation, failed  be performed on new    data that came from different distribution.
- To make sure that our models are robust here are some techniques that we will apply.
  1.	When gathering data, I made sure that I downloaded car images from multiple regions like Delhi, UP, Maharashtra etc.
  2.	Delete images which are not car, incorrect model, closeup images, interior images. Anything which a human cannot recognize most likely ML model probably can’t recognize. Interestingly one of the class of data is WagonR, the model I owned for 7 years so it was easy for me to curate the data manually. This is where domain experience comes in handy.
  3.	The data is randomized and split into Training Set with 2400 images, Validation Set with 800 images and test set with 800 images. The model will be trained on training set, fine-tuned with feedback from validation data and test data will not evaluated until end when we have found our best model based on validation set. This is important because the test set acts as unseen data which model will encounter in future and if it performs well on this data, our model  will perform well on future data.  To further challenge our model, I have created another set of test data which was taken from different used car website and  different city (Hyderabad).

Hold on with your seat belts, grab some popcorn, and be ready for an exciting as well as thrilling ride, this sure will be a long and interesting one.

# Training a base CNN model

The CNN model architecture is shown below.
![model_arch](https://github.com/ajayrawatsap/Identify-a-Car-Model-with-Deep-Learning/blob/master/resources/cnn_arch.PNG)

- The images are converted to 150 X 150 X 3 shape and fed to CNN model. The first CONVD layer performs convolution on the input image with 32 filters with filter size of 3, resulting in layer of dimension 148 X 148 X 32,  which is then down sampled by a Max Pool layer of filter size 2 and stride of 2 resulting in layer of dimensions 74X74X32.  We are using four CONVD layers each with filter size of 3, followed by a Max Pooling layer of filter size 2 and stride of 2. The output from last MAX pool layer is flattened and converted to Dense layer of shape 512 X 1. The Final output layer consists of a single layer with sigmoid activation function. The other layers use Relu Activation Function

- You can notice that convolution operation increases the depth of the layer while keeping height and width almost same, while Max pool layer halves the height and width while keeping depth same. There is very simple math behind it which is not in scope of this tutorial. Andrew Ng explains this very well in his course
## Training in Keras
The keras code for buliding model is shown below
```python

def build_cnn(display_summary =False):
    model = models.Sequential()
    model.add( layers.Conv2D(32, (3,3),  activation= 'relu', input_shape = (150, 150, 3)) )
    model.add(layers.MaxPooling2D((2,2)))

    model.add( layers.Conv2D(64, (3,3),  activation= 'relu') )
    model.add(layers.MaxPooling2D((2,2)))

    model.add( layers.Conv2D(128, (3,3),  activation= 'relu') )
    model.add(layers.MaxPooling2D((2,2)))

    model.add( layers.Conv2D(128, (3,3),  activation= 'relu') )
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation= 'relu'))
    model.add(layers.Dense(1, activation= 'sigmoid'))

    model.compile(loss = 'binary_crossentropy',
                  optimizer = optimizers.RMSprop(lr = 1e-4),
                  metrics = ['acc']
                  )
    if display_summary:
       model.summary()
    return model
```
Model summary 
```
______________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 6272)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               3211776   
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
```

### Keras: Training and Validations Results
The model was trained for 50 epochs on a Kaggle notebook with GPU and achived accuracy of 88.125 percent on validation set.
![results](https://github.com/ajayrawatsap/Identify-a-Car-Model-with-Deep-Learning/blob/master/resources/results_keras_cnn_base.PNG)
- I had set a conservative target of 80% accuracy, but it seems our baseline CNN model performed better than expected with 88% accuracy.  If you ask me if this a good accuracy, and I might say it’s pretty good considering that a random classifier will be 50% accurate as there are  equal number of samples of each class.  But how is the performance compared to a human, and I will agree that humans will typically perform with 96% accuracy. The benchmark is raised, and our goal will be to achieve near human performance. At this point we have no idea if we can achieve the target accuracy.
- Ok, the model is pretty good for a baseline model, what about its robustness, can it perform well on unseen data?. As we can see from the screenshot that the training accuracy is much higher than the validation accuracy throughout all epochs, and now we are talking the bias and variance tradeoff. The model is clearly showing classic symptoms of low bias (since training accuracy is near 100%), and high variance(since validation accuracy is much lower at 88) or overfitting.  I would be not willing to put this model into production even if you think the accuracy is good enough. As we will see there many ways to deal with this and we will explore it in next sections.

### Keras source code 
 - [github](https://github.com/ajayrawatsap/Identify-a-Car-Model-with-Deep-Learning/blob/master/keras/cars_keras_cnn_baseline.ipynb)
 - [kaggle](https://www.kaggle.com/ajaykgp12/cars-keras-cnn?scriptVersionId=20357823)
 ## Training in Pytorch
    The same model in pytorch can be written as shown below
    
```python
 class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels=32, kernel_size= 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride= 2)
        
        self.conv2 =  nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= 3)
        self.conv3 =  nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3)
        self.conv4 =  nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3)
    
       #128 * 128 * 7 is the output of the last max pool layer
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 2)       

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        #this is similar to flatten in keras but keras is smart to figure out dimensions by iteself.
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
 ```

### Pytorch: Training and Validations Results
![keras_cnn_base_results](https://github.com/ajayrawatsap/Identify-a-Car-Model-with-Deep-Learning/blob/master/resources/results_pytorch_cnn_base.PNG)
- It was little time consuming to re-write the same model in Pytorch as it does not have Keras like fit method to train as well as evaluation your model on validation set. In Pytorch you will have to write your own training and test methods and run each method for every epoch. The advantage is you have more control at every epoch and can write custom metrics or loss functions easily.We can see that the Pytorch model displays same pattern as Keras with low bias and high variance.
- The Pytorch performs  better with validation accuracy of 0.896 compared to 0.88125 in Keras. This can be due to normalization, in Pytorch the image pixel values are scaled to [-1,1] while in Keras the same are scaled to [0,1].  When I used the scaling of [0,1] in Pytorch the results were even worse than Keras. So, its would be worthwhile to revisit and check how Keras performs under same normalization

### Pytorch source code
- [kaggle](https://www.kaggle.com/ajaykgp12/cars-pytorch-cnn?scriptVersionId=20577825)
- [github](https://github.com/ajayrawatsap/Identify-a-Car-Model-with-Deep-Learning/blob/master/pytorch/cars_pytorch_cnn.ipynb)
# Using Data Augmentation and Dropout
