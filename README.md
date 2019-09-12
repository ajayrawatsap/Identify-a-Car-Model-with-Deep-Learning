# Identify-a-Car-Model-with-Deep-Learning
Explore how to practice real world Data Science by collecting data, curating it and apply advance Deep Learning techniques to create high quality models which can be deployed in Production.
# Data
There are 4000 images of two of the popular cars (Swift and Wagonr) in India of make Maruti Suzuki with 2000 pictures belonging to each model. The data is divided into training set with 2400 images , validation set with 800 images and test set with 800 images. The data was randomized before splitting into training, test and validation set.
<br>The data was collected using a web scraper written in python. Selenium Library was used to load the full HTML page and then Beautifulsoup library was used to extract and download images from HTML tags
<br> The data is hosted at [Kaggle](https://www.kaggle.com/ajaykgp12/cars-wagonr-swift) which can be downloaded or you can also create a notebook directly on Kaggle and access data in your notebook.
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

