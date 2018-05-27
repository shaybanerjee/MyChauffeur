# MyChauffeur
self-driving convolutional neural network model
<h2>Inspiration:</h2> 
After completing an online machine learning course, I was learning about neural networks. The concept of neural networks was fascinating and I wanted to build my own from scratch. I quickly learned that this was not an easy task; hence, I decided to use the TFlow AlexNet algorithm to better understand how it fits and trains a model. However, before using the algorithm, I needed to collect training data. So, I decided to build a fun project that could use convolutional neural networks to create a self-driving car. 

My first approach was without a neural network. I followed a tutorial on Udemy where they place a bunch of lines on objects in the environment so that the car knows where the street is and will not go towards the other objects. This approach made sense to me, but I figured instead of considering numerous lines, a KMeans algorithm can be used to only select/create two lines for which the car can use to travel. This approach was still not precise enough. 

Hence, I decided that a convolution neural network would be very useful. My model would be able to classify different directions of movement (N, S, E, W), by taking a frame as input. I recorded two hours of gameplay, on an online car simulator, for my training data, in which I collected the frames and as well as the key presses during those frames. Using this data, I applied AlexNet to train a model with 84% accuracy. 

Built with: Python, TFlow, AlexNet.
