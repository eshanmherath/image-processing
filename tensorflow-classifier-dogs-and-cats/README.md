This code can be trained for most of the real time image classification tasks.
Training data for your particular scenario has to be put in to training_data folder.
Eg: this particular scenario is for identifying dogs vs cats from the web camera input
(in fact it should be dogs vs cats vs None;
But since this is a bare bone code, the multi-class classification is not included)

prerequisite:
	python 3.x
	numpy
	tensorflow
	opencv

Step 0: Put your training images to training_data directory. (for this code to work there should be at least 130+ images
per category. You can go through the train.py and change the code if you have less then 130 per category)

Step 1:
set number of iterations to your preferred value based on the number of training examples and time you have
Run train.py  (clean the previous models in models directory if you prefer)

Step 2:
set the path to your opencv 'haarcascade_frontalface_default.xml' in run.py line 14
(if you want to run with already trained model, adjust the paths in models/checkpoint. but it is recommended to train
your own model since the model in this code base is not trained for a long time)
Run run.py

press 'q' on teh keyboard to quit the program

when starting run.py over and over again make sure to clean the content of real_time_images manually (or write a code
snippet to clean pragmatically and put it on the top of run.py)