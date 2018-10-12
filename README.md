# Semantic Segmentation

[//]: # (Image References)
[image1]: ./runs/1539311316.9751396/um_000000.png
[image2]: ./runs/1539311316.9751396/um_000041.png
[image3]: ./runs/1539311316.9751396/um_000062.png
[image4]: ./runs/1539311316.9751396/um_000023.png
[image5]: ./runs/1539311316.9751396/uu_000001.png
[image6]: ./runs/1539311316.9751396/uu_000010.png
[image7]: ./runs/1539311316.9751396/uu_000020.png
[image8]: ./runs/1539311316.9751396/umm_000000.png

### Introduction
The objective of this project, is to label the pixels of a road in images using a Fully Convolutional Network (FCN). 

### Architecture
The architecture of the FCN that I used is the [FCN-8 architecture](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) developed at Berkeley. The FCN-8 architecture encoder is based off of the VGG16 model that is per-trained on an ImageNet classifier. The fully connected layers in the FCN-8 have been replaced with 1x1 convolutions. The decoder portion of the architecture up-samples the 1x1 convolutions back to the original image size. Skip connections are added to the model by combining the outputs of two layers, in this case layers 3 with 7 and 4 with 7.

### Optimizer
The optimizer is an Adam Optimizer minimizing cross-entropy loss. 

### Stop-Loss
Instead of implementing a loop with a set number of epochs, I've opted for implementing a stop-loss function. If the average loss of the batches ran through the model for an epoch is less than the same loss two epochs ago the model ceases it's training. This resulted is 54 epochs for the given hyper-parameters that I've chosen to use.

### Hyper-Parameters
- Learning-Rate: .0005
- Keep Probability: .60
- Batch Size: 25

### Results
![][image1]
![][image2]
![][image3]
![][image4]
![][image5]
![][image6]
![][image7]
![][image8]
As can be seen from the pictures above, it does well but there are some critical errors in some images that could cause serious problems in a production environment. 

### Reflection
I think had I tweaked the hyper-parameters more, augmented the original images, fed the model a larger dataset or did all of those it would have performed much better. This is probably something I will return to in the future to improve on, as it feels like a very promising endeavour.

---
# ~Below is the Readme from Udacity~

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
