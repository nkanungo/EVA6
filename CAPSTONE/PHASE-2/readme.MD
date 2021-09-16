Train your Custom Dataset to produce Panoptic Segmentation 
==============================================================
The Task here is to Train a Pretrained DETR Network using Custom Dataset. Here is the task that we need to perform in order to achieve the required result 

# Ground Truth
The DETR Panoptic segmentation is designed to predict two things 
1.	The Things class: This basically refers to the class in the foreground 
2.	The Stuff class: This refers to the items in the background 
The DETR network trained on COCO datasets is supposed to predict both thing class and stuff class 
There are 90 things and 109 stuff classes in coco dataset. However, for us in the custom dataset we plan to 
1.	Annotate the Construction Materials (Around 45 classes)
2.	Take the Stuff classes from COCO

# Thing class Ground Truth Preparation 
This was a collaborative task which was performed by each student of TSAI taking EVA6. Every individual was allocated a specific item to annotate and around 500 images for each task. The COCO type annotation was generated combinedly

# The Stuff Class Ground Truth
We decided to take the validation dataset which contains around 5000 images as the ground truth for stuff classes 

# The Process of Preparing the Ground Truth

1.	Setup a Python program to perform inferencing on the custom images using DETR panoptic pretrained model 
2.	Collect the inferences which also includes the Bounding Boxes, Segments, Area and Class for each object within the image and for each image 
3.	Extract the output to a csv file 
4.	Consider all things predicted by the model as “misc_stuff” as for us the things are nothing but the objects that we annotated. So basically, any prediction with category < 90 must be considered as misc_stuff 
5.	Consider the stuffs it predicted as is
6.	Prepare Annotation JSON from the above prediction 
7.	Add the above annotation to the things annotation prepared during manual tagging 
8.	Ensure to modify the categories accordingly 
9.	Also, as multiple classes contain the same file name, ensure to rename the files to have unique name. Here I added class short prefix to the images 
10.	Descale the Bounding boxes and masks that the model predicted based on the size of the original image 
11.	Ensure to split the annotations and images with a ratio of 80:20 for the thing classes. This is helpful for handling class imbalance problem 
12.	Add the COCO stuff validation annotations and images in a ratio of 80:20 to the above dataset.
13.	Check the coordinates of the things that we annotated manually and the coordinates of the items that the model predicted 
14.	Remove the overlaps programmatically like the above to ensure that things are not impacted while training 
15.	That’s it. If you have done this successfully then this completes the ground truth preparation process of bounding box prediction 


![](images/false_image.png')

# Training Process
Once the Ground truth is prepared the model needs to be trained along with this data to predict the bounding boxes for both things and stuffs. This information needs to be fed to our next model which would take the images with bounding boxes to predict the panoptic segmentation. However, that’s the task for phase-3 which we would do next 

# Where am I in this 
1.	Prepare the Dataset using multiple python scripts and setup the programs to inferences and train the DETR model for custom datasets 
2.	Ran Test on raw images to ensure that it’s displaying properly 
3.	Ran Training for sample classes 
4.	However, the Training threw error while training with complete dataset. This might be possible as I modified my logic to fine tune the overlap. 
5.	Spent today again to prepare the dataset and it’s 80% complete. 
6.	Plan to resume training tomorrow. I know I am going to miss the deadline and possibly the chance to move phase-2 of EVA -6. However, I will try my best to complete the capstone with good accuracy. 
