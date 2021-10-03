# Panoptic Segmentation Task

1.	Consolidate the Segment Information from the individual .json files 
2.	Capture the segmentation information from the inferencing of individual predictions 
3.	Add the Coco stuff segmentations after converting the RLE format 
4.	Freeze the weights of previous run (Bounding Boxes)
5.	Train the Model again to generate the Attention Maps and Panoptic Segmentation 

# Modification Done to the code repository

1.	Modified the number of classes to 80
2.	Updated the method to load the images and annotations for segmentation
3.	Method to convert RLE format 

# Loss Function planned to be used and analyzed 
•	L1 Loss  
•	GIOU Loss
•	CE Loss
•	Dice Loss
•	Focal Loss
•	Bounding Box Loss:
•	Mask Loss
•	Total Loss
# Detail Discussion on DETR and how to implement
https://github.com/nkanungo/EVA6/tree/main/DETR

# Current Status 
1.	Setup the Basic Code structure to Train 
2.	Prepared the Dataset (updated. json)
3.	Getting code errors while trying to run the first epoch (for testing)

# Completion Status 
I will not be able to complete even 1 training run till the deadline, however will try to complete the training soon.


