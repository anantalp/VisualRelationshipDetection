## *VIsual Relationship Detection 2019*

### Introduction:
This was a course project for the class CAP6614 - Current Topics in Machine Learning, Fall2019. The task was to find visual relationship between objects in a frame on [OpenImages V5 dataset](https://storage.googleapis.com/openimages/web/index.html).

### Abstract:
Visual Relationship Detection is relatively a newer problem in the field of Computer Vision, and the task is to find the relationship between objects in an image. It includes detection of the object followed up with determining the relationship between objects since object detection alone cannot analyze semantic information present in the image. Furthermore, the task becomes even more complex as the Visual Relationship Challenge 2019 dataset is very large and diverse. In our work, we convert this task into a classification problem by cropping the images by making use of the bounding box annotations that are provided and then feed into our model. Further, we analyze the effects on our classification metrics by varying the number of layers in our model, and by varying the dataset usage during our training, and hyperparameters. Lastly, we discuss shortcomings and future work.

### Dataset Description:
The dataset used for our experiments is a subset of the OpenImages V5 dataset. Whilst the original images contain over 9 million annotated images, we shorten the dataset to around 330, 000 images that are relevant to our visual relationship detection problem.

<img src="https://github.com/anantalp/VisualRelationshipDetection/blob/main/figures/fig1.PNG">

Distribution of train and validation instances in the shortened dataset1. While the x-axis indicates the number of classes in our model, the y-axis which is in the logarithmic scale indicates the number of instances in our modified dataset.

### Network Training:
Use or modify the required dataloader present in the dataloader folder. To train the model, run Python train.py. For inference, use eval.py. 

### Visualization:
<img src="https://github.com/anantalp/VisualRelationshipDetection/blob/main/figures/fig3.PNG">

Few qualitative results shown from different relationship categories in a 3x3 grid

### Results:
|Model Backbone |Training Loss  |Validation loss  |Validation loss  |F1 Score  |Precision  |Recall|
|---|---|---|---|---|---|---|
|Resnet-152|0.61|0.58|82.39|0.36|0.50|0.35|


