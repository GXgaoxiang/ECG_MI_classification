# ECG_MI_classification
## Multi-branch Fusion Network for Myocardial Infarction Screening from 12-lead ECG Images
I have implemented paper named "Multi-branch Fusion Network for Myocardial Infarction Screening from 12-lead ECG Images" whcih develop an 
approach that can detect the MI automatically by simply inputting ECG images. First, we use text detection and position alignment to automatically generate the bounding box of 12 leads. Then, those 12 leads are input into the multi-branch network which is constructed
by a shallow neural network to get 12 feature maps. After concatenating those feature maps by depth fusion, the output of feature fusion is fed into a modified Densenet model to do classification.

## Extraction of leads
This part consists of two steps. 
* first step
The first step is to detect the text above each lead. This part we use the Yolo3 model implemented on github: https://github.com/qqwweee/keras-yolo3. In order to print 12 boxes better, we changed the score of output.
* second step
The positions of those texts are used to locate the position of each lead.

