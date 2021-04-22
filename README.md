# FaceAnime
The source code (pytorch) for our project: Image-to-Video Generation via 3D Facial Dynamics. 

Training & testing datasets.
We use the IJBC video datasets for the training of our generation model. All the images are cropped by the 2DAL algorithm. 
Specifically, the face IDs with poor quality are removed firstly, 2447 video folders are selected for the training & testing. 20 video folders are randomly selected, others are used for training. The training and testing subsets can be downloaded at: Baidu Yun.  


Usage:
1. Download the files at: Baidu yun.
2. Extract the files under the root folder.
3. Run the main script "main_paraltrain_withMask_300vw_idloss" for training, all the implementation details are contained in this script. 
