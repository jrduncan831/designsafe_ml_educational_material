# VGG16 transfer learning for image classfication 

## Single GPU training

This tutorial contains a series of jupyter notebooks that utilize a single gpu. These notebooks demo a series of new deep learning techniques:

0. copy-data.ipynb : Copies data to the compute node  
1. torch-train-1st.ipynb : Performs VGG16 transfer learning for design safe image classification  
2. torch-train-2nd.ipynb : Performs VGG16 transfer learning for design safe image classification with data augmentation 
3. torch-train-3rd.ipynb : Performs VGG16 transfer learning for design safe image classification with data augmentation, label smoothing in loss function, and learning rate decay   
4. torch-infer.ipynb : Explore the performance of your top performing by seeing what your model gets right and wrong. 

## Multi GPU training 

In the previous notebooks, a single gpu was utilized.  Run run-distributed.ipynb to utilizes multiple gpus for training (3rd notebooks above) and see speedup. The notebooks run-dstributed.ipynb runs the script torch-train-3rd-distributed.py.  
