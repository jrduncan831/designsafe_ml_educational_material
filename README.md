# Designsafe Machine Learning Educational Material

This repo contains a series of educational material which utilizes a Design Safe data set on image classification of house damage. There are a series of 4 tutorials 

1. Demo 1 (image_processing): This tutorial introduces basic image processing techniques. 
2. Demo 2 (image_classification_supervised_learning): This tutorial utilizes various machine learning technqiues to perform image classification.  The best performing model is explored.  
3. Demo 3 (unsupervised_learning): This tutorial uses various unsupervised learning techniques for image compression. 
4. Demo 4 (image_classification_cnn): This tutorial uses transfer learning to perform image classification. Training is performed on single and multiple gpus. 

## Environment Setup 

To set up the environment on TACC systems for all tutorials:

1. Log in to machine; Move to desired directory 
2. Load modules specified in commands below and save : 
`module load gcc/9.1.0 python3/3.9.2 cuda/11.3 cudnn nccl`
`module save default`
3. Clone this repo
4. Move into the repository directory: `cd sci_tacc_education_materials`
5. Create a python virtual environment: `python3 -m venv venv`
6. Activate the virtual environment: `. $PWD/venv/bin/activate`
7. Install the required python packages: `python -m pip install -r requirements-3.9.txt`
