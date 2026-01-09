# Sclera Identity Classificatoin
## MLOps Exam project 

Martin Vlacil 254127

Amalie Sommer Thomsen s200655

Oskar Kocjancic 253070

Andrei Soldan, s243873

### Project Goal
The goal of the project is to identify a person's identity based on segmented images of their scleras (the white part of the eye). 

### What Frameworks Will We Use?
We intend to use several frameworks that will simplify our work from the development and the MLOps perspective.

For the package management, we intend to use 'uv'.

We are using cookiecutter and the template provide to us to generate the project repository. 

We will use git for version control.

We will use Docker for containerization to ensure that the project runs on the 'same' system.

We will use hydra for the management of configurations.

We will use pytorch profiling for profiling the code. 

For logging we will use loguru.

Finally, we intend to use wandb for tracing our runs.

### What data are you going to run on (initially, may change)
The data we are going to use (initially) is comprised of segmented images of human scleras. 
There are 110 people with 2 identities per person (each identity mapping to a person's eye). Each identity (person's eye) has on average 4 images from different angles. There are around 3704 images in the dataset. 

!Important! - this project does not understake the parsing of images with human eyes into segmented images. 


### What models do you expect to use
We are going to use Convolutional Neural Networks (CNNs) for image classification.
