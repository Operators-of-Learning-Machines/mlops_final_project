# Sclera Identity Classificatoin
## MLOps Exam project 

Martin Vlacil 254127

Amalie Sommer Thomsen s200655

Oskar Kocjancic 253070

Andrei Soldan s243873

### Project Goal
The goal of the project is to identify a person's identity based on segmented images of their scleras (the white part of the eye). 

### What Frameworks Will We Use?
#### We intend to use several frameworks that will simplify our work from the development and the MLOps perspective.

For the package management, we intend to use **uv**.

We are using **cookiecutter** and the template provided to us to generate the project repository. 

We will use **git** for version control.

We will use **Docker** for containerization to ensure that the project runs on the 'same' system.

We will use **hydra** for the management of configurations.

We will use **pytorch profiling** for profiling the code. 

For logging we will use **loguru**.

Finally, we intend to use **wandb** for tracing our runs.

### What data are you going to run on (initially, may change)
The data we are going to use (initially) is comprised of segmented images of human scleras. 

There are 110 people with 2 identities per person (each identity mapping to a person's eye). Each identity (person's eye) has on average 4 images from different angles. There are around 3704 images in the dataset. 

!Important! - this project does not understake the parsing of images with human eyes into segmented images. 


### What models do you expect to use
We are going to use Convolutional Neural Networks (CNNs) for image classification.

We are starting out with SqueezeNet and itterate from that, using more modern models.


## Just for running the training script
`uv run invoke train --experiment=exp_{number}` 


## To build the docker image with a GPU-compatible base image:
- `docker build -f .\dockerfiles\train.dockerfile  -t sclera-train:gpu .`

Or run 

- `uv run invoke docker-build --progress={auto/tty}`

## To run the docker image:

We now need to have the sclera-trainer.key.json file locally when building images and running containers locally.
To get it, run the following commands:

1. gcloud iam service-accounts keys create sclera-trainer.key.json ^
  --iam-account=sclera-trainer@eternal-lodge-484208-j6.iam.gserviceaccount.com

2. move sclera-trainer.key.json C:\PATH_TO_YOUR_PROJECT\mlops_final_project\

It is already added to the .gitignore file.

Now, to run the container, use:

docker run --rm --gpus all --shm-size=2g `
  -e WANDB_API_KEY=you_api_key `
  -v "${PWD}\sclera-trainer.key.json:/secrets/gcp.key.json:ro" `
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp.key.json `
  sclera-train:gpu
