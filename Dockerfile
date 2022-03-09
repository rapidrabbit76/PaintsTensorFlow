FROM tensorflow/tensorflow:2.7.1-gpu

WORKDIR /devel

RUN apt update \ 
    && apt install -y \ 
    git htop vim \ 
    ranger \
    && apt clean

RUN pip install \
    wandb==0.12.11 \
    tensorflow-addons==0.16.1 \
    opencv-python-headless==4.5.5.62 \
    albumentations==1.1.0 \
    tqdm \
    scipy==1.8.0
