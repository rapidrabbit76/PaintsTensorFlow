# PaintsTensorFlow 
### [PaintsChainer](https://github.com/taizan/PaintsChainer/) by Tensorflow
<img src="./src/GUI/3.jpeg" width="800">

# Model Structure
<img src="./src/model structure/Model 01.jpeg" width="800">
<img src="./src/model structure/Model 02.jpeg" width="800">

# Results
### input(line) - input(hint) - draft - output - ground truth
Gray background in hint for visualization.  

<img src="./src/sample/1.jpg" width="800">  
<img src="./src/sample/2.jpg" width="800">  
<img src="./src/sample/3.jpg" width="800">  
<img src="./src/sample/4.jpg" width="800">  
<img src="./src/sample/5.jpg" width="800">  
<img src="./src/sample/6.jpg" width="800">  

# GUI
<img src="./src/GUI/1.jpeg" width="800">
<img src="./src/GUI/2.jpeg" width="800">
<img src="./src/GUI/3.jpeg" width="800">  


# Requirements  
* tensorflow_gpu==1.12.0 or tensorflow==1.12.0
* numpy==1.15.4
* tensorlayer==1.11.1
* tqdm==4.28.1
* opencv_python==3.4.4.19
* scipy==1.1.0
* Pillow==5.3.0
* PyQt5==5.11.3

# Install  
1. pip install -r requirements.txt
2. Download pretrained models [PaintsTensorFlow](https://drive.google.com/file/d/1uzLbfUeG8qnrv2zWemTmK0fp_VupExCe/view?usp=sharing)
and [Liner](https://drive.google.com/file/d/1h6rKAyWUfYGZd2J_L_nalPPfKbL8Br7Y/view?usp=sharing)
    - **Liner** is **SketchKeras** model
3. Copy the files in **PaintsTensorFlow** into **"./GUI/model/PaintsTensorFlow/"**, and
 the files in **Liner** into **"./GUI/model/Liner"**
3. python3 runGUI.py


# Training
- #### My Datasets are over 700,000 images and I created a lines, using [SketchKeras](https://github.com/lllyasviel/sketchKeras)

- #### I uses Eager mode in training step

- #### datasets path structure (**image-line fileName must be matched**)
        ├─ root
        │    ├─ train
        │    │    ├─ image (hyperparameter: train_image_datasets_path)
        │    │    │    └─ 1.jpg, 2.jpg, 3.jpg 
        │    │    ├─ line  (hyperparameter: train_line_datasets_path)
        │    │    │    └─ 1.jpg, 2.jpg, 3.jpg
        │    ├─ test
        │    │    ├─ image (hyperparameter: test_image_datasets_path)
        │    │    │    └─ 1.jpg, 2.jpg, 3.jpg
        │    │    └─ line  (hyperparameter: test_line_datasets_path)
        │    │         └─ 1.jpg, 2.jpg, 3.jpg
        
- ### step 1: Training draft model 128X128 size **Total 10 epoch**


        1.1. python3 training.py -loadEpochs 0 -mode draft


    hyperparameter: lr =  1e-4 , epoch = 5 , batch_size = in my case 16 recommendation is 4  

        1.2. python3 training.py -loadEpochs 4 -mode draft


    hyperparameter: lr =  1e-5 , epoch = 5 , batch_size = same as step 1.1
    
- ### step 2: Training model 512x512 size **Total 2 epoch**


        2.1. python3 training.py -loadEpochs 0 -mode 512


    hyperparameter: lr =  1e-4 , epoch = 1 , batch_size = in my case 3 recommendation is 4

        2.2. python3 training.py -loadEpochs 0 -mode 512


    hyperparameter: lr =  1e-5 , epoch = 1 , batch_size = same as step 2.1


# Loss
### Draft model Generator Loss
<img src="./src/loss/Draft%20model%20Generator%20Loss.png" width="800">

### 512x512px model Generator Loss
<img src="./src/loss/512x512px%20model%20Generator%20Loss.png" width="800">


# References
- [PaintsChainer](https://github.com/taizan/PaintsChainer/)
- [SketchKeras](https://github.com/lllyasviel/sketchKeras)
- [pix2pix](https://arxiv.org/pdf/1611.07004.pdf)
