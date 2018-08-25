# Tensorflow implementation of Faceboxes (for UROP)

An implementation of the following paper by Zhang et al. in Vanilla Tensorflow. Original paper can be found [here](https://arxiv.org/abs/1708.05234). It is a relatively straight-forward paper to reimplement (for those with experience in SSD related systems). That being said, the paper refers to Receptive Fields when detailing it's anchors - I'm still a bit unsure what this is referencing (if you have any insight, please do let me know!).

On the CPU on my home computer, I get an average of ~8fps using this model, which is more than enough for good object-tracking.

A sample semi-trained face-model (that was verified to work on my home webcam) can be found in `~/models/` - feel free to use that as a baseline to start training. If necessary, you may need to tweak the piecewise constant values in `model.py` to adjust your learning rate to the desired value.

 ## Repository Information
Note, the repository is structured as follows:

 - `Tests` folder includes tests to verify that different parts of the function (including anchor densification strategy etc.) are working as prescribed in the paper. Both GPU augmentation and CPU augmentation are also tested.
 - `wider.py` extracts and pre-processes the dataset for efficient sampling
 - `augmenter.py` details GPU-based data augmentation (including bounding-box augmentation). Note, I also include a feature to rotate the bounding box and image. There are a *lot* of hyperparameters that you can tune here, if you'd like (an example test script can be found in `~/test/`)
 - `data.py` includes a CPU-based multiprocessing augmentation tool. Running this ended-up being too slow, which is why I developed the GPU-based tool.
 - `webcam_run.py` is a script that runs the model on your webcam using your CPU (though, this can be disabled if you'd like). To try and get detections on faces which are further away from the camera, I upsample some of the images (with some success, though hyperparameters may need tweaking). 
 - The remaining files should be relatively self explanatory (model files etc.).
 
 ## Key Takeaways
 
 - This model/approach **really** struggles with small faces. That being said, modifications to the anchor densification strategy + other tweaks may result in a model that can handle this (without too much of a loss in CPU speed).
 - For more complicated objects (i.e. cars), the model may need to include more parameters (i.e. more features at each conv-layer) - again, at the slight expense of speed.
 - This is definitely a good baseline for CPU based computation with deep learning, but changes are needed to really make this stand-out over traditional systems (such as pico) in face-detection. That being said, because of the nature of the transferability of architectures to different learning domain, this model can be easily adapted to other scenarios. 
 - There are a **lot** of hyperparameters to tune - this can definitely be done more efficiently than my preliminary search. Please do  let me know if you tune things yourself and find a better strategy with the same code (and feel free to push changes).  
