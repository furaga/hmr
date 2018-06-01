# End-to-end Recovery of Human Shape and Pose

This project is forked from https://github.com/akanazawa/hmr

Angjoo Kanazawa, Michael J. Black, David W. Jacobs, Jitendra Malik
CVPR 2018
[Project Page](https://akanazawa.github.io/hmr/)

### Requirements
- Python 3.5
- [TensorFlow](https://www.tensorflow.org/) tested on version 1.8

### Installation

Install chocolatey:
https://chocolatey.org/install#installing-chocolatey

```
chocolatey wget

# setup virtual environment
conda create -n hmr3 pip python=3.5

# install tensorflow
pip install --ignore-installed --upgrade tensorflow-gpu 
# Download and install CUDA 9.0 from this URL: https://developer.nvidia.com/cuda-toolkit

# clone projects 
git clone https://github.com/furaga/hmr.git
cd hmr

# install packages
pip install -r requirements

# Download the pre-trained models
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/hmr/models.tar.gz && tar -xf models.tar.gz

```

Run the demo
```
python -m demo --img_path data/coco1.png
python -m demo --img_path data/im1954.jpg
```

