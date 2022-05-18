# Evolutionary strategies
This is a repository for training machine learning models using NES, and running adversarial attacks against said models. 

## Setup
`
git clone https://github.com/Evolutionary-strategies/Evolutionary-strategies.git \
cd Evolutionary-strategies \
pip3 install -r requirements.txt \
`

## structure
In the scripts folder you will find scripts for setup, and for plotting model accuracy to graph.

In the models folder you will find the starting weights used for model training in our experiments.

In the es folder you will find the code. 

In the es folder files:

- adversarial_attack.py contains a foolbox implementation of adversarial attacks
- dist.py is used for communication between master and worker using redis
- es.py contains the training algorithm and relevant methods
- main.py contains startup configuration and the main function
- model.py contains our model architecture and relevant methods
- print_img.py is used for printing images from the dataset
- util.py contains different helepr methods
