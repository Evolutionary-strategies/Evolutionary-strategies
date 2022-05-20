# Evolutionary strategies
This is a repository for training machine learning models using NES, and running adversarial attacks against said models. 

## Setup
```
git clone https://github.com/Evolutionary-strategies/Evolutionary-strategies.git   
cd Evolutionary-strategies   
pip3 install -r requirements.txt
```

## Running
First change directory to es:
```
cd es
```
Then open `main.py` and choose what to run within the file, then use the command:
```
python main.py
```
To train NES models change the branch to training, and run main

## Structure
In the [scripts folder](https://github.com/Evolutionary-strategies/Evolutionary-strategies/tree/master/scripts) you will find scripts for setup, and for plotting model accuracy to graph.

In the models folder you will find the starting weights used for model training in our experiments.

In the [es folder](https://github.com/Evolutionary-strategies/Evolutionary-strategies/tree/master/es) you will find the main research code. 

Files in the es folder:

- adversarial_attack.py contains a foolbox implementation of adversarial attacks
- dist.py is used for communication between master and worker using redis
- es.py contains the NES training algorithm and relevant methods
- main.py contains startup configuration and the main function
- model.py contains our model architecture and relevant methods
- print_img.py is used for printing images from the dataset
- util.py contains different helper methods

## Known issues

Running `adveserial_attack.py` might cause an error on some computers. We believe the error is caused by multiprosessing the foolbox library in 
conjunction with cuda. 


