#!/bin/bash
sudo apt-get update
sudo apt install software-properties-common
sudo apt-get install python3.7
git clone https://github.com/Evolutionary-strategies/Evolutionary-strategies.git
cd Evolutionary-strategies
pip3 install -r requirements.txt

