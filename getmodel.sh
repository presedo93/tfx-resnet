#!/bin/bash

# Fetch the model
wget -O resnet_50_classification_1.tar.gz https://tfhub.dev/tensorflow/resnet_50/classification/1\?tf-hub-format\=compressed

# Create the assets folder
mkdir -p assets/resnet/1538687457

# Extract the model and variables
tar -xvf resnet_50_classification_1.tar.gz -C assets/resnet/1538687457

# Delete the .tar.gz
rm resnet_50_classification_1.tar.gz

# Echo
echo "Done!"
