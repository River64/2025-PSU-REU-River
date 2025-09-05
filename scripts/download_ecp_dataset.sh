#!/bin/bash
# This script downloads the ECP dataset files necessary for this project
# and places them in a working directory for them to be processed
# Written by River Johnson, 2025
#
# If you are not comfortable giving any password to a random shell script: fair!
# Here's how to do this manually:

# Check usage, print usage message if not enough command-line arguments
# This section is unfinished, because I haven't learned flow control in shell scripts yet
echo $0 " usage:"
echo "./$0 ECP_username ECP_password"

# Make a directory for the zip files
mkdir ../../ECP_working

# Download the necessary ECP dataset files
echo "Downloading validation image labels..."
wget --auth-no-challenge --user=$1 --password=$2 --output-document=ECP_day_labels_val.zip http://eurocity-dataset.tudelft.nl//eval/downloadFiles/downloadFile/detection?file=ecpdata%2Fecpdataset_v1%2FECP_day_labels_val.zip

echo "Downloading training image labels..."
wget --auth-no-challenge --user=$1 --password=$2 --output-document=ECP_day_labels_train.zip http://eurocity-dataset.tudelft.nl//eval/downloadFiles/downloadFile/detection?file=ecpdata%2Fecpdataset_v1%2FECP_day_labels_train.zip

echo "Downloading validation images..."
wget --auth-no-challenge --user=$1 --password=$2 --output-document=ECP_day_img_val.zip http://eurocity-dataset.tudelft.nl//eval/downloadFiles/downloadFile/detection?file=ecpdata%2Fecpdataset_v1%2FECP_day_img_val.zip 

echo "Downloading training images... (this will take a longer time)"
wget --auth-no-challenge --user=$1 --password=$2 --output-document=ECP_day_img_train.zip http://eurocity-dataset.tudelft.nl//eval/downloadFiles/downloadFile/detection?file=ecpdata%2Fecpdataset_v1%2FECP_day_img_train.zip 

# Move the files into the relevant directory
echo "Moving zip files into ECP working directory..."
mv ECP_day_*.zip ../../ECP_working

echo "Done!"
