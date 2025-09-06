#!/bin/bash
# This program sets up the ECP dataset in the required format and organization for YOLO model training.
# Run this after running download_ecp_dataset.sh or doing what that script does manually
# HAS NOT BEEN TESTED YET so may or may not work correctly at this point
# I just wanted to get it into the repository
# Written by River Johnson, 2025

# Create dataset destination folder, if it does not already exist
echo "Creating dataset folder..."
mkdir ../../datasets/

# Unzip ECP dataset files
# Assumes they have been downloaded
# and placed in the ECP_working directory
echo "Unzipping validation image labels..."
unzip ../../ECP_working/ECP_day_labels_val.zip -d ../../datasets/

echo "Unzipping training image labels..."
unzip ../../ECP_working/ECP_day_labels_train.zip -d ../../datasets/

echo "Unzipping validation images..."
unzip ../../ECP_working/ECP_day_img_val.zip -d ../../datasets/

echo "Unzipping training images... (this will take a longer time)"
unzip ../../ECP_working/ECP_day_img_train.zip -d ../../datasets/

# Rename the default folder to old_labels
# and make a new labels folder for the YOLO format labels
# This is convenient to keep because the data converter script may be
# run several times after setup for different configurations
echo "Putting original-format labels in old_labels folder..."
mv ../../datasets/ECP/labels ../../datasets/ECP/old_labels
mkdir ../../datasets/ECP/labels

# Run the data converter to put the image labels in the right place + format
echo "Running label data converter..."
python3 data_converter.py

# Move the images out of the city folders
echo "Moving images out of city folders..."
mv ../../datasets/ECP/images/train/*/* ../../datasets/ECP/images/train/
mv ../../datasets/ECP/images/val/*/* ../../datasets/ECP/images/val/

# Delete city folders
echo "Deleting city folders..."
rm -r `ls -d ../datasets/ECP/images/train/`
rm -r `ls -d ../datasets/ECP/images/val/`
# Hmmm. Is that how quote expansion works? I'll learn more later in the semester, I guess

echo "Done!"
exit 0
