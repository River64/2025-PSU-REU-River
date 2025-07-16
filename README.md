# 2025-PSU-REU-River
Code used in publication Capturing Non-motorized Counts at Intersections Using Ultralytics YOLOv8 Image and Video Tagging by Alicia Hopper, Tammy Lee, and Sirisha Kothuri

Code written by Alicia Hopper, during a 2024 NSF REU program at Portland State University (PSU)
Code edited by River Johnson, during a 2025 NSF REU program at Portland State University (PSU)

To be used with filesystem structure:

Project

├ 2025-PSU-REU-River (this code)

├	datasets (any datasets intended to be used for training or validation)

    └ ECP (eurocity persons database) -- can be obtained from https://eurocity-dataset.tudelft.nl/eval/overview/statistics

        ├ images

            ├ train (training images)

            └ val (validation images)
            
        └ labels

            ├ train (training labels - one corresponding to each image)

            └ val (val labels - one corresponding to each image)

        └ old_labels (folder to put original labels in ECP format before conversion into YOLO format)
