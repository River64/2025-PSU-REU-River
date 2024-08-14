# 2024-PSU-REU
Code used in publication Capturing Non-motorized Counts at Intersections Using Ultralytics YOLOv8 Image and Video Tagging by Alicia Hopper, Tammy Lee, and Sirisha Kothuri

Code written by Alicia Hopper, during a 2024 NSF REU program at Portland State University (PSU)



To be used with filesystem structure:

Project

├ 2024-PSU-REU (this code)

├	datasets (any datasets intended to be used for training or validation)

    └ ECP (eurocity persons database)

        ├ batchX (the current batch of data to train on (eg. batch1, batch2, etc))

            ├ images

                ├ train (training images)

                └ val (validation images)
            
            └ labels

                ├ train (training labels - one corresponding to each image)

                └ val (val labels - one corresponding to each image)

        ├ labels (folder to put converted YOLO labels in before organizing into batches)

        └ old_labels (folder to put original labels in ECP format before conversion into YOLO format)

