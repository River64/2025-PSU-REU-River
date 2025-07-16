"""
This program is intended to convert labels from the EuroCity Persons (ECP) Dataset 
into txt files in the format used to train a YOLO model.

Written by Allie Hopper, 2024

Edited by River Johnson, 2025
Absolute file paths -> relative file paths from current working directory
(current working directory must be repository directory, otherwise the script will not work)
Now processes training and validation data together
"""

import json
import os

# define a dictionary of all the possible tagged objects
identities = {
  "pedestrian": 0,
  "rider+bicycle": 1,
  "rider+scooter": 2,
  "rider+wheelchair": 3,
  "rider+tricycle": 4,
  "rider+buggy": 5,
  "rider+motorbike": 6,
  "person-group-far-away": 7,
  "rider+vehicle-group-far-away": 8,
  "bicycle-group": 9,
  "buggy-group": 10,
  "motorbike-group": 11,
  "scootergroup": 12,
  "tricycle-group": 13,
  "wheelchair-group": 14,
  'scooter-group': 15,
  'rider+co-rider': 16,
  'skateboarder': 17,
  'rider': 18, 
  'bicycle': 19, 
  'motorbike': 20
}

def read_data(train_val, file_name, city):
  """Opens and reads a JSON file, and generates the path to the new file

  @Params:
    file_name: The JSON file name (as a string)
    batch: The city the data was taken in(as a string)

  @Returns:
    The data loaded from the JSON file
    The name of the txt file to write the data to
  """
  
  file = './../datasets/ECP/old_labels/' + train_val + '/' + city + "/" + file_name
  with open(file, 'r') as f:
    data = json.load(f)

    txt_version = file_name.replace('json', 'txt')
    final = './../datasets/ECP/labels/' + train_val + '/' + txt_version
    return(data, final)

def process_data(data):
  """Processes the data from one JSON file.

  @Params:
    data: The JSON data loaded from the file.

  @Returns:
    A string containing the data converted from ECP to YOLOv8 formatting
  """

  image_height = data['imageheight']
  image_width = data['imagewidth']
  to_return = ''

  for item in data['children']:
    if ("occluded>40" not in item['tags']) and ("occluded>10" not in item['tags']):
      # read though each object, grabbing the necessary info
      identity = item['identity']
      if "far-away" not in identity: #for now, lets focus on items that aren't far away
        children = item['children']
        x0, y0, x1, y1 = item['x0'], item['y0'], item['x1'], item['y1']

        # For riders, we want to annotate them as rider+vechicle
        # unless they are skating, in which case we'll call them a skateboarder
        if 'skating' in item['tags']:
          identity = 'skateboarder'
        elif identity == 'rider' and len(children)>0:
          vechicle = children[0]['identity']
          identity = identity + "+" + vechicle

        # Change the identity string into an integer value in the dict
        try:
          identity = identities[identity]
        except KeyError:
          print("changing dictionary: adding " + identity)
          identities[identity] = len(identities)
          identity = identities[identity]

        # Normalize the bounding box coords- YOLO treats an image as 1 by 1
        x0 = x0/image_width
        y1 = y1/image_height
        y0 = y0/image_height
        x1 = x1/image_width

        #YOLO format requires x_center, y_center, x_dif, and y_dif to define a bounding box
        x_center = (x0 + x1)/2
        y_center = (y0 + y1)/2
        x_dif = x1 - x_center
        y_dif = y1 - y_center

        to_return = to_return + (f"{identity} {x_center} {y_center} {x_dif} {y_dif}") + "\n"

    
  return(to_return)

def write_data(filename, data):
  """Writes the data to a text file.

  Args:
    filename: The name of the file to create or overwrite.
    data: The YOLO formatted data to write to the file.
  """

  with open(filename, 'w') as file:
    file.write(data)


if __name__ == "__main__":
    for train_val in ["train", "val"]:
        cities = os.listdir('./../datasets/ECP/old_labels/' + train_val + '/') # retrieve all city names
        
        for city in cities:
            # Take all the files in the old city directory, convert them and put them in the new one
            files = os.listdir('./../datasets/ECP/old_labels/' + train_val + '/'+ city)
            
            for file in files:
                print("processing file " + file)
                ecp_data, new_file = read_data(file, city)[0], read_data(train_val, file, city)[1]
                print("new file = " + new_file)
                write_data(new_file, process_data(ecp_data))

  # write_data('identities_dict.txt', str(identities))
