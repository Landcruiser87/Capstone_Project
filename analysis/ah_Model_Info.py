import pandas as pd
import os
import numpy as np
import json

# reads it back
with open("trial.json","r") as f:
  data = f.read()

# decoding the JSON to dictionay
d = json.loads(data)


# #Returns a list of all of the filenames in the 'raw' directory
# def GetRawFilenames(p):
#     return GetFilenames(p + "raw/")

# #Creates the new name for the file
# def NewFilename_info(file):
#     #exerciseid_amtofexercise_sessionid_subjectid
#     return file.split("/")[-1]

# @keras_export('keras.models.model_from_json')
# def model_from_json(json_string, custom_objects=None):
#   """Parses a JSON model configuration file and returns a model instance.
#   Arguments:
#       json_string: JSON string encoding a model configuration.
#       custom_objects: Optional dictionary mapping names
#           (strings) to custom classes or functions to be
#           considered during deserialization.
#   Returns:
#       A Keras model instance (uncompiled).
#   """
# 	config = json.loads(json_string)
# 	from tensorflow.python.keras.layers import deserialize  # pylint: disable=g-import-not-at-top
# 	return deserialize(config, custom_objects=custom_objects)