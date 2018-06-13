"""
  Import the necessary python paths
  Written by: Peter Tanugraha
"""

import os
import sys

def add_path(path):
    if path not in sys.path:
        print("##### Inserting a new path ######")
        sys.path.insert(0,path)
    else:
        print("Path already exist, not inserting anymore")

cur_dir = os.path.dirname(__file__)
facenet_path = os.path.join(cur_dir,'facenet','src')
print(facenet_path)
add_path(facenet_path)
