import os
import numpy as np
import json
from PIL import Image, ImageDraw

def draw_bounding_boxes(Im, bbox):
    draw = ImageDraw.Draw(Im)
    for ibox in bbox:
        tl_row,tl_col,br_row,br_col = ibox
        draw.line([(tl_col,tl_row),(tl_col,br_row)],fill=(255,0,255),width=5)
        draw.line([(tl_col,tl_row),(br_col,tl_row)],fill=(255,0,255),width=5)
        draw.line([(br_col,br_row),(br_col,tl_row)],fill=(255,0,255),width=5)
        draw.line([(br_col,br_row),(tl_col,br_row)],fill=(255,0,255),width=5)
        
        point_col= (br_col + tl_col)/2
        width = point_col - tl_col
        point_row = tl_row + width
        draw.line([(point_col-5,point_row-5),(point_col+5,point_row+5)],fill=(0,255,0),width=5)
        draw.line([(point_col+5,point_row-5),(point_col-5,point_row+5)],fill=(0,255,0),width=5)


data_path = './RedLights2011_Medium'

# path to json: 
preds_path = './hw01_preds' 
out_path = '/test_results/'
# os.makedirs(out_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# read in all bounding boxes from json
with open(preds_path+'/preds.json', 'r') as f:
    bounding_boxes = json.load(f)

# plot magenta boxes on images. The center of red lights are marked with a green cross.
for i in range(len(file_names)):
     # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    box = bounding_boxes[file_names[i]]
    draw_bounding_boxes(I, box)
    
    I.save(os.path.join('./'+preds_path+out_path,file_names[i]),"JPEG")


       