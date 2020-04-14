import os
import numpy as np
import json
from PIL import Image

def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    '''
    BEGIN YOUR CODE
    '''
    ############# split the image into color channels
    redarr = I[:,:,0]
    greenarr = I[:,:,1]
    bluearr = I[:,:,2] 

    ##### normalization
    sumarr = 0.1*redarr+0.1*greenarr+0.8*bluearr
    mask = np.where(sumarr==0)
    redN = redarr/sumarr
    greenN = greenarr/sumarr
    blueN = greenarr/sumarr
    redN[mask] = 0
    greenN[mask]=0
    blueN[mask] =0

    ##### standardization
    redmax =np.max(redN)
    redmin = np.min(redN)
    redS = (redN-redmin)/(redmax-redmin)
    redS = redS*255
    
    greenmax = np.max(greenN)
    greenmin = np.min(greenN)
    greenS = (greenN-greenmin)/(greenmax-greenmin)
    greenS = greenS*255

    bluemax = np.max(blueN)
    bluemin = np.min(blueN)
    blueS = (blueN-bluemin)/(bluemax-bluemin)
    blueS = blueS*255
    
    shape =redarr.shape

    #### find all the target
    ind = np.where((redS >= np.percentile(redS,90)) & (greenS<np.percentile(greenS,5)) & (blueS<np.percentile(blueS,5)))
    points=[]
    for j in range(len(ind[0])):
        points.append((ind[1][j],ind[0][j]))


    #### find the center of a red light from all nearby targets
    new_points = [] # saves the center 
    width =[]
    height = []
    points_smp = np.array(points)
    while points_smp.shape[0]>1:
        # calculate the distance between all targets
        distance = np.sum((points_smp[0] - points_smp[1:])**2,axis=-1)
        # choose nearby targets
        limit = np.percentile(distance,5)
        same_loc = np.append(0, np.where(distance <=limit)[0] + 1)
        same_ind = points_smp[same_loc]
        # pick the center target
        redS_same = []
        for si in same_ind:
            redS_same.append(redS[si[1],si[0]])
        merged_loc = np.argmax(redS_same) # the index of point
        merged_ind = same_ind[merged_loc] # the x,y of point

     # calculate the size of bounding boxes
        distance_from_center = np.sqrt(np.sum((points_smp[merged_loc] - points_smp[same_loc])**2,axis=-1))
        w0 = int(np.trunc(np.percentile(distance_from_center,50)/2))
        if (w0<20) & (merged_ind[1]<shape[0]/2):
            new_points.append(merged_ind)
            width.append(w0)
            height.append(5*w0)
        points_smp = np.delete(points_smp,same_loc,axis=0)

    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    for k in range(len(new_points)):
        tl_row = int(new_points[k][1]-width[k])
        tl_col = int(new_points[k][0]-width[k])
        br_row = int(new_points[k][1]+height[k])
        br_col = int(new_points[k][0]+width[k])
        bounding_boxes.append([tl_row,tl_col,br_row,br_col])

    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes

# set the path to the downloaded data: 
data_path = './RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

preds = {}
for i in range(len(file_names)):
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
