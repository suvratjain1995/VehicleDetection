import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label

# box_list = pickle.load( open( "bbox_pickle.p", "rb" ))

# Read in image similar to one shown above 
# image = mpimg.imread('test_image.jpg')
# heat = np.zeros_like(image[:,:,0]).astype(np.float)

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def head_pipeline(image,window_list,heat_map):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat,window_list)
    #### Test Image Print #####
    # plt.subplot(131)
    # plt.imshow(heat,cmap = 'gray')
    # plt.imsave('heatmap.jpg',heat,cmap='gray')
    # heat= apply_threshold(heat,4)
    # heat_map = np.clip(heat,0,255)
    # plt.subplot(132)
    # plt.imshow(heat_map,cmap='hot')
    # plt.imsave('thresh.jpg',heat_map,cmap='hot')
    # labels = label(heat_map)
    # plt.subplot(133)
    # plt.imshow(labels[0],cmap = 'hot')
    # plt.imsave('labeled.jpg',labels[0],cmap = 'gray')
    # fig.tight_layout()
    #####
    heat = heat_map.append(heat)
    heat = np.sum(heat_map,axis=0)
    heat = apply_threshold(heat,50)    # print(heat)
    heatmap = np.clip(heat,0,255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.imshow(draw_img)
    # plt.title('Car Positions')
    # plt.subplot(121)
    # plt.imshow(heatmap, cmap='gray')
    # plt.title('Heat Map')
    # fig.tight_layout()
    return draw_img

# Add heat to each box in box list
# heat = add_heat(heat,box_list)
    
# # Apply threshold to help remove false positives
# heat = apply_threshold(heat,1)

# # Visualize the heatmap when displaying    
# heatmap = np.clip(heat, 0, 255)

# # Find final boxes from heatmap using label function
# labels = label(heatmap)
# print(labels[1])
# print(labels[0])
# draw_img = draw_labeled_bboxes(np.copy(image), labels)

# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(draw_img)
# plt.title('Car Positions')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='gray')
# plt.title('Heat Map')
# fig.tight_layout()