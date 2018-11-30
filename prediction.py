#%%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from imageprocess import *
from moviepy.editor import VideoFileClip
from hotmap import *
# load a pe-trained svc model from a serialized (pickle) file

# get attributes of our svc object

#%%
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,vis=False):
    window_list = []
    draw_img = np.copy(img)
    # img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    if(vis == True):
        plt.imsave('channel1.jpg',ch1)
        plt.imsave('channel2.jpg',ch2)
        plt.imsave('channel3.jpg',ch3)


    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    if(vis != True):
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog1,hog1_image = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False,vis=vis)
        hog2,hog2_image = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False,vis=vis)
        hog3,hog3_image = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False,vis=vis)
        plt.imsave('channel1_hog.jpg',hog1_image)
        plt.imsave('channel2_hog.jpg',hog2_image)
        plt.imsave('channel3_hog.jpg',hog3_image)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                window_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return window_list

#%%
def load_pickles():
    with open('model.pickle','rb') as f:
        svc = pickle.load(f)
    with open('parameter.pickle','rb') as f:
        parameter_dict = pickle.load(f)
    return svc,parameter_dict

count=0
#%%
def pipeline(image):
    # svc = None
    # parameter_dict = {}
    # with open('model.pickle','rb') as f:
    #     svc = pickle.load(f)
    # with open('parameter.pickle','rb') as f:
    #     parameter_dict = pickle.load(f)
    X_scaler = parameter_dict["X_scaler"]
    orient = parameter_dict["orient"]
    pix_per_cell = parameter_dict["pix_per_cell"]
    cell_per_block = parameter_dict["cell_per_block"]
    spatial_size = parameter_dict["spatial_size"]
    hist_bins = parameter_dict["hist_bins"]
    window1=find_cars(image, 400, 480, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,vis=False)
    window2=find_cars(image, 400, 520, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,vis=False)
    window3=find_cars(image, 400, 550, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,vis=False)
    window4=find_cars(image, 400, 580, 1.7, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,vis=False)
    final_window = window1+window2+window3+window4
    # print(len(final_window))
    # for windows in window1:
    #     # print(windows[0],windows[1])
    #     cv2.rectangle(image,windows[0],windows[1],(0,0,255),6)
    # for windows in window2:
    #     # print(windows[0],windows[1])
    #     cv2.rectangle(image,windows[0],windows[1],(0,255,0),6)
    # for windows in window3:
    #     # print(windows[0],windows[1])
    #     cv2.rectangle(image,windows[0],windows[1],(255,0,0),6)
    # for windows in window4:
    #     # print(windows[0],windows[1])
    #     cv2.rectangle(image,windows[0],windows[1],(120,255,0),6)
    # plt.imshow(image)
    # plt.imsave('classified_car.jpg',image)
    
    result = head_pipeline(image,final_window,heat_map)
    # cv2.imwrite('output_images/image'+str(count+1),result)
    plt.imshow(result)
    return result
    
#%%
# y_margins = [(400,480),(400,520),(400,550),(400,580)]
svc,parameter_dict=load_pickles()
test_image = mpimg.imread("test_images/test5.jpg")
from collections import deque
heat_map = deque(maxlen=25)
pipeline(test_image)
heat_map = deque(maxlen=25)


#%%

def video():
    white_output = "output.mp4"
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    %time white_clip.write_videofile(white_output, audio=False)

#%%

video()