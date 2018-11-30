## Writeup 
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car]: ./output_images/1.png
[noncar]: ./output_images/image1.png
[test]: ./output_images/test5.jpg
[carchannel1]: ./output_images/channel1_car.jpg
[carchannel2]: ./output_images/channel2_car.jpg
[carchannel3]: ./output_images/channel3_car.jpg
[carhog1]: ./output_images/channel_hog_car1.jpg
[carhog2]: ./output_images/channel_hog_car2.jpg
[carhog3]: ./output_images/channel_hog_car3.jpg
[noncarchannel1]:./output_images/channel1_notcar.jpg
[noncarchannel2]:./output_images/channel2_notcar.jpg
[noncarchannel3]:./output_images/channel2_notcar.jpg
[noncarhog1]: ./output_images/channel_hog_notcar1.jpg
[noncarhog2]: ./output_images/channel_hog_notcar2.jpg
[noncarhog3]: ./output_images/channel_hog_notcar3.jpg
[prediction]: ./output_images/classified_car.jpg
[heat]: ./output_images/heatmap.jpg
[thres]: ./output_images/thresh.jpg
[label]: ./output_images/labeled.jpg
[final]: ./output_images/predictionbox.png




## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting HOG Features from training image is as follows

```python

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```
Paramters chosen from this were

```python
parameter_dict ={}
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
parameter_dict["orient"] = orient
pix_per_cell = 12 # HOG pixels per cell
parameter_dict["pix_per_cell"] = pix_per_cell
cell_per_block = 2 # HOG cells per block
parameter_dict["cell_per_block"] = cell_per_block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
parameter_dict["spatial_size"] = spatial_size
hist_bins = 32    # Number of histogram bins
parameter_dict["hist_bins"] = hist_bins
```

To extract hog features we used sklearn library *HOG* that takes in parameters mentioned above and return hog features either as feature vector or a 5-d array.We could also visulaize the result of HOG by setting visualization argument to true.

Library 
```python
from sklearn.image import hog
```
#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the parameters chosen above the best accuray for my classifier and minimum false result for my test images.

##### Output of Hog #####

**Car**

![car] 

**Channels**

![carchannel1]  ![carchannel2] ![carchannel3]

**HOG**

![carhog1] ![carhog2] ![carhog3]

**Not Car**

![noncar] 

**Channels**

![noncarchannel1] ![noncarchannel2] ![noncarchannel3]

**HOG**

![noncarhog1] ![noncarhog2] ![noncarhog3]



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained using SVM linear Classifier using **Hog features, spatial features,color histogram features**

1 . The following code will bring out the feature extraction as a whole from the pickled image file.

```python
def extract_features(imgs_pickle, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    images_list =[]
    with open(imgs_pickle,'rb') as f:
        import pickle
        images_list = pickle.load(f)
    for image in images_list:
        file_features = []
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
```

Steps that the previous code defines are :- 

1. Read the list of images from the pickle file.

2. Change the color channel of the image depending on the argument passed.Here we chose as YCrCb

3. Extrat Spatial feature,this feature is extracted by following function 

    ```python
    def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features
    ```
4. Extract Histogram Feature, feature is extracted by following function

    ```python
    def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

    ```
5. Extract HOG features depending upon number of channels we want to extract from . The following code as been mentioned above.

6. Combine all the features together and return it.


2 . I normalized my feature vector using following code

```python
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rand_state)
        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    parameter_dict["X_scaler"] = X_scaler
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    ```
    The *StandardScalar().fit() is using to fit the scaling function on training data first and then 
    Fucntion.transform is used to convert the training and testing data to the normalized format.
```

3 . I trained my SVM linear Classifier on the following Train and test data as follows and saved it to *model.pickle* file.

```python
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
import pickle
with open('model.pickle','wb') as f:
    pickle.dump(svc,f)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the approach suggested in the hog subsampling lesson.The hog Image was partioned into 4 strips ranging in differnet height. The Strips was then scaled depending upn the scaling number. No of windows are calculated depending scaled strip.The window is the rescaled back to the size of the trained image and is then using svc model predict if that window is a of car or not.If the window is part of the car the windows cordinates are then recorded into a list and passed for further filtering to remove false data.

Here is the following code to describe the function :- 

```python
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
```



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. 

**Actual Image**
![test]
**Predicted boxes**
![prediction]
**Filtered out Result**
![final]
---

Here is the Pipeline Code.

```python
def pipeline(image):
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
    result = head_pipeline(image,final_window,heat_map)
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
```

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

I also made a queue for storing 25 frames and then summed those frame and then added threshold of 50 to remove the false outcomes and also to smooth out the boxes.

### Here corresponding heatmaps:

![alt text][heat]

### Here is the output after thresholding the number of boxes:

![alt text][thres]
### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap:
![alt text][label]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Discussion :- 

1. Chosing right window size and optimal threshold was a tremendous challenge.

2. Chosing HOG parameters to get the best accuracy of Linear SVC was also a challenge

3. Chosing Number of windows and cells per steps for sliding window was also a challenge. It was important that we tune the that parameter to ignore the small cars on the left lane.

4. The Pipeline is sometime detecting the car of the opposite lane and that can be removed by adding filter to x axis of the image,but that would not generalize the solution.

5. The Pipeline is likely to fail on images of cars that hasnt been trained on and hence the linear svc will not be able to detect it.




