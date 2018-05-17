# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_comp.png "Undistortion - Example on a Grid"
[image2]: ./output_images/distcorr.jpg "Input Image, distortion corrected"
[image3]: ./output_images/color_filt.png "Different Color Channels and corresponding threshold filters"
[image4]: ./output_images/sobel_filt.png "Different Sobel filters and Final binary output"
[image5]: ./output_images/warped_lines.png "Warp Example"
[image6]: ./output_images/sliding_windows.png "Sliding Windows Fit"
[image7]: ./output_images/recurrent_fit.png "Recurrent Fit"
[image8]: ./output_images/output.png "Output"
[video1]: ./project_video.mp4 "Project Video"
[video2]: ./challenge_video.mp4 "Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

In this writeup I will focus on the pipeline of an input stream (video). This is basically a description of the file [pipeline.ipynb](./pipeline.ipynb). A single image is computed the same way as an video frame, when the lane lines were not detected in the previous frames (excluding smoothing of lane lines).
The example images are used from a scenario where the left line detection partially fails. It is a good demonstration of the current algorithm's limitation.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration is done once before actually starting to work on the image/video. Several images with chess-boards (9x6) are loaded into the program. An array of touples is created with a location for each found corner in the chessoards. The CV2 function `cv2.findChessboardCorners`  can retrieve the location of each detected corner of each chessboard-image and saves those in another array of tuples. After all the calibration images are analysed the two concatenated arrays are compared using `cv2.calibrateCamera` and several calibration parameters can be obtained. From those only the camera matrix, and the distortion coefficients are used in the further computation, rotation and translation vectors are neglected.

```python
def retrieve_distortion_points(image_shape=(1280, 720)):
    nx = 9
    ny = 6
    c=1
    
    objpoints=[]
    imgpoints=[]
    objp = np.zeros((nx*ny,3), np.float32)
    for i in range(ny):
        for j in range(nx):
            objp[j+nx*i,:]=(j, i, 0)

    # Go through images in folder collect data for calibration
    for fname in glob.glob('camera_cal/*.jpg'):
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        print('Image: ', c, " - Retrieved corners: ", ret )
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        c=c+1
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    return mtx, dist
```

![Undistortion - Example on a Grid][image1]

## Pipeline

### 1. Provide an example of a distortion-corrected image.

Using "undistorted = cv2.undistort(img, mtx, dist, None, mtx)" in the image pipeline, those aforementioned calibration parameters can be applied to each input image or frame. 
![Input Image, distortion corrected][image2]

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


#### Color filtering

After distortion correction, a binary map is to be created through color filtering (`def color_filt(img)`), that highlights the lane lines. For this, the image is transformed into HLS- colorspace  An area in front of the car (roughly the size of 5m x 1m length/wide)  is taken and the high and low Quartil of its HLS-values stored separately. Those two arrays represent the color-value-range of the driving surface in front of the car. The long and narrow shape is chosen, so that changes in the nature of the surface (shadow, different material...) can be found in the values as early as possible while driving. 
```python
imshape=img.shape
pavement_color_low =np.percentile(hls[int(imshape[0]*0.70):int(imshape[0]*0.95),int(imshape[1]*0.45):int(imshape[1]*0.55),:],25, axis=(0,1))
pavement_color_high=np.percentile(hls[int(imshape[0]*0.70):int(imshape[0]*0.95),int(imshape[1]*0.45):int(imshape[1]*0.55),:],75, axis=(0,1))
```
With the help of those values two color channels were used, to create a final color threshold image:
* **Saturation threshold:** This is the main color channel for the color filtering. Everything that is more saturated than the most saturated Quartil of the driving surface (including a safety margin of 40)  could be a lane line.
```python
	s_channel = hls[:,:,2]

	s_thresh_min = pavement_color_high[2]+40
	s_thresh_max = 255
    
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
```
* **Hue threshold:** Only colors in that range are used for lane detection in the saturation channel. One could argue that white stripes with a blueish tint would also be deleted by this, but in that case the edge detection mechanism should capture at least the outline of those stripes. Furthermore, in the example videos all of the white lines had a slight yellow tint to it, probably by dust and aging under the sun.
```python
    h_thresh_min = 10
    h_thresh_max = 60
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh_min) & (h_channel <= h_thresh_max)] = 1
```
This is then combined for a color binary image:
```python
    comb_binary_c=copy.copy(s_binary)
    comb_binary_c[h_binary==0]=0
```
The information about the road surface color is used to create a new RGB image of the frame neglecting everything that is darker ("less light") than the brightest surface Quartil. By this, cracks and uneven patches of the surface are deleted (or at least muted) and can no longer throw off the following edge detection mechanism.
```python
    lnew=np.copy(l_channel)
    lnew[l_channel<pavement_color_high[1]]=pavement_color_high[1]
	new_hls=np.dstack((h_channel,lnew,s_channel))
    img_new=cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)
``` 

![Different Color Channels and corresponding threshold filters][image3]
#### Edge detection using Sobel-gradient-filter
The altered image is loaded into the Sobel-gradient-filter for edge detection.
There three filters are used to calculate a high likelihood of an edge created by a lane line:
- **Gradient magnitude filter:** Only taking into account the horizontal edges with a certain strength *-> Later illustrated as red*
-  **Directional gradient filter:** Only taking into account the edges with slopes between ~35° and ~52° -*> Later illustrated as green*
- **Total magnitude gradient filter** Taking into account all stronger edges  *-> Later illustrated as blue*

(Values in the function definition)
```python
def sobel_thresh(gray, sobel_kernel=7, orient='x', orient_thresh=(30, 100), dir_thresh=(0.7, 1.3), mag_thresh=(50, 255)):
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    #Gradient magnitude in one direction
    orient_binary=np.zeros_like(gray)
    if orient=='x':
        abs_sobel=np.absolute(sobelx)
    if orient=='y':
        abs_sobel=np.absolute(sobely)
        
    abs_sobel=np.uint8(255*abs_sobel/(np.max(abs_sobel)))
    flags_o=(abs_sobel >= orient_thresh[0])&(abs_sobel <=orient_thresh[1])
    orient_binary[flags_o]=1
    
    #Directional gradient filter
    dir_binary=np.zeros_like(gray)
    sobelxy=np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    flags_d=(sobelxy >= dir_thresh[0])&(sobelxy <=dir_thresh[1])
    dir_binary[flags_d]=1
    
    #Total magnitude gradient filter
    mag_binary=np.zeros_like(gray)
    sobelm=np.sqrt(sobelx*sobelx+sobely*sobely)
    sobelm=np.uint(255*sobelm/np.max(sobelm))
    flags_m=(sobelm >= mag_thresh[0])&(sobelm <=mag_thresh[1])
    mag_binary[flags_m]=1
        
    return orient_binary, dir_binary, mag_binary
```
Only if two of those filters find an edge, it will be taken into account as part of a lane line. Finally the binary image from the color filter is (boolian) added.
```python
	comb_binary = np.zeros_like(gray)
    twos=np.ones_like(gray)*2
    comb_binary[(orient_binary+dir_binary+mag_binary)>=twos] = 1
   
    #add detected lanes from color filter
    comb_binary[(comb_binary_c == 1)]= 1
```


![Different Sobel filters and Final binary output][image4]

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
A trapezoid area of the frame is transformed into a squared top-down view of the road with a side length of `osize=1024`.

The code has to work on the "project video" as well as on the "challenge video". While it is easy to look far ahead in the project video, the challenge video has sharper corners and an period of uphill driving,  introducing disturbances in the area in which a lane line could be detected. Because of this, three different scenarios are introduced here, each represented by different location of the top edge of the trapezoid in the frame:
```python
#Parameters of warping Trapezoid--> Square
    
#Lower left corner
low_y=670
low_x=255
    
#Upper left corner
if horizon==0: #long, wide highway (project video)
	top_y=440
    top_x=607
    o_sidemargin=osize/3 #margin if something is found left and right of the center lane (--> Curve)
        
if horizon==1: #hilly highway (challenge video) 
	top_y=460
    top_x=580
    o_sidemargin=osize/4.5
        
if horizon==2: #countryside_road (harder challenge video)
    top_y=500
    top_x=510
    o_sidemargin=osize/3 #margin if something is found left and right of the center lane (--> Curve)
    
    
    
src=np.float32([[low_x,low_y],
                [top_x,top_y],
                [image_width-top_x,top_y],
                [image_width-low_x,low_y]])
dst=np.float32([[o_sidemargin,osize],
                [o_sidemargin,0],
                [osize-o_sidemargin,0],
                [osize-o_sidemargin,osize]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```
This results in the following source and destination points (neglecting the change of the margin on the side in the case of hilly highways):

| Source        | Destination   | 
|:-------------:|:-----------------:| 
| `top_x` , `top_y`     | 341, 0        | 
| 255, 670      | 341, 1024      |
| 1025, 670     | 683, 1024     |
| 1280-`top_x` , `top_y`      | 683, 0        |


![Warp Example][image5]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For this, I adapted the example code. There are two procedures depending on whether there already exist Polynomes that realistically approximate the position of the lane lines.

If a new lane line has to be detected I created a function called `initial_line_detector`  that takes the binary image and returns two best fitting 2nd degree Polynomes for this image's lane lines. I won't go into detail of the code, as it was discussed in the lecture, but where I introduced a few modifications of tried new things: 

- for most frames it is much more precise to use the lower quarter or even less to build the histogram for the starting point of the window search, however this whole function is used on frames when the other function failed (except for the first frame), so it has to be robust. Thus the value remained at its original 50% 
```python
start_of_hist=int(binary_warped.shape[0]*0.5)
histogram = np.sum(binary_warped[start_of_hist:,:], axis=0)
```
- using a squared binary warped image, there are more pixels in y direction to identify as part of a lane line. Therefore the number of sliding windows is changed from 9 to 15
- my previous filters are quite aggressive, leaving rather empty binary maps, so I lowered the minimum number of pixels found to recenter each sliding window from 50 to 10
- I created a penalty counter, for when the lane (and thus one window boundary) leaves the warped binary image horizontally  `out_r=0, out_l=0`.  Each window can leave the frame only up to 50% before becoming penalized by +2. If a following window slides back into the frame: -1. Only if the penalty is lower than 4 and the window is not completely filled with pixels, it will be appended to the corresponding line indices
```python
if (win_xleft_low<-margin/2) or (win_xleft_high>binary_warped.shape[1]+margin/2):
	out_l=out_l+2
elif out_l>0:
	out_l=out_l-1
```

```python
if out_l<4 and (len(good_left_inds) < window_height*margin*1.8):
	left_lane_inds.append(good_left_inds)
```

- in all following calculations the y-coordinate of the line indices/pixels is swapped upside-down. This way the resulting Polynome is easier to work with: ay²+by+c=x, where c is the horizontal lane-line position in front of the car. a,b are the slope parameters. Now changes of the parameters between frames are mainly limited to a and b.
- Additionally, in case no lane line was detected (less that 3 detected pixels) I let the function return an array of [0,0,0]

This is how the result would look

![Sliding Windows Fit][image6]

If there already exist Polynomes that realistically approximate the position of the lane lines in previous frames, the function ``recurrent_line_detector`` is called, which takes the binary image and the coefficients of the approximated fits and returns the new fit coefficients. Those coefficient where found using all pixels in a margin from the last frames lane line.

I did not make alterations here. In case no lane line was detected (less that 3 detected pixels) I let the function return an array of [0,0,0]

![Recurrent Fit][image7]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Again I took the proposed version of the code and only made small changes to ist:
- Here, the factors that transform the lane pixels to real world dimensions have to be adjusted accordingly, as I used different parameters for the warping of the binary image, depending on the road.  Again I used different values depending on whether the road is wide open, a hilly highway or a countryside road.
- at the end I calculated the offset as the mean of the first coefficients, normed to zero in real world dimension.  Thus, a positiv number indicates driving on the left side within the lane.
```python
def find_curvature_and_offset(left_fitx, right_fitx, horizon):
    y_eval = np.max(ploty)
    if horizon==0:
        ym_per_pix = 15/y_eval # pixel per meter in y dimension
        xm_per_pix = 3.7/y_eval*0.38 # meters per pixel in x dimension
        
    if horizon==1:
        ym_per_pix = 20/y_eval # pixel per meter in y dimension
        xm_per_pix = 3.7/y_eval*0.65 # meters per pixel in x dimension
        
    if horizon==2:
        ym_per_pix = 10/y_eval # pixel per meter in y dimension
        xm_per_pix = 3.7/y_eval*0.38 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    
    # Calculate the new radii of curvature 
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    
    offset=(left_fitx[0]+ right_fitx[0]-osize)/2.0*xm_per_pix
    
    return left_curverad, right_curverad, offset
```

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Again I used the code given as an example in the classroom. I added text overlay in which the mean curvature and the offset of the center of the lane is shown. (added part from `def create_lane_marked(...)`)
```python
  
    # Create a mean curvature and bring it into a nice form
    mean_curve=(left_curverad+right_curverad)//2
    if mean_curve>9800:
        mean_curve_s='inf'
    else:
        mean_curve_s=str(int(round(mean_curve, -2)))
    
    #Name offset side
    if offset>0:
        lor='left'
    else:
        lor='right'

    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Mean Curvature : %s m' %mean_curve_s, (10,650), font, 1, (0,0,0),3)
    cv2.putText(result,'Offset: %10.4f m %s of center' %(np.abs(offset), lor), (10,680), font, 1, (0,0,0),3)
    return result

```
![Output][image8]

---


### 7. Putting it all together

Up until now we are only looking at single frames. When working with videos a pipline was introduced to handle each frame.
Using  `VideoFileClip` from `moviepy.editor` each frame and its time-position in the movie is processed in this main pipeline(similar to the first Lane detection project) .

First, the image is loaded, undistorted, filtered and warped.  A frame number is computed from the time.
```python
	f=int(t*Framerate) 
    img=get_frame(t)
	undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    binary=filter_for_lane_lines(undistorted)
    binary_warped = cv2.warpPerspective(binary, M, (osize,osize), flags=cv2.INTER_LINEAR)
```
If its the first frame of the movie or if in the previous frame the line was not detected `initial_line_detector` is called, otherwise the current fit is used and modified by `recurrent_line_detector`
```python
if f<1 or R_line.detected==False or L_line.detected==False:
	#using sliding windows
    L_line.current_fit, R_line.current_fit=initial_line_detector(binary_warped)
else:
     #using an area around a predefined, known lane marking from the previous frame
     L_line.current_fit, R_line.current_fit=recurrent_line_detector(binary_warped, L_line.best_fit, R_line.best_fit)
   
# Reset Line Detection
L_line.detected=True
R_line.detected=True
```
```python
if f==0:
	#Initialization in first frame
	L_line.best_fit=L_line.current_fit
    R_line.best_fit=R_line.current_fit
```
The new fit is then rated whether it represents a detected lane line:
1.  If each detected lane line curves in the same direction as the current lane
2.  If the left line starts on the left side and the right line on the right side
3. If each fit resembles the averaged best fit of the line within a certain margin

Each time one case fails, a failed line detection is noted.
In case 3 the current fit would then be replaced by the best (averaged) fit
In case 1 the current fit would be replaced by the best (averaged) fit, and the difference between the current fit and the best (averaged) fit of the other line (left<-> right) would be added. This assumes that only one line detection fails at a time and that both lines should behave comparably. If this step does not work as intended, it is detected in case 3. 

```python
if f>0:
	L_line.diffs=L_line.current_fit-L_line.best_fit
    R_line.diffs=R_line.current_fit-R_line.best_fit
        
    # Newly detected line compared to line on other side, considering divergence
    avg_curve=(L_line.best_fit[0:1]+R_line.best_fit[0:1])/2.0
    if (L_line.current_fit[0]>0 and avg_curve[0]<0) or\
		    (L_line.current_fit[0]<0 and avg_curve[0]>0):
	    L_line.current_fit=L_line.best_fit+R_line.diffs
        L_line.detected=False
    if (R_line.current_fit[0]>0 and avg_curve[0]<0) or\
		    (R_line.current_fit[0]<0 and avg_curve[0]>0):
		 R_line.current_fit=R_line.best_fit+L_line.diffs
         R_line.detected=False
                
# Newly detected line starting point verification (just before car)
    if L_line.current_fit[2]>osize/2 or L_line.current_fit[2]<-100:
        L_line.detected=False
    if R_line.current_fit[2]>osize+100 or R_line.current_fit[2]<osize/2:
        L_line.detected=False
                
#Newly polynom of detected line differs from averaged previous lines
    if np.any(np.abs(L_line.diffs)>(1e-3, 2e-1, 50)):
        L_line.detected=False
        L_line.current_fit=L_line.best_fit
    if np.any(np.abs(R_line.diffs)>(1e-3, 2e-1, 50)):
       R_line.detected=False
       R_line.current_fit=R_line.best_fit
```
Next, this current line is added to an exponential moving average with a factor `av` set to 5 (The current values influence the average with a weight of 20%) if the line was marked as detected. If there was no line detected the current fit is still added to the moving average, but with a quarter of the weight. This way, if there is a sudden change in the shape or position of the lane line, after about 20 frames it has influenced the average enough to become the new best fit. A single (or very few) incorrect detection has a negligible influence on the lane line
```python
if  L_line.detected==True:
	#If line is detected add it to the exponential average
	L_line.best_fit=1.0/av*L_line.current_fit+(1-1.0/av)*L_line.best_fit
else:
	#If line is not detected still add it to the exponential average, but with a quarter of the weight
	L_line.best_fit=1.0/(av*4)*L_line.current_fit+(1-1.0/(av*4))*L_line.best_fit
                
if R_line.detected==True:
	#If line is detected add it to the exponential average
	R_line.best_fit=1.0/av*R_line.current_fit+(1-1.0/av)*R_line.best_fit
else:
	#If line is not detected still add it to the exponential average, but with a quarter of the weight
	L_line.best_fit=1.0/(av*4)*L_line.current_fit+(1-1.0/(av*4))*L_line.best_fit
```
With this averaged, best fitted parameters a line is created and added to an array of the last five (`av`=5) fits
```python
left_fitx = L_line.current_fit[0]*ploty**2 + L_line.current_fit[1]*ploty + L_line.current_fit[2]
ight_fitx = R_line.current_fit[0]*ploty**2 + R_line.current_fit[1]*ploty + R_line.current_fit[2]

#add current line (if there was a line detected in the recent frames) to an array of 5 lines
if f<av:
	L_line.xfitted[f,:]=left_fitx
elif not(np.any(L_line.best_fit==(0,0,0))):
	L_line.xfitted=np.roll(L_line.xfitted, 1,axis=0)
	L_line.xfitted[0,:]=left_fitx
        
if f<av:
	R_line.xfitted[f,:]=right_fitx
elif not(np.any(R_line.best_fit==(0,0,0))):
	R_line.xfitted=np.roll(R_line.xfitted, 1,axis=0)
	R_line.xfitted[0,:]=right_fitx
```
The median of those lines is used to compute the curvature and offset of the lines and for fiting a polynomial in the real world dimensions and plotting the lane in the final output frame. Before displaying the curvature, it is smoothed using a exponential average similar to before.

```python
#find curvature of best current fit (median-values of last n lines that where found; median dismisses 'hiccups')
l_curve, r_curve , offset=find_curvature_and_offset(np.median(L_line.xfitted, axis=0), np.median(R_line.xfitted, axis=0),horizon)
    
    
# even out the curvature over the last secound (roughly) by using exponential average, takes care that Radius will never exceed 10000    
L_line.radius_of_curvature=1.0/(av*5)*np.minimum(l_curve,10000)+(1-1.0/(av*5))*L_line.radius_of_curvature
R_line.radius_of_curvature=1.0/(av*5)*np.minimum(r_curve,10000)+(1-1.0/(av*5))*R_line.radius_of_curvature
       
#create result
result=create_lane_marked(img,binary_warped, np.median(L_line.xfitted, axis=0), np.median(R_line.xfitted, axis=0), L_line.radius_of_curvature, R_line.radius_of_curvature, offset)
   
```


## Output

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./project_video.mp4)
Here's a [link to my challenge video result](./challenge_video.mp4)

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
#### Lessons learned
It took me a very long time to finish this project. Partly because of private reasons but also because of reasons that arose from working on this project:
- being a mechanical engineer (and a rather "traditional" direction of it) it was the first time I had to handle that much code which needed improvement in several parts at once. Looking back, I realized that I never really used a strategic approach but once it was running I often time "fiddled" around until it was satisfactory. This is very time ineffective.
- I was aiming to high and tried to make in work on the "brutal" harder challenge video. I failed and only then went back to work on the basics.

#### Where the pipeline will likely fail:
- Currently the car goes shortly  blind when in leaves or enters areas with very different lightning conditions. Only after receiving the road surface colors from the area before it, is going to pick up lane lines again. This could face a problem with a lot of sudden changes in the lightning such as driving through a forest (as in the hardest video). 
- When the offset of the car from the lane center ("cutting corners") becomes too big the curvatures of the right and left lines are going to differ more and the algorithms that identify a successful line detection could fail.
- If a strong curvature of the road (curvy road) pushes most part of the inner lane line out of the camera image the pipeline currently does not know how to handle the situation.

#### Parts that could be improved:
- It is a decision to take where most of the filtering takes place. Either the filtered binary images are very clean but maybe lack features that belong to a lane line, or the binary images still contain a lot of information and the correct positions of the lane lines are determined by a more sophisticated line detection algorithm (sliding windows...).
- I worked with fixed values or parameters (margins, corner-points of the transformation map, averaging factors...). Those could be computed as a function of the current lane curvature. That way the same lane detection could be applied on hilly countryside roads and wide, straight highways without change in the parameters.
- Color filtering and sobel gradient filtering are both convolutions in some sense. Color filtering is a 1x1 convolution in the color channels while sobel gradient filters are convolutions within one channel.  It would be very effective to use very small convolutional network to "carve out" the lane lines in the pictures. It should have a maximum depth of maybe 3 convolutions and could be trained on a bunch of manually cleaned lane line binary images (maybe with one thickened and one thinned version each, so that the results always lie in between those versions). Those simple convolutional weights could than be hard-coded into this pipeline.
