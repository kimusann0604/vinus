import numpy as np
import cv2
import math
import skimage.exposure

img = cv2.imread("image.img")

# set location and radius
cx = 130
cy = 109
radius = 30

# set distortion gain
gain = 1.5

# crop image 
crop = img[cy-radius:cy+radius, cx-radius:cx+radius]

# get dimensions
ht, wd = crop.shape[:2]
xcent = wd / 2
ycent = ht / 2
rad = min(xcent,ycent)

# set up the x and y maps as float32
map_x = np.zeros((ht, wd), np.float32)
map_y = np.zeros((ht, wd), np.float32)
mask = np.zeros((ht, wd), np.uint8)

# create map with the spherize distortion formula --- arcsin(r)
# xcomp = arcsin(r)*x/r; ycomp = arsin(r)*y/r
for y in range(ht):
    Y = (y - ycent)/ycent
    for x in range(wd):
        X = (x - xcent)/xcent
        R = math.hypot(X,Y)
        if R == 0:
            map_x[y, x] = x
            map_y[y, x] = y
            mask[y,x] = 255
        elif R >= .90:    # avoid extreme blurring near R = 1
            map_x[y, x] = x
            map_y[y, x] = y
            mask[y,x] = 0
        elif gain >= 0:
            map_x[y, x] = xcent*X*math.pow((2/math.pi)*(math.asin(R)/R), gain) + xcent
            map_y[y, x] = ycent*Y*math.pow((2/math.pi)*(math.asin(R)/R), gain) + ycent
            mask[y,x] = 255
        elif gain < 0:
            gain2 = -gain
            map_x[y, x] = xcent*X*math.pow((math.sin(math.pi*R/2)/R), gain2) + xcent
            map_y[y, x] = ycent*Y*math.pow((math.sin(math.pi*R/2)/R), gain2) + ycent
            mask[y,x] = 255

# remap using map_x and map_y
bump = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT, borderValue=(0,0,0))

# antialias edge of mask
# (pad so blur does not extend to edges of image, then crop later)
blur = 7
mask = cv2.copyMakeBorder(mask, blur,blur,blur,blur, borderType=cv2.BORDER_CONSTANT, value=(0))
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=blur, sigmaY=blur, borderType = cv2.BORDER_DEFAULT)
h, w = mask.shape
mask = mask[blur:h-blur, blur:w-blur]
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5,255), out_range=(0,1))

# merge bump with crop using grayscale (not binary) mask
bumped = (bump * mask + crop * (1-mask)).clip(0,255).astype(np.uint8)

# insert bumped image into original
result = img.copy()
result[cy-radius:cy+radius, cx-radius:cx+radius] = bumped

# save results
cv2.imwrite("portrait_of_mussorgsky2_bump.jpg", result)

# display images
cv2.imshow('img', img)
cv2.imshow('crop', crop)
cv2.imshow('bump', bump)
cv2.imshow('mask', mask)
cv2.imshow('bumped', bumped)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
