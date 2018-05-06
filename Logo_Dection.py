# import cv2
# import numpy as np
# img = cv2.imread('ut-dallas-logo.jpg')
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray,None)
# img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg',img)
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
# img1 = cv2.imread('ut-dallas-logo.jpg',0)          # queryImage
# img2 = cv2.imread('T_shirt.jpg',0) # trainImage
# # Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
# plt.imshow(img3),plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE =0
flannParam = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
flann = cv2.FlannBasedMatcher(flannParam,{})

#load the training image
trianImg = cv2.imread('ut-dallas-logo.jpg',0) # trainImage
trainKp, trainDesc = detector.detectAndCompute(trainImg,None)

cam = cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR = cam.read()
    QueryImg = cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc = detector.detectAndCompute(QueryImg,None)
    matches = flann.knnMatch(queryDesc,trainDesc,k= 2)
    
img1 = cv2.imread('T_shirt.jpg',0)          # queryImage
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
