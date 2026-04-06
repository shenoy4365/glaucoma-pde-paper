import cv2
import json
import os
import numpy as np
imglist = os.listdir("ROI images")

for imgpath in imglist:
    img = cv2.imread('ROI images/'+imgpath)

    jsonfile = json.load(open('json/'+imgpath.replace('.jpg','.json')))
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours0,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    points  =jsonfile['shapes'][0]
    contours=[]
    for pio in points['points']:
        contours.append([[int(pio[0]),int(pio[1])]])
    contours= tuple([np.array(contours)])
    img = cv2.drawContours(img, contours, -1, (0,255,0), 1)

    points = jsonfile['shapes'][1]
    contours=[]
    for pio in points['points']:
        contours.append([[int(pio[0]),int(pio[1])]])
    contours= tuple([np.array(contours)])


    img = cv2.drawContours(img, contours, -1, (0,0,255), 1)

    cv2.imwrite('Annotated Images/'+imgpath,img)
    #cv2.imshow('drawimg',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()