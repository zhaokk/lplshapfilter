import numpy as np
import cv2
import math
from math import atan2, degrees, pi
import os
import matplotlib.pyplot as plt #plt.plot(x,y) plt.show()


def pointDis(p0,p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

def getDegree(cut):
    #culculate the average lane
    x2 = cut[0][0][0]+cut[3][0][0]
    x1 = cut[1][0][0]+cut[2][0][0]
    y1 = cut[1][0][1]+cut[2][0][1]
    y2 = cut[0][0][1]+cut[3][0][1]
    width = pointDis([x1,y1],[x2,y2])
    rx2 = cut[0][0][0]+cut[1][0][0]
    rx1 = cut[3][0][0]+cut[2][0][0]
    ry1 = cut[3][0][1]+cut[2][0][1]
    ry2 = cut[0][0][1]+cut[1][0][1]
    height = pointDis([rx1,ry1],[rx2,ry2])
    if(width<height):
        media = width
        width = height
        height = media
        x2 = rx2
        x1 = rx1
        y1 = ry1
        y2 = ry2
    dx = x2 - x1
    dy = y2 - y1
    rads = atan2(-dy, dx)
    rads %= 2 * pi
    degs = -degrees(rads)
    return width,height,degs

def getWHR(cut):
    #culculate the average lane
    x2 = cut[0][0][0]+cut[3][0][0]
    x1 = cut[1][0][0]+cut[2][0][0]
    y1 = cut[1][0][1]+cut[2][0][1]
    y2 = cut[0][0][1]+cut[3][0][1]
    width = pointDis([x1,y1],[x2,y2])
    rx2 = cut[0][0][0]+cut[1][0][0]
    rx1 = cut[3][0][0]+cut[2][0][0]
    ry1 = cut[3][0][1]+cut[2][0][1]
    ry2 = cut[0][0][1]+cut[1][0][1]
    height = pointDis([rx1,ry1],[rx2,ry2])
    if(width<height):
        media = width
        width = height
        height = media
    ratio = width/height
    qualify = False
    if(ratio>2 and ratio<5):
        qualify = True
    return qualify

def shapeFilter(img):
    height,width,channel = img.shape
    maxSize = int(height*width/40)
    minSize = int(height*width/1000)
    print(maxSize)
    #forth value to adjust threshold for areas the bigger the more
    #0.2 plate have same color like car
    #dark area

    mser = cv2.MSER_create(10,minSize,maxSize,0.25)
    #mser = cv2.MSER_create(10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()

    regions = mser.detectRegions(gray, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    i = 0
    cts = []
    for hull in hulls:
        epsilon = 0.05 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if(len(approx)==4):
            cts.append(approx)
    centers = []
    centers.append([0,0])
    Newcts = []
    cropImages = []
    for hull in cts:
        biggestX = 0
        biggestY = 0
        smallestX = hull[0][0][0]
        smallestY = hull[0][0][1]
        for point in hull:
            x = point[0][0]
            y = point[0][1]
            #cv2.circle(vis, (x, y), 1, (0, 0, 255), -1)
            if(x>biggestX):
                biggestX=x
            if(x<smallestX):
                smallestX = x
            if(y>biggestY):
                biggestY=y
            if(y<smallestY):
                smallestY=y
        #cv2.circle(vis, (smallestX, smallestY), 1, (0, 0, 255), -1)
        center = [int((biggestX + smallestX) / 2), int((biggestY + smallestY) / 2)]
        rectangle = True
        tooClose = False
        for acenter in centers:
            if(pointDis(acenter,center)<5):
                tooClose = True
        if not (tooClose):
            distanceSum = 0
            heightSum = 0
            widthSum = 0
            for point in hull:
                x = point[0][0]
                y = point[0][1]
                distanceSum =distanceSum + pointDis(center,point[0])
            averDis = distanceSum/4
            for point in hull:
                ratio = pointDis(center,point[0])/averDis
                if(ratio>1.2 or ratio<0.8):
                    print("not rectangle")
                    rectangle = False
            if(rectangle):
                if(getWHR(hull)):
                    centers.append(center)
                    cv2.circle(vis, (center[0], center[1]), 1, (0, 0, 255), -1)
                    Newcts.append(hull)
                    mask = np.zeros((height,width), np.uint8)
                    cv2.drawContours(mask, [hull], 0, (255,255,255), -1)
                    cv2.polylines(vis, Newcts, 1, (0, 255, 0))
                    nWidth, nHeight,fltCorrectionAngleInDeg=getDegree(hull)
                    rotationMatrix = cv2.getRotationMatrix2D((center[0],center[1]), fltCorrectionAngleInDeg, 1.0)
                    imgRotated = cv2.warpAffine(vis, rotationMatrix, (width, height))
                    imgCropped = cv2.getRectSubPix(imgRotated, (int(nWidth * 1.2), int(nHeight * 1.2)),(center[0], center[1]))
                    cropImages.append(imgCropped)
                    Newcts = []
                else:
                    print("width / height ratio not correct")
            else:
                print("not rectangle")
        else:
            print("to close")
    return cropImages
    # cv2.imshow('img', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def test():

    root = '/home/connor/honour/carsalecompare'
    y = 0
    z=0
    for subdir, dirs, files in os.walk(root):
        for dir in dirs:
            i=0
            for subdir, dirs, files in os.walk(root + '/' + dir):
                for file in files:
                    filePath = root + '/' + dir+ '/' + file
                    imgOriginalScene = cv2.imread(filePath)
                    height,width,channels = imgOriginalScene.shape
                    if(width<400):
                        os.rename(filePath,root+'/'+dir+"/tosmall"+str(i))
                        cv2.imshow("deleted image", imgOriginalScene)
                        cv2.waitKey(100)
                    else:
                        filteResult = shapeFilter(imgOriginalScene)
                        if filteResult is not None:
                            for imc in filteResult:
                                i = i + 1
                                cv2.imwrite(root + '/' + dir + '/posiblPlate' + str(i) + '.png', imc)
def testOne():
    filteResult = shapeFilter(cv2.imshow('1.jpg'))
    if filteResult is not None:
        for imc in filteResult:
            cv2.imshow('img', imc)
            cv2.waitKey(0)
test()
