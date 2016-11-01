import numpy as np
import cv2
import math
from math import atan2, degrees, pi
import os
import PossiblePlate
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
    cos = math.cos(rads)
    return width,height,cos,degs


def GetAngleOfLineBetweenTwoPoints(x1,y1, x2,y2):
    xDiff = x2 - x1
    yDiff = y2 - y1
    return (atan2(yDiff, xDiff)%(pi))

def  checkParallel(cut):
    #culculate the average lane
    paraPair11 = GetAngleOfLineBetweenTwoPoints(cut[0][0][0],cut[0][0][1],cut[1][0][0],cut[1][0][1])
    paraPair12 = GetAngleOfLineBetweenTwoPoints(cut[2][0][0],cut[2][0][1],cut[3][0][0],cut[3][0][1])
    paraPair21 = GetAngleOfLineBetweenTwoPoints(cut[1][0][0],cut[1][0][1],cut[2][0][0],cut[2][0][1])
    paraPair22 = GetAngleOfLineBetweenTwoPoints(cut[0][0][0],cut[0][0][1],cut[3][0][0],cut[3][0][1])
    if(abs(paraPair11-paraPair12)>0.02):
        return False
    if (abs(paraPair21-paraPair22) > 0.08):
        print("too big",abs((paraPair21-paraPair22))/paraPair22)
        return False
    return True
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
    print(ratio)
    if(ratio>1.7 and ratio<7):
        qualify = True

    return qualify


def shapeFilter(img):
    test = False
    height,width,channel = img.shape
    maxSize = int(height*width/40)
    minSize = int(height*width/400)
    print(maxSize)
    #forth value to adjust threshold for areas the bigger the more
    #0.2 plate have same color like car
    #dark area
    mser = cv2.MSER_create(10,minSize,maxSize,0.39)
    #mser = cv2.MSER_create(10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    regions = mser.detectRegions(gray, None)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    i = 0
    cts = []
    cv2.polylines(vis, hulls, 1, (0, 255, 255))
    #cv2.imshow('img', vis)
    #cv2.waitKey(0)
    for hull in hulls:
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if(len(approx)==4):
            cts.append(approx)
    cv2.polylines(vis, cts, 1, (0, 255, 0))
    centers = []
    centers.append([0,0])
    Newcts = []
    cropImages = []
    for hull in cts:
        sumX=0
        sumY=0
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
            sumX = sumX + x
            sumY = sumY + y
        averX = int(sumX/4)
        averY = int(sumY/4)
        #cv2.circle(vis, (smallestX, smallestY), 1, (0, 0, 255), -1)
        #center = [int((biggestX + smallestX) / 2), int((biggestY + smallestY) / 2)]
        center = [averX, averY]
        rectangle = True
        tooClose = False
        for acenter in centers:
            if(pointDis(acenter,center)<10):
                tooClose = True
        if not (tooClose):
            if(checkParallel(hull)):
            # distanceSum = 0
            # heightSum = 0
            # widthSum = 0
            # farestDis = 0
            # smallestDis = 1000
            # for point in hull:
            #     x = point[0][0]
            #     y = point[0][1]
            #     if(pointDis(center,point[0])>farestDis):
            #         farestDis=pointDis(center,point[0])
            #     if(pointDis(center,point[0])<smallestDis):
            #         smallestDis = pointDis(center, point[0])
            #     distanceSum =distanceSum + pointDis(center,point[0])
            # averDis = distanceSum/4
            # ratio2 = (farestDis - smallestDis)/smallestDis
            # cv2.circle(vis, (center[0], center[1]), 2, (255, 0, 0), -1)
            # if(ratio2>0.25):
            #     cv2.putText(vis, "FSRf", (center[0]+5, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255)
            #     rectangle = False
            # for point in hull:
            #     ratio = pointDis(center,point[0])/averDis
            #     if(ratio>1.3 or ratio<0.7 ):
            #         cv2.putText(vis, "Af", (center[0], center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255))
            #         rectangle = False
                if(rectangle):
                    if(getWHR(hull)):
                        mask = np.zeros((height,width), np.uint8)
                        cv2.drawContours(mask, [hull], 0, (255,255,255), -1)
                        nWidth, nHeight,cos,degs=getDegree(hull)
                        vertical = False
                        if(cos<0.5):
                            vertical = True
                            print("rectangle is too vertical",cos,center[0], center[1])
                        if(not vertical):
                            cv2.circle(vis, (center[0], center[1]), 1, (0, 0, 255), -1)
                            centers.append(center)
                            Newcts.append(hull)
                            cv2.polylines(vis, Newcts, 1, (0, 0, 255))
                            rotationMatrix = cv2.getRotationMatrix2D((center[0],center[1]), degs, 1.0)
                            imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))
                            imgCropped = cv2.getRectSubPix(imgRotated, (int(nWidth * 1), int(nHeight * 1)),(center[0], center[1]))
                            possiblePlate = PossiblePlate.PossiblePlate()
                            possiblePlate.imgPlate = imgCropped
                            possiblePlate.rrLocationOfPlateInScene = ((center[0], center[1]), (nWidth, nHeight), degs)
                            cropImages.append(possiblePlate)
                            Newcts = []
                            if(test):
                                cv2.imshow('img', vis)
                                cv2.waitKey(0)

                else:
                    print("width / height ratio not correct",center[0], center[1])
            else:
                print("not rectangle",center[0], center[1])
        else:
            print("to close",center[0], center[1])
    if(not test):
        cv2.imshow('img', vis)
        cv2.waitKey(0)

    return cropImages

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
                        os.remove(filePath)
                        #cv2.imshow("deleted image", imgOriginalScene)
                        #cv2.waitKey(500)
                    else:
                        filteResult = shapeFilter(imgOriginalScene)
                        if filteResult is not None:
                            for imc in filteResult:
                                i = i + 1
                                cv2.imwrite(root + '/' + dir + '/posiblPlate' + str(i) + '.png', imc)
def testOne():
    test = True
    filteResult = shapeFilter(cv2.imread('5.jpg'))

#test()
testOne()
