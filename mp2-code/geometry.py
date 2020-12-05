# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    x = start[0] + int(length * math.cos(math.radians(angle)))
    y = start[1] - int(length * math.sin(math.radians(angle)))

    return (x, y)

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    for arm in armPosDist:
        x1 = arm[0][0]
        y1 = arm[0][1]
        x2 = arm[1][0]
        y2 = arm[1][1]
        for obj in objects:
            x = obj[0]
            y = obj[1]
            r = obj[2]
            dist = getDis(x1,y1,x2,y2,x,y)
            if(isGoal):
                if(dist<=r):
                    return True
            else:
                if dist <= (r+5):
                    return True


    return False
def getDis(x1,y1,x2,y2,x,y):
    x_a = x-x1
    y_a = y-y1

    x_b = x2-x1
    y_b = y2-y1

    crossDot = y_a*y_b+x_a*x_b
    length = x_b**2 +y_b**2
    dis = -1

    if(length!=0):
        dis = crossDot/length
    
    if(dis<0): #check closet to start
        x0 = x1
        y0 = y1
    elif(dis>1): #closet to end
        x0 = x2
        y0 = y2
    else:
        x0 = x1+dis*x_b
        y0 = y1+dis*y_b

    return math.sqrt((x-x0)**2+(y-y0)**2)
   
def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tip touches any goal. False if not.
    """
    tipX = armEnd[0]
    tipY = armEnd[1]
    for i in goals:
        x = i[0]
        y = i[1]
        r = i[2]
        distance = math.sqrt((tipX-x)**2 + (tipY-y)**2) #check if arm tips within the circle
        if distance <= r:
            return True
     
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    for arm in armPos:
        startX = arm[0][0]
        startY = arm[0][1]
        endX = arm[1][0]
        endY = arm[1][1]
        width = window[0]
        height = window[1]
        if (startX >width or startX < 0): #check if all parts are in the window
            return False
        elif(startY > height or startY < 0):
            return False
        elif (endX > width or endX < 0):
            return False
        elif(endY> height or endY < 0):
            return False
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            #print(testArmPosDist)
            #print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
    
    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
