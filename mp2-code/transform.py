
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    alpha, beta = arm.getArmLimit()
    alphaMin = alpha[0]
    alphaMax = alpha[1]
    betaMin = beta[0]
    betaMax = beta[1]
    row = int(abs(alphaMax - alphaMin)//int(granularity)) + 1 #get row by using alpha
    col = int(abs(betaMax - betaMin)//int(granularity)) + 1 #get col by using beta

    mazes = [] #construct a maze
    for i in range(row):
        cols = []
        for j in range(col):
            cols.append(WALL_CHAR)
        mazes.append(cols)
    iniAlpha = arm.getArmAngle()[0]
    iniBeta = arm.getArmAngle()[1]
    
    for i in range(row):
        for j in range(col):
            Alphas = int(i * granularity) + alphaMin
            Betas = int(j * granularity) + betaMin
            arm.setArmAngle((Alphas, Betas)) #set all arm angle for (alpha, beta)
            if not isArmWithinWindow(arm.getArmPos(), window): #check if it's within window
                mazes[i][j] = WALL_CHAR
                continue
            elif doesArmTouchObjects(arm.getArmPos(), goals, isGoal=True) and (not doesArmTipTouchGoals(arm.getEnd(), goals)):
                mazes[i][j] = WALL_CHAR
                continue
            elif doesArmTipTouchGoals(arm.getEnd(), goals):
                if not doesArmTouchObjects(arm.getArmPos(), obstacles, isGoal=False): #check goals
                   mazes[i][j] = OBJECTIVE_CHAR
                   continue
                elif doesArmTouchObjects(arm.getArmPos(), goals, isGoal=True):
                    mazes[i][j] = OBJECTIVE_CHAR
                    continue
                else:
                    mazes[i][j] = WALL_CHAR
                    continue
            elif doesArmTouchObjects(arm.getArmPos(), obstacles,isGoal=False): #check obstacles
                mazes[i][j] = WALL_CHAR
                continue
            else:
                mazes[i][j] = SPACE_CHAR

                
    iniMin = (iniAlpha - alphaMin)//granularity
    iniMax = (iniBeta - betaMin)//granularity 
    mazes[iniMin][iniMax] = START_CHAR
    return Maze(mazes, [alphaMin, betaMin], granularity)
    