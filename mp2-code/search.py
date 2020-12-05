# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    start = maze.getStart() #get start of Maze
    startP = start #set a startPoint in queue
    dot = maze.getObjectives()[0] # get the single dot position
    visited = {} # a visited dictionary
    queue = [] # queue list for bfs
    queue.append(startP)
    visited[startP] = True
    path= {} # set the path to record each vertex came from
    path[startP] = None # set the start Point as the key, the value is none
    while queue:
        startP = queue.pop(0)
        if(startP == dot): # early exit if start point = dot
            break
        neighbor = maze.getNeighbors(startP[0],startP[1])
        for i in neighbor:
            if maze.isValidMove(i[0],i[1]) and (not i in visited): # check the neighbor if is visited and is valid move or not.
                visited[i] = True
                path[i] = startP  #set the neighbor as the key, the current vertex as the value in order to avoid multiple neighbor that cause the updating dict value
                queue.append(i)

   
    pathToFindDot = getPathToDot(start,dot,path)
    return pathToFindDot # return the path


def getPathToDot(start, dot, path):
    pathToFindDot = []  # path to find the dot
    pathToFindDot.append(dot) # append the dot into the list
    nexts = dot
    while(nexts != start): # find the path from dot to the start point
        if(nexts not in path):
            return None
        pathToFindDot.append(path[nexts])
        nexts = path[nexts]

    ##pathToFindDot.append(start)
    return pathToFindDot[::-1] # return the reversed path
