import numpy as np
import utils
import random


class Agent:
    
    def __init__(self, actions, Ne, C, gamma):
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma

        # Create the Q and N Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()

        self.reset() #calling reset

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        self.points = 0
        self.s = None
        self.a = None
    
    def getMaxQ(self,awx, awy, fdx, fdy, abt, abb, abl, abr):
        maxQ = -99999
        for i in self.actions:
            if maxQ < self.Q[awx][awy][fdx][fdy][abt][abb][abl][abr][i]:
                maxQ = self.Q[awx][awy][fdx][fdy][abt][abb][abl][abr][i]
        return maxQ
    
    def act(self, state, points, dead):
        '''
        :param state: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] from environment.
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: the index of action. 0,1,2,3 indicates up,down,left,right separately

        TODO: write your function here.
        Return the index of action the snake needs to take, according to the state and points known from environment.
        Tips: you need to discretize the state to the state space defined on the webpage first.
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the 480x480 board)

        '''
        (awx, awy, fdx, fdy, abt, abb, abl, abr) = self.MDP(state)
        if self._train and self.a!=None and self.s!=None:

            reward = -0.1
            if dead:
                reward = -1
            elif points > self.points:
                reward = 1

            #get maxQ(s',a')
            maxQ = self.getMaxQ(awx, awy, fdx, fdy, abt, abb, abl, abr)

            (curr_awx, curr_awy, curr_fdx, curr_fdy, curr_abt, curr_abb, curr_abl, curr_abr) = self.s
            # learning rate = c/(c+N(s,a))
            learn_rate = self.C / (self.C + self.N[curr_awx][curr_awy][curr_fdx][curr_fdy][curr_abt][curr_abb][curr_abl][curr_abr][self.a])
            #q(s,a) = q(s,a)+learn_rate(r(s)+gamma*maxQ(s',a')-q(s,a))
            self.Q[curr_awx][curr_awy][curr_fdx][curr_fdy][curr_abt][curr_abb][curr_abl][curr_abr][self.a] += learn_rate * (reward + self.gamma * maxQ - self.Q[curr_awx][curr_awy][curr_fdx][curr_fdy][curr_abt][curr_abb][curr_abl][curr_abr][self.a])
        
        if not dead :
            self.s = (awx, awy, fdx, fdy, abt, abb, abl, abr)
            self.points = points
        else:
            self.reset()
            return 0

        best = -99999
        action = 0
        for i in self.actions:
            nTable = self.N[awx][awy][fdx][fdy][abt][abb][abl][abr][i]
            if self.Ne <= nTable:
                qTable = self.Q[awx][awy][fdx][fdy][abt][abb][abl][abr][i]
                if qTable >= best:
                    best = qTable
                    action = i
            else:
                if best <= 1:
                    best = 1
                    action = i

        self.N[awx][awy][fdx][fdy][abt][abb][abl][abr][action] += 1
        self.a = action

        return action

    def MDP(self, state):
        (hx, hy, snake, fx, fy) = state
        grid = utils.GRID_SIZE

        awx = 0 #adjoining_wall_x
        if hx == grid :
            awx = 1
        elif hx == 12 * grid:
            awx = 2

        awy = 0 #adjoining_wall_y
        if hy == grid:
            awy = 1
        elif hy == 12 * grid:
            awy = 2

        abt = 0 #adjoining_body_top
        abb = 0 #adjoining_body_bottom
        abl = 0 #adjoining_body_left
        abr = 0 #adjoining_body_right
        for i in snake:
            if hx == i[0] and i[1] == hy - grid:
                abt = 1
            if hx == i[0] and i[1] == hy + grid:
                abb = 1
            if hy == i[1] and i[0] == hx - grid:
                abl = 1
            if hy == i[1] and i[0] == hx + grid:
                abr = 1

        fdx = 2
        if hx == fx:
            fdx = 0
        elif hx > fx:
            fdx = 1

        fdy = 2
        if hy == fy :
            fdy = 0
        elif hy > fy :
            fdy = 1

        return (awx, awy, fdx, fdy, abt, abb, abl, abr)
