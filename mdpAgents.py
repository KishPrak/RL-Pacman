# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import math


class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print("Starting up MDPAgent!")
        name = "Pacman"

    
    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print("Running registerInitialState for MDPAgent!")
        print("I'm at:")
        print(api.whereAmI(state))
        
        #Value iteration hyperparameters
        self.discountFactorGhostClose = 0.8
        self.discountFactorGhostFar = 0.95
        self.epsilon = 5e-3
        
        self.foodReward = 10
        self.capsuleReward = 20
        self.ghostReward = -500
        
        self.visited = set()
        self.rewards = {}
        self.utilities = {}
        self.neighbours = {}
        
        self.initialFoodCount = len(api.food(state))
        
        self.makeNeighbourMap(state) 
        self.makeRewardMap(state)
        self.makeUtilityMap()
        
        self.intendedDirectionProb, self.stochasticMovementProb = 0.8, 0.1
        
        
        
    # This is what gets run in between multiple games
    def final(self, state):
        print("Looks like the game just ended!")


    def get_size(self, state):
        return api.corners(state)[-1]
    
    def getCoordFromDirection(self, x, y, direction):
            if direction == "North":
                return (x, y+1)
            elif direction == "East":
                return (x+1, y)
            elif direction == "South":
                return (x, y-1)
            elif direction == "West":
                return (x-1, y)
    
    def makeNeighbourMap(self, state):
        walls = set(api.walls(state))
        maxX, maxY = self.get_size(state)
        directions = ["North", "East", "South", "West"]

        for x in range(maxX + 1):
            for y in range(maxY + 1):
                pos = (x, y)
                if pos in walls:
                    continue
                self.neighbours[pos] = []
                for direction in directions:
                    neighbour = self.getCoordFromDirection(x, y, direction)
                    nx, ny = neighbour
                    if (0 <= nx <= maxX and 0 <= ny <= maxY and neighbour not in walls):
                        self.neighbours[pos].append(direction)
        #print(self.neighbours)
                        
    def makeRewardMap(self, state):
        self.visited.add(api.whereAmI(state))
        self.rewards = {}

        for pos in self.neighbours:
            if pos in self.visited:
                self.rewards[pos] = 0
            else:
                self.rewards[pos] = 0
        
        for pos in api.food(state):
            if pos not in self.visited:
                self.rewards[pos] = self.foodReward

        for pos in api.capsules(state):
            if pos not in self.visited:
                self.rewards[pos] = self.capsuleReward


        for pos, edibleTime in api.ghostStatesWithTimes(state):
            if pos not in self.neighbours:
                continue
            distance = self.distFromObject(state, pos)
            if edibleTime == 0 or distance > edibleTime:
                self.rewards[pos] = self.ghostReward
            else:
                self.rewards[pos] = self.ghostReward*0.7
            
    
    def makeUtilityMap(self):
        self.utilities = {pos: 0 for pos in self.neighbours.keys()}


    #bfs to the object from the current pacman pos
    def distFromObject(self, state, objectPos):
        curPos = api.whereAmI(state)

        if curPos == objectPos:
            return 0

        visited = {curPos}
        cur_neighbours = [curPos]
        distanceAway = 0

        while cur_neighbours:
            next_neighbours = []

            for pos in cur_neighbours:
                if pos == objectPos:
                    return distanceAway

                for direction in self.neighbours[pos]:
                    neighbour = self.getCoordFromDirection(
                        pos[0], pos[1], direction
                    )

                    if neighbour in visited:
                        continue

                    visited.add(neighbour)
                    next_neighbours.append(neighbour)

            cur_neighbours = next_neighbours
            distanceAway += 1

        return -1
        
    
    def actionUtility(self, pos, direction, values, discountFactor):
        if direction in ("North", "South"):
            perpendicular = ("East", "West")
        else:
            perpendicular = ("North", "South")

        utility = 0.0
        
        directionsToConsider = [direction, perpendicular[0],perpendicular[1]]
        transitionProbabilities = [self.intendedDirectionProb, self.stochasticMovementProb, self.stochasticMovementProb]

        for move, probability in zip(directionsToConsider, transitionProbabilities):
            if move in self.neighbours[pos]:
                next_pos = self.getCoordFromDirection(pos[0], pos[1], move)
            else:
                next_pos = pos

            utility += probability * (self.rewards[next_pos] + discountFactor * values[next_pos])

        return utility
    
        
    def value_iteration(self, discount_factor, epsilon):
        
        while True:
            old_values = self.utilities.copy()
            delta = 0.0
            for pos in self.neighbours:
                actions = self.neighbours[pos]
                if not actions:
                    continue
                self.utilities[pos] = max(self.actionUtility(pos, direction, old_values, discount_factor) for direction in actions)

                delta = max(
                    delta,
                    abs(self.utilities[pos] - old_values[pos])
                )

            if delta < epsilon:
                break
            
    def getDiscountFactor(self, state):
        for pos, edibleTime in api.ghostStatesWithTimes(state):
            distFromGhost = self.distFromObject(state, pos)
            if distFromGhost > edibleTime and distFromGhost <= 2:
                return self.discountFactorGhostClose
        return self.discountFactorGhostFar
                        
                        
    def getAction(self, state):
        self.makeRewardMap(state)

        discountFactor = self.getDiscountFactor(state)

        self.value_iteration(
            discountFactor,
            self.epsilon
        )

        pos = api.whereAmI(state)
        neighbours = self.neighbours[pos]

        bestAction = Directions.STOP
        prevBiggestUtility = float("-inf")

        for direction in neighbours:
            utility = self.actionUtility(
                pos,
                direction,
                self.utilities,
                discountFactor
            )

            if utility > prevBiggestUtility:
                prevBiggestUtility = utility
                bestAction = direction

        return api.makeMove(bestAction, api.legalActions(state))