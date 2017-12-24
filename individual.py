import numpy as np
class indivdual:

    def __init__(self,strategy,location,fitness = 1):
        self.fitness = fitness
        self.currentStategy = strategy
        self.historicalStratege = [strategy]
        self.historicalPayoff = []
        self.location = location
        self.age = 0
        self.neighbor = []


    def updateStra(self,stra):
        self.currentStategy =stra
        self.historicalStratege.append(stra)

    def addNeighbor(self,location):
        self.neighbor.append(location)

    def updatePayoff(self,po):
        self.historicalPayoff.append(po)

    def findpayoff(self,topo,strategys,games):
        self.payoff = {};
        for strategy in strategys:
            self.payoff[strategy] = 0
            for nb in self.neighbor:
                self.payoff[strategy] += games.getPayofffromMatrix(strategy,topo[nb[0]][nb[1]].historicalStratege[topo[nb[0]][nb[1]].age])
            self.payoff[strategy] /= self.neighbor.__len__()
        return self.payoff

    def findOpimalStra(self):
        best = ''
        pay = -1000000
        for key in self.payoff.keys():
            if self.payoff[key] > pay:
                best = key
                pay = self.payoff[key]
        return best
    def actBasedonPayoff(self):
        freqDict = {}
        totalPayoff = 0
        minPayoff = 100000
        for key in self.payoff.keys():
            if minPayoff > self.payoff[key]:
                minPayoff = self.payoff[key]
        if minPayoff<0:
            for key in self.payoff.keys():
                self.payoff[key] -= (minPayoff)
        for key in self.payoff.keys():
            totalPayoff += self.payoff[key]
        if totalPayoff == 0:
            for key in self.payoff.keys():
                freqDict[key] = float(1)/self.payoff.keys().__len__()
        else:
            for key in self.payoff.keys():
                freqDict[key] = self.payoff[key]/totalPayoff
        return np.random.choice(list(freqDict.keys()),1,p=list(freqDict.values())).item()








