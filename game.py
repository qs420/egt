import numpy as np
import egt.species as sp
import egt.individual as indi
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import os
import glob
import imageio

class game:

    def __init__(self,payoffMatrix,rowPlayerTypes,colPlayerTypes = None):
        self.rowPlayerTypes = rowPlayerTypes
        self.colPlayerTypes = colPlayerTypes
        try:
            if isinstance(payoffMatrix,np.matrixlib.defmatrix.matrix):
                self.payoffMatrix = payoffMatrix.tolist()
            elif isinstance(payoffMatrix,list):
                self.payoffMatrix = payoffMatrix
            else:
                raise TypeError("only accept list or numpy.matrix as input")




            if colPlayerTypes == None:
                self.colPlayerTypes = self.rowPlayerTypes
                if self.rowPlayerTypes.__len__() != self.payoffMatrix.__len__() :
                    raise ValueError("Number of row player not equal to number of col players")
                else:
                    self.rowPlayerDict = {}
                    n = 0
                    for player in self.rowPlayerTypes:
                        self.rowPlayerDict[player] = n
                        n += 1
                    self.colPlayerDict = self.rowPlayerDict
            else:
                if self.rowPlayerTypes.__len__() != self.payoffMatrix.__len__() or self.colPlayerTypes.__len__() != self.payoffMatrix[0].__len__() :
                    raise ValueError(("Number of row player  equal to number of row players' types or Number of col player  equal to number of col players' types "))
                else:
                    self.rowPlayerDict = {}
                    self.colPlayerDict = {}
                    i,j =0
                    for player in self.rowPlayerTypes:
                        self.rowPlayerDict[player] = i
                        i += 1
                    for player in self.colPlayerTypes:
                        self.colPlayerDict[player] = j
                        j += 1
        except:
            pass

    def getPayofffromMatrix(self,player1,player2):
        return self.payoffMatrix[self.rowPlayerDict[player1]][self.colPlayerDict[player2]]

    def onSpecies(self,fitness=1,population=100):
        return speciesGame(self.payoffMatrix,self.rowPlayerTypes,fitness,population)

    def onIndividual(self,gType):
        try:
            if gType == 'geo':
                return geoIndividualGame(self.payoffMatrix,self.rowPlayerTypes)
            elif gType == 'normal':
                return self
            else:
                raise ValueError(" gameType only accecpt 'geo' ro 'normal")
        except:
            pass




class speciesGame(game):

    def __init__(self,payoffMatrix,speciesList,fitness,population ):
        game.__init__(self,payoffMatrix,speciesList,colPlayerTypes = None)
        self.speciesDict = {}
        try:
            if isinstance(fitness,list) and isinstance(population,list) :
                if fitness.__len__() != speciesList.__len__() and population.__len__() != speciesList.__len__():
                    raise ValueError("fitness list length and population list length must equal to number of species")
                else:
                    self.totalPopulation = 0
                    for p in population:
                        self.totalPopulation += p
                    for i in range(fitness.__len__()):
                        self.speciesDict[speciesList[i]] = sp.species(fitness[i],population[i],float(population[i])/self.totalPopulation)
            else:
                self.totalPopulation = population * speciesList.__len__()
                for species in speciesList:
                    self.speciesDict[species] = sp.species(fitness,population,float(1)/speciesList.__len__())
        except:
            pass

    def play(self,type,rounds,func = lambda x,y:x*y):
        if type == "normal":
            for i in range(rounds):
                for species in self.speciesDict.keys():
                    if self.speciesDict[species].fitness > 0:
                        fitnessDelta = 0
                        for opponent in self.speciesDict.keys():
                            fitnessDelta += func(self.speciesDict[opponent].frequencyHistory[self.speciesDict[opponent].age], self.getPayofffromMatrix(species,opponent))
                        fitness = self.speciesDict[species].fitness + fitnessDelta
                        if fitness > 0:
                            self.speciesDict[species].updateFit(fitness)
                        else:
                            self.speciesDict[species].updateFit(0)
                    else:
                        self.speciesDict[species].updateFit(0)
                averageFit = 0
                for species in self.speciesDict.keys():
                    averageFit += self.speciesDict[species].fitness * self.speciesDict[species].frequency
                freqDict = {}
                for species in self.speciesDict.keys():
                    freqDict[species] = self.speciesDict[species].frequency * (float(self.speciesDict[species].fitness)/averageFit)
                totalFreq = 0
                for species in freqDict.keys():
                    totalFreq += freqDict[species]
                for species in freqDict.keys():
                    self.speciesDict[species].updateFreq(float(freqDict[species])/totalFreq)
                for species in self.speciesDict.keys():
                    self.speciesDict[species].updateAge(1)

        return self

    def showFitness(self,latestN = 'all'):
        if latestN == 'all':
            for species in self.speciesDict.keys():
                plt.plot(range(0,self.speciesDict[species].fitnessHistory.__len__()),self.speciesDict[species].fitnessHistory,label = species)
        else:
            for species in self.speciesDict.keys():
                plt.plot(range(self.speciesDict[species].fitnessHistory.__len__()-latestN,self.speciesDict[species].fitnessHistory.__len__()),self.speciesDict[species].fitnessHistory[self.speciesDict[species].fitnessHistory.__len__()-latestN:],label = species)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('fitness')
        plt.title('fitness for all the species')
        plt.show()
    def showFrequency(self,latestN = 'all'):
        if latestN == 'all':
            for species in self.speciesDict.keys():
                plt.plot(range(0,self.speciesDict[species].frequencyHistory.__len__()),self.speciesDict[species].frequencyHistory,label = species)
        else:
            for species in self.speciesDict.keys():
                plt.plot(range(self.speciesDict[species].frequencyHistory.__len__()-latestN,self.speciesDict[species].frequencyHistory.__len__()),self.speciesDict[species].frequencyHistory[self.speciesDict[species].frequencyHistory.__len__()-latestN:],label = species)

        plt.legend()
        plt.xlabel('time')
        plt.ylabel('fitness')
        plt.title('fitness for all the species')
        plt.show()

class doveHarkGame(speciesGame):

    def __init__(self,c,v,fitness = 1,population =100):
        self.payoff = []
        self.payoff.append([float(v-c)/2,v])
        self.payoff.append([0,float(v)/2])
        speciesGame.__init__(self,payoffMatrix=self.payoff,speciesList = ['Hawk','Dove'],fitness=fitness,population=population)

# class individualGame(game):
#
#     def __init__(self,payoffMatrix,rowPlayerTypes,gameType):
#         try:
#             if gameType == 'geo':
#                 return geoIndividualGame(payoffMatrix,rowPlayerTypes)
#             elif gameType == 'normal':
#                 return self
#             else:
#                 raise ValueError(" gameType only accecpt 'geo' ro 'normal")
#         except:
#             pass

class geoIndividualGame(game):

    def __init__(self,payoffMatrix,rowPlayerTypes,colPlayerTypes = None):
        game.__init__(self,payoffMatrix,rowPlayerTypes,colPlayerTypes = None)
        self.topo = []
    def addTopo(self,length,width,speciesDict,sphereofAct):
        self.length = length
        self.width = width
        self.indiMatrix = []
        totalPop = 0
        for key in speciesDict.keys():
            totalPop += speciesDict[key]
        for key in speciesDict.keys():
            speciesDict[key] = speciesDict[key]/totalPop
        self.speciesList = list(speciesDict.keys())
        self.frequencyList = list(speciesDict.values())
        for i in range(self.length):
            self.indiMatrix.append([])
            self.topo.append([])
            for j in range(self.width):
                self.indiMatrix[i].append(np.random.choice(self.speciesList,1,p=self.frequencyList).item())
                self.topo[i].append(indi.indivdual(self.indiMatrix[i][j],(i,j)))
        self.hisIndiMatrix = []
        currentIndiMatrix = []
        for i in range(self.length):
            currentIndiMatrix.append([])
            for j in range(self.width):
                currentIndiMatrix[i].append(self.indiMatrix[i][j])
        self.hisIndiMatrix.append(currentIndiMatrix)
        for i in range(self.length):
            for j in range(self.width):
                self.findNeibors(self.topo[i][j],sphereofAct)
    def specifyTopo(self,individualMatrix):
        self.length = individualMatrix.__len__()
        self.width = individualMatrix[0].__len__()
        self.indiMatrix = individualMatrix
        self.hisIndiMatrix = [self.indiMatrix]
        for i in self.indiMatrix.__len__():
            self.topo.append([])
            for j in self.indiMatrix[i].__len():
                self.topo[i].append(indi.indivdual(self.indiMatrix[i][j]),(i,j))
    def findNeibors(self,me,sphereofAct):
        for i in range(sphereofAct+1):
            for j in range(sphereofAct+1-i):
                if(me.location[0] + i in range(self.length) and me.location[1] + j in range(self.width)):
                    if(self.topo[me.location[0] + i][me.location[1] + j] != 'blank'):
                        me.addNeighbor((me.location[0] + i , me.location[1] + j ))
                if (me.location[0] + i in range(self.length) and me.location[1] - j in range(self.width)):
                    if (self.topo[me.location[0] + i][me.location[1] - j] != 'blank'):
                        me.addNeighbor((me.location[0] + i, me.location[1] - j))
                if (me.location[0] - i in range(self.length) and me.location[1] + j in range(self.width)):
                    if (self.topo[me.location[0] - i][me.location[1] + j] != 'blank'):
                        me.addNeighbor((me.location[0] - i, me.location[1] + j))
                if (me.location[0] - i in range(self.length) and me.location[1] - j in range(self.width)):
                    if (self.topo[me.location[0] - i][me.location[1] - j] != 'blank'):
                        me.addNeighbor((me.location[0] - i, me.location[1] - j))
        me.neighbor = list(set(me.neighbor))
        me.neighbor.remove((me.location))
    def play(self,rounds,selectorType):
        for i in range(rounds):
            matrix = []
            for j in range(self.length):
                matrix.append([])
                for k in range(self.width):
                    if self.topo[j][k] != 'blank':
                        payoff = self.topo[j][k].findpayoff(self.topo,self.colPlayerTypes,self)
                        self.topo[j][k].updatePayoff(payoff)
                        if selectorType == 'BestResponse':
                            bestResponse = self.topo[j][k].findOpimalStra()
                        elif selectorType == 'RandomOnPayoff':
                            bestResponse = self.topo[j][k].actBasedonPayoff()
                        self.indiMatrix[j].append(bestResponse)
                        self.topo[j][k].updateStra(bestResponse)
                        matrix[j].append(bestResponse)
            self.hisIndiMatrix.append(matrix)

            for j in range(self.length):
                for k in range(self.width):
                    if self.topo[j][k] != 'blank':
                        self.topo[j][k].age +=1


    def showtopo(self,iterationID):
        data = []
        for i in range(self.hisIndiMatrix[iterationID].__len__()):
            data.append([])
            for j in range(self.hisIndiMatrix[iterationID][i].__len__()):
                data[i].append(self.rowPlayerDict[self.hisIndiMatrix[iterationID][i][j]])
        sns.set(font_scale=0.8)
        cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=self.rowPlayerTypes.__len__())
        grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)
        ax = sns.heatmap(data, ax=ax, cbar_ax=cbar_ax, cmap=ListedColormap(cmap),
                         linewidths=.5, linecolor='lightgray',
                         cbar_kws={'orientation': 'vertical'})
        cbar_ax.set_yticklabels(list(self.rowPlayerDict.keys()))
        numberOfSpe = self.rowPlayerTypes.__len__()
        print(self.rowPlayerTypes)
        ticks = list(range(1,2*numberOfSpe,2))
        for i in range(ticks.__len__()):
            ticks[i] = float(ticks[i])/(2*numberOfSpe)
        print(ticks)
        cbar_ax.yaxis.set_ticks(ticks)
        ax.set_ylabel('Y')
        ax.set_xlabel('X')

        # Rotate tick labels
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=0)
        locs, labels = plt.yticks()
        plt.setp(labels, rotation=0)
        plt.show()
    def savetopo(self,iterationID,savePath):
        data = []
        for i in range(self.hisIndiMatrix[iterationID].__len__()):
            data.append([])
            for j in range(self.hisIndiMatrix[iterationID][i].__len__()):
                data[i].append(self.rowPlayerDict[self.hisIndiMatrix[iterationID][i][j]])
        sns.set(font_scale=0.8)
        cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=self.rowPlayerTypes.__len__())
        grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.18}
        fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws)
        ax = sns.heatmap(data, ax=ax, cbar_ax=cbar_ax, cmap=ListedColormap(cmap),
                         linewidths=.5, linecolor='lightgray',
                         cbar_kws={'orientation': 'vertical'})
        cbar_ax.set_yticklabels(list(self.rowPlayerDict.keys()))
        numberOfSpe = self.rowPlayerTypes.__len__()

        ticks = list(range(1,2*numberOfSpe,2))
        for i in range(ticks.__len__()):
            ticks[i] = float(ticks[i])/(2*numberOfSpe)

        cbar_ax.yaxis.set_ticks(ticks)
        ax.set_ylabel('Y')
        ax.set_xlabel('X')

        # Rotate tick labels
        locs, labels = plt.xticks()
        plt.setp(labels, rotation=0)
        locs, labels = plt.yticks()
        plt.setp(labels, rotation=0)
        plt.savefig(savePath + '.png')
        plt.close()

    def saveTopoGIF(self,starttime,endtime,gif_name):
        frames =[]
        os.system("mkdir tmp")
        for i in range(starttime, endtime):
            self.savetopo(i,'./tmp/topo_'+str(i))
        for i in range(starttime, endtime):
            frames.append(imageio.imread('./tmp/topo_'+str(i)+'.png'))
        kargs = {'duration': 0.5}
        imageio.mimsave(gif_name+'.gif', frames, 'GIF', **kargs)
        os.system("rm -r tmp/")


    def showSpeciesChanges(self):
        countDict = {}
        for species in self.rowPlayerTypes:
            countDict[species] = []
        for i in range(self.hisIndiMatrix.__len__()):
            for species in countDict.keys():
                countDict[species].append(0)
            for j in range(self.hisIndiMatrix[i].__len__()):
                for k in range(self.hisIndiMatrix[i][j].__len__()):
                    countDict[self.hisIndiMatrix[i][j][k]][i] +=1
        for species in countDict.keys():
            plt.plot(range(0, self.hisIndiMatrix.__len__()),
                     countDict[species], label=species)
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('number of individuals of speciex')
        plt.title('number of individuals of species')
        plt.show()







