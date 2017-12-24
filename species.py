class species:

    def __init__(self,fitness,population,frequency):

       self.fitness = fitness
       self.population = population
       self.frequency = frequency
       self.fitnessHistory = [fitness]
       self.populationHistory = [population]
       self.frequencyHistory = [frequency]
       self.age = 0

    def updateFit(self,newFit):
        self.fitness = newFit
        self.fitnessHistory.append(newFit)

    def updateFreq(self,newFreq):
        self.frequency = newFreq
        self.frequencyHistory.append(newFreq)

    def updateAge(self,n):
        self.age += n


