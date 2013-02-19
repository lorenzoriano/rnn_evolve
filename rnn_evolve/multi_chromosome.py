import pyevolve
import pyevolve.GenomeBase
import pyevolve.G1DList
import pyevolve.G1DBinaryString

class RealBin(pyevolve.GenomeBase.GenomeBase):
    def __init__(self,  reallength,  binarylength):
        pyevolve.GenomeBase.GenomeBase.__init__(self)
        
        self.reallength = reallength
        self.binarylength = binarylength
        
        self.real_genome = pyevolve.G1DList.G1DList(reallength)
        self.binary_genome = pyevolve.G1DBinaryString.G1DBinaryString(binarylength)
        
        self.crossover.set(self.cross)
        self.mutator.set(self.mutate)
        self.initializator.set(self.initialize)
        
    def initialize(self,  **args):
        self.real_genome.initialize()
        self.binary_genome.initialize()
        
    def mutate(self,  **args):
        n = self.real_genome.mutate(**args)
        m = self.binary_genome.mutate(**args)
        return n + m
   
    def cross(self, genome,  **args):
        
        gMom = args["mom"]
        gDad = args["dad"]
        
        sister = gMom.clone()
        brother = gDad.clone()
        
        realMom = gMom.real_genome
        realDad = gDad.real_genome
        
        args["mom"] = realMom
        args["dad"] = realDad
        for it in realMom.crossover.applyFunctions(**args):
            (realSister, realBrother) = it
        
        binMom = gMom.binary_genome
        binDad = gDad.binary_genome
        
        args["mom"] = binMom
        args["dad"] = binDad
        for it in binMom.crossover.applyFunctions(**args):
            (binSister, binBrother) = it
                  
        sister.real_genome = realSister
        sister.binary_genome = binSister
        
        brother.real_genome = realBrother
        brother.binary_genome = binBrother
        
        return sister,  brother

    def __repr__(self):
        return self.real_genome.__repr__() + "\n" + self.binary_genome.__repr__()

    def copy(self,  g):
        pyevolve.GenomeBase.GenomeBase.copy(self, g)
        
        g.reallength = self.reallength
        g.binarylength = self.binarylength
        self.real_genome.copy(g.real_genome)
        self.binary_genome.copy(g.binary_genome)
    
    def clone(self):
        newcopy = RealBin(self.reallength,  self.binarylength)
        self.copy(newcopy)
        return newcopy
