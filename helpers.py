from PIL import Image
import colorsys
import math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import notebook

class mendelSim():
    def __init__(self, width = 1000, x =-0.65, y = 0, xRange = 3.4, aspectRatio = 4/3, 
                 simStrat = "random", draw = True, niceColors = True,
                 num_points = 30000, precision = 200):
        self.draw = draw
        self.niceColors = niceColors
        self.simStrat = simStrat
        
        
        # frame parameters
        self.width = width #pixels
        self.x = x
        self.y = y
        self.xRange = xRange
        self.aspectRatio = aspectRatio 

        self.height = round(self.width / self.aspectRatio)
        self.yRange = self.xRange / self.aspectRatio
        self.minX = self.x - self.xRange / 2
        self.maxX = self.x + self.xRange / 2
        self.minY = self.y - self.yRange / 2
        self.maxY = self.y + self.yRange / 2
        
        self.num_points = num_points
        self.precision = precision
    
    
    def run_sim(self):
        if self.draw: 
            self.drawPicture()
                        
        # run monte carlo simulation
        inCount = 0
        if self.simStrat == "random":
            coordinates = self.getCoordinates()
        elif self.simStrat == "latin":
            coordinates = self.getLatinCube()
        elif self.simStrat == "orthogonal":
            #TODO
            pass
        else:
            raise ValueError
        for c in range(len(coordinates[0])):
            x = self.minX + coordinates[0][c] * self.xRange / self.width
            y = self.maxY - coordinates[1][c] * self.yRange / self.height
            oldX = x
            oldY = y
            for i in range(self.precision + 1):
                a = x*x - y*y #real component of z^2
                b = 2 * x * y #imaginary component of z^2
                x = a + oldX #real component of new z
                y = b + oldY #imaginary component of new z
                if x*x + y*y > 4:
                    break
            if i == self.precision:
                inCount += 1
            if self.draw: 
                rgb = self.redBlue(i)
                self.pixels[coordinates[0][c],coordinates[1][c]] = rgb


        if self.draw: 
            self.img.save('output.png')

        areaPicture = self.xRange * self.yRange
        percentageMendel = inCount/len(coordinates[0])
        areaMendel = areaPicture * percentageMendel
        
        
        return areaMendel
    
    
    def drawPicture(self):
        # draw mandelbrot image
        self.img = Image.new('RGB', (self.width, self.height), color = 'white')
        self.pixels = self.img.load()

        for row in range(self.height):
            for col in range(self.width):
                x = self.minX + col * self.xRange / self.width
                y = self.maxY - row * self.yRange / self.height
                oldX = x
                oldY = y
                for i in range(self.precision + 1):
                    a = x*x - y*y #real component of z^2
                    b = 2 * x * y #imaginary component of z^2
                    x = a + oldX #real component of new z
                    y = b + oldY #imaginary component of new z
                    if x*x + y*y > 4:
                        break
                if i < self.precision:
                    rgb = (0,0,0)
                    if self.niceColors:
                        distance = (i + 1) / (self.precision + 1)
                        rgb = self.powerColor(distance, 0.2, 0.27, 1.0)
                    self.pixels[col,row] = rgb
        
                    
    def getCoordinates(self):
        # returns random coordinates on a grid 
        x = []
        y = []
        for i in range(self.num_points):
            x.append(np.random.random())
            y.append(np.random.random())
        return ([i*self.width for i in x], [i*self.height for i in y])
    
    
    def getLatinCube(self):
        # returns a cube with one coordinate in every row and column
        # check if the input is a cube
        assert self.height == self.width
        x = []
        y = []
        possibilities = np.arange(self.width)
        np.random.shuffle(possibilities)
        for row in range(self.width):
            x.append(row)
            y.append(possibilities[row])

        return (x, y)
    
    
    def powerColor(self, distance, exp, const, scale):
        # color a pixel dependent on how long the belonging values took to diverge
        color = distance**exp
        rgb = colorsys.hsv_to_rgb(const + scale * color,1 - 0.6 * color,0.9)
        return tuple(round(i * 255) for i in rgb)

    
    def redBlue(self, distance):
        # color a pixel depending on it's divergence. Red for diverging and green for not diverging.
        if distance == self.precision:
            rgb = (0,255,0)
        else:
            rgb = (255,0,0)
        return rgb