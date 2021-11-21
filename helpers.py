from PIL import Image
import colorsys
import math
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import notebook
import random
import scipy.stats as st

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
        if simStrat == "antiThetic":
            self.minY = 0
            self.maxY = self.y + self.yRange
        
        self.num_points = num_points
        self.precision = precision
    
    
    def run_sim(self):
        """Runs a Monte Carlo simulation using the defined strategy on the Mendelbrot set.

        Raises:
            ValueError: The strategy should be one of the predefined strategies.

        Returns:
            float: estimated area of the mendelbrot set.
        """
        if self.draw: 
            self.drawPicture()
                        
        # run monte carlo simulation
        inCount = 0
        if self.simStrat == "random":
            coordinates = self.getCoordinates()
        elif self.simStrat == "latin":
            coordinates = self.getLatinCube()
        elif self.simStrat == "orthogonal":
            coordinates = self.getOrthogonal()
        elif self.simStrat == "orthogonalFast":
            coordinates = self.getOrthogonalFast()
        elif self.simStrat =="antiThetic":
            coordinates = self.getAntithetic()
        else:
            raise ValueError(404)

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
        
        if self.simStrat == "antiThetic":
            areaMendel *= 2


        return areaMendel
            
                    
    def getCoordinates(self):
        """creates random coordinates depending on the requested number of coordinates

        Returns:
            (list[floats], list[floats]): row, column
        """
        # this code is 7 times slower for some reason, interesting to find out why
        #return(list(np.random.random(self.num_points)*self.width), list(np.random.random(self.num_points)*self.height))
        
        x = []
        y = []
        xappend = x.append
        yappend = y.append
        for i in range(self.num_points):
            xappend(np.random.random())
            yappend(np.random.random())
        
        return ([i*self.width for i in x], [i*self.height for i in y])
    

    def getLatinCube(self):
        """creates random coordinates ordered in a latin hypercube

        Returns:
            (list[floats], list[floats]): row, column
        """
        assert self.height == self.width, "the aspect ratio should be 1"
        x = []
        y = []
        possibilities = np.arange(self.width)
        np.random.shuffle(possibilities)
        for row in range(self.width):
            x.append(row)
            y.append(possibilities[row])

        return (x, y)
    

    #https://codereview.stackexchange.com/questions/207610/orthogonal-sampling
    def getOrthogonal(self):
        """creates random coordinates ordered orthogonally in a latin hypercube

        Returns:
            (list[floats], list[floats]): row, column
        """
        assert self.height == self.width, "Make sure that aspect ratio is 1"
        assert np.sqrt(self.width) % 1 == 0,"Please insert a width of which the square root is an integer"
        n = int(np.sqrt(self.width))
        # Making a datastructure of a dict with coordinate tuples of a bigger grid with subcoordinate of sub-grid points
        blocks = {(i,j):[(a,b) for a in range(n) for b in range(n)] for i in range(n) for j in range(n)}
        x = []
        y = []
        appendx = x.append
        appendy = y.append
        for block in blocks:
            # select a random minor cell from every major cell
            point = random.choice(blocks[block])
            lst_row = [(k1, b) for (k1, b), v in blocks.items() if k1 == block[0]]
            lst_col = [(a, k1) for (a, k1), v in blocks.items() if k1 == block[1]]

            for col in lst_col:
                blocks[col] = [a for a in blocks[col] if a[1] != point[1]]

            for row in lst_row:
                blocks[row] = [a for a in blocks[row] if a[0] != point[0]]
           #Adjust the points to fit the grid they fall in  
            appendx(point[0] + n * block[0])
            appendy(point[1] + n * block[1])

        return (x, y)


    def permute(self, l, N):
        """changes a list into a random order of itself

        Args:
            l (list): the list that will be permuted
            N (INT): The number of times the list will be shuffled

        Returns:
            list: a permutation of the input list
        """
        for i in range(N-1, 0, -1):
            before = l[i]
            new = np.random.randint(0,i+1)
            l[i] = l[new]
            l[new] = before

        return l


    def getOrthogonalFast(self):
        """Fastly creates an orthogonal latin hypercube.

        Returns:
            [(floats, floats)]: A list containing a tuple with the x coordinates on the first entry and the y coordinates on the second entry.
        """
        assert self.height == self.width, "Make sure that aspect ratio is 1"
        assert np.sqrt(self.width) % 1 == 0,"Please insert a width of which the square root is an integer"
        
        
        major = int(np.sqrt(self.width))
        # intialise the x and y lists
        # the xlist keeps track what minor column has the sample in major cell with major cell i, j
        # the ylist does the same for the minor row
        xlist = np.zeros((major,major), np.int32)
        ylist = np.zeros((major,major), np.int32)

        # start with the most simple solution
        m = 0
        for i in range(major):
            for j in range(major):
                xlist[i][j] = m
                ylist[j][i] = m
                m += 1

        # get a random solution
        for i in range(major):
            xlist[i] = self.permute(xlist[i], major)
            ylist[i] = self.permute(ylist[i], major)
        
        x = []
        y = []
        xappend = x.append
        yappend = y.append
        for i in range(major):
            for j in range(major):
                xappend(xlist[i][j])
                yappend(ylist[i][j])
        return (x,y)


    def getAntithetic(self):
        """creates Antithetic coordinates, so the variance of sampling will be lowered

        Returns:
            [(floats, floats)]: A list containing a tuple with the x coordinates on the first entry and the y coordinates on the second entry.
        """
        x = np.random.random(round(self.num_points/2))
        y = np.random.random(round(self.num_points/2))
        x_ = 1-x
        y_ = 1-y
        xs = np.concatenate((x, x_))
        ys = np.concatenate((y, y_))

        return([i*self.width for i in xs], [i*self.height for i in ys])


    #https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f
    def drawPicture(self):
        """creates an image of the Monte Carlo simulation on the Mendelbrot set.
        """
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


    def powerColor(self, distance, exp, const, scale):
        """Chooses colors depending on the depth until convergences in the Mendelbrot set.

        Args:
            distance (int): The amount of iterations until the coordinates convergenced.
            exp (int): Settings for choosing the color palet.
            const (int): Settings for choosing the color palet.
            scale (int): Settings for choosing the color palet.

        Returns:
            tuple: A tuple of colors in RGB format
        """
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



def getResults(inpSort):
    """Sorts the results of a shuffled input.

    Args:
        inpSort (np.array(3,n)): A result that has a independent variable in the first index and a dependent variable in the second.

    Returns:
        np.array(2,n): A numpy array sorted on the independent variable.
    """
    results = []
    for entry in range(len(inpSort[0])):
        results.append((inpSort[0][entry], inpSort[1][entry]))
    
    def getKey(item):
        return item[0]

    resultsSorted = sorted(results, key=getKey)
    resultsArray = np.zeros((2, len(inpSort[0])), dtype=float)
    for i in range(len(inpSort[0])):
        resultsArray[0][i] = float(resultsSorted[i][0])
        resultsArray[1][i] = float(resultsSorted[i][1])
    
    return resultsArray

    
def createBars(bars, result, title, filename):
    """Creates a barplot with confidence intervals of results.

    Args:
        bars (int): The amount of bars requested
        result (np.array(2,n)): A numpy array containing a independent variable in the first index and a dependent variable in the second index.
        title (string): A title for the plot
        filename (string): A location for the plot to be saved

    Returns:
        list, list: Returns a list of standard deviations and a list of means for every bar in order.
    """
    resultsArray = getResults(result)
    means = []
    stds = []
    currIndex = 0
    for bar in range(bars):
        nextIndex = int(currIndex + len(resultsArray[1])/bars)
        means.append(np.mean(resultsArray[1][currIndex:nextIndex]))
        stds.append(np.std(resultsArray[1][currIndex:nextIndex]))
        currIndex = nextIndex


    fig, ax = plt.subplots()
    ax.bar(np.arange(bars), means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_title(title + f"\nfinal standard deviation was {stds[-1]}")
    ax.set_ylabel("estimated area")
    plt.savefig(filename)
    return stds, means


