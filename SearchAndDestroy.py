import numpy as np
from numpy import zeros
import random
from matplotlib import pyplot as plt
from collections import OrderedDict
import operator


class Node:#Can initizalize a node with attributes of a square in the grid
    def __init__(self, row, col, prob):
        self.row = row
        self.col = col
        self.prob = prob

def improvedAgent():
    d = 10
    grid, target = generateGrid(d)
    knowledgeBase = np.empty(shape=(len(grid),len(grid)),dtype='object')#knowledgeBase will represent chance for each cell to be target
    knowledgeBase.fill(1.0/(len(grid)*len(grid)))
    fringe = generateFringe2(knowledgeBase, grid)#fringe will contain best cells to search in order
    t = 0
    targetFound = False
    count = 0
    visited = np.empty(shape=(d,d), dtype='object')
    visited.fill(0)
    totalDistance = 0
    current = Node(random.randint(0,len(grid)-1), random.randint(0,len(grid)-1), 0)
    ties = np.empty(0, dtype='object')
    prev = None
    while(not(targetFound)):
        count+=1
        if(count%100==0):
            print("Working: "+str(count))
        if(current.row==target.row and current.col==target.col):
            if(random.uniform(0,1)>grid[target.row][target.col]):#success found target
                targetFound = True
                break
        knowledgeBase = numpyUpdate(knowledgeBase, grid[current.row][current.col], current)
        visited[current.row][current.col] = visited[current.row][current.col]+1
        prev = current
        current = improvedFringe1(knowledgeBase, grid, visited, current)

        totalDistance = totalDistance + getDistance(prev, Node(current.row, current.col, 0))
        #current = fringe[0]
        t+=1
    return t+totalDistance

def itesta():
    d = 50
    grid, target = generateGrid(d)
    knowledgeBase = np.empty(shape=(len(grid),len(grid)),dtype='object')#knowledgeBase will represent chance for each cell to be target
    knowledgeBase.fill(1.0/(len(grid)*len(grid)))
    #fringe = generateFringe2(knowledgeBase, grid)#fringe will contain best cells to search in order
    t = 0
    targetFound = False
    count = 0
    visited = np.empty(shape=(d,d), dtype='object')
    visited.fill(0)
    totalDistance = 0
    current = Node(random.randint(0,len(grid)-1), random.randint(0,len(grid)-1), 0)
    while(not(targetFound)):
        count+=1
        if(count%100==0):
            print("Working: "+str(count))
        if(current.row==target.row and current.col==target.col):
            if(random.uniform(0,1)>grid[target.row][target.col]):#success found target
                targetFound = True
                break
        knowledgeBase = numpyUpdate(knowledgeBase, grid[current.row][current.col], current)
        visited[current.row][current.col] = visited[current.row][current.col]+1
        t+=1
    return t+totalDistance

def agent1():
    d = 10
    grid, target = generateGrid(d)
    knowledgeBase = np.empty(shape=(len(grid),len(grid)),dtype='object')#knowledgeBase will represent chance for each cell to be target
    knowledgeBase.fill(1.0/(len(grid)*len(grid)))
    fringe = generateFringe1(knowledgeBase, grid)#fringe will contain best cells to search in order
    t = 0
    targetFound = False
    count = 0
    visited = np.empty(shape=(d,d), dtype='object')
    visited.fill(0)
    totalDistance = 0
    current = Node(random.randint(0,len(grid)-1), random.randint(0,len(grid)-1), 0)
    ties = np.empty(0, dtype='object')
    while(not(targetFound)):
        count+=1
        if(current.row==target.row and current.col==target.col):
            if(random.uniform(0,1)>grid[target.row][target.col]):#success found target
                targetFound = True
                break
        knowledgeBase = numpyUpdate(knowledgeBase, grid[current.row][current.col], current)
        visited[current.row][current.col] = visited[current.row][current.col]+1
        fringe = updateFringe1(knowledgeBase, grid, visited)
        i = 0
        if(fringe[i].prob==fringe[i+1].prob):
            while(fringe[i].prob==fringe[i+1].prob):
                if(i==0):
                    ties = np.append(ties, Node(fringe[i].row, fringe[i].col, getDistance(current, fringe[i])))
                    ties = np.append(ties, Node(fringe[i+1].row, fringe[i+1].col, getDistance(current, fringe[i+1])))
                else:
                    ties = np.append(ties, Node(fringe[i+1].row, fringe[i+1].col, getDistance(current, fringe[i+1])))
                i+=1
            ties = sorted(ties, key=operator.attrgetter("prob"))
            totalDistance = totalDistance + getDistance(current, Node(ties[0].row, ties[0].col, 0))
            current = ties[0]
            ties = np.empty(0, dtype='object')
        else:
            totalDistance = totalDistance + getDistance(current, Node(fringe[0].row, fringe[0].col, 0))
            current = fringe[0]
        t+=1

    return t+totalDistance

def agent2():
    d = 10
    grid, target = generateGrid(d)
    knowledgeBase = np.empty(shape=(len(grid),len(grid)),dtype='object')#knowledgeBase will represent chance for each cell to be target
    knowledgeBase.fill(1.0/(len(grid)*len(grid)))
    fringe = generateFringe2(knowledgeBase, grid)#fringe will contain best cells to search in order
    t = 0
    targetFound = False
    count = 0
    visited = np.empty(shape=(d,d), dtype='object')
    visited.fill(0)
    totalDistance = 0
    current = Node(random.randint(0,len(grid)-1), random.randint(0,len(grid)-1), 0)
    ties = np.empty(0, dtype='object')
    while(not(targetFound)):
        #print("here")
        count+=1
        if(current.row==target.row and current.col==target.col):
            if(random.uniform(0,1)>grid[target.row][target.col]):#success found target
                targetFound = True
                break
        knowledgeBase = numpyUpdate(knowledgeBase, grid[current.row][current.col], current)
        visited[current.row][current.col] = visited[current.row][current.col]+1
        fringe = updateFringe2(knowledgeBase, grid, visited)
        i = 0
        if(fringe[i].prob==fringe[i+1].prob):
            while(fringe[i].prob==fringe[i+1].prob):
                if(i==0):
                    ties = np.append(ties, Node(fringe[i].row, fringe[i].col, getDistance(current, fringe[i])))
                    ties = np.append(ties, Node(fringe[i+1].row, fringe[i+1].col, getDistance(current, fringe[i+1])))
                else:
                    ties = np.append(ties, Node(fringe[i+1].row, fringe[i+1].col, getDistance(current, fringe[i+1])))
                i+=1
            ties = sorted(ties, key=operator.attrgetter("prob"))
            totalDistance = totalDistance + getDistance(current, Node(ties[0].row, ties[0].col, 0))
            current = ties[0]
            ties = np.empty(0, dtype='object')
        else:
            totalDistance = totalDistance + getDistance(current, Node(fringe[0].row, fringe[0].col, 0))
            current = fringe[0]
        t+=1

    return t+totalDistance

def getDistance(node1, node2):
    return abs(node1.row-node2.row) + abs(node1.col-node2.col)

def agent():
    grid, target = generateGrid(50)
    knowledgeBase = np.empty(shape=(len(grid),len(grid)),dtype='object')#knowledgeBase will represent chance for each cell to be target
    knowledgeBase.fill(1.0/(len(grid)*len(grid)))
    fringe = generateFringe2(knowledgeBase, grid)#fringe will contain best cells to search in order
    t = 0
    targetFound = False
    count = 0
    visited = []
    while(not(targetFound)):
        count+=1
        if(count%100):
            print("Working: "+str(count))
        if(fringe[0].row==target.row and fringe[0].col==target.col):
            if(random.uniform(0,1)>grid[target.row][target.col]):#success found target
                targetFound = True
        knowledgeBase = updateBelief(knowledgeBase, grid[fringe[0].row][fringe[0].col], fringe[0])
        visited.append(fringe[0])
        fringe = updateFringe2(knowledgeBase, grid, visited)
        t+=1
    return t

def visits(visited, node):#in this function, I use prob as visits
    times = 0
    for x in range(len(visited)):
        if(visited[x].row==node.row and visited[x].col==node.col):
            times = times + 1
    return times

def generateGrid(d):#generates a landscape with dimension d
    set = np.array([0.1,0.3,0.7,0.9]*((d*d)/4))
    grid = np.random.choice(set,size=(d,d),replace=False)
    target = Node(random.randint(0, d-1),random.randint(0, d-1),0)
    return grid, target

def updateBelief(knowledgeBase, fn, node):#conditional prob, P=(Target in cell|Observations through time)
    print(numpyUpdate(knowledgeBase, fn, node))
    cells = len(knowledgeBase)*len(knowledgeBase)
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase)):
            if(node.row==x and node.col==y):
                knowledgeBase[x][y] = fn*(1.0/cells)/((1.0-knowledgeBase[x][y])+knowledgeBase[x][y]*fn)
            else:
                knowledgeBase[x][y] = (1.0/cells)/((1.0-knowledgeBase[x][y])+knowledgeBase[x][y]*fn)
    return knowledgeBase

def probFound(grid, belief):
    return ((1.0-grid[belief.row][belief.col])*(belief.prob))

def generateFringe1(knowledgeBase, grid):
    fringe = []
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase[0])):
            fringe.append(Node(x,y,knowledgeBase[x][y]))
    fringe = sorted(fringe, key=operator.attrgetter("prob"))
    fringe.reverse()
    return fringe

def generateFringe2(knowledgeBase, grid):
    fringe = np.empty(0, dtype='object')
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase[0])):
            fringe = np.append(fringe, Node(x,y,probFound(grid, Node(x,y,knowledgeBase[x][y]))))
    fringe = sorted(fringe, key=operator.attrgetter("prob"))
    fringe.reverse()
    return fringe

def improvedFringe(knowledgeBase, grid, visited, current):
    fringe = np.empty(0, dtype='object')
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase[0])):
            v = visited[x][y]
            d = getDistance(current, Node(x,y,0))
            d = d*(len(grid)*len(grid))*.5
            if(d<1):
                d=1
            fringe = np.append(fringe, Node(x,y,(probFound(grid, Node(x,y,knowledgeBase[x][y]))/((v**v)+1.0))*((1.0/d))))
    fringe = sorted(fringe, key=operator.attrgetter("prob"))
    fringe.reverse()
    return fringe

def improvedFringe1(knowledgeBase, grid, visited, current):
    g = 1-grid
    temp1 = knowledgeBase
    temp2 = g
    temp3 = np.multiply(temp1.flatten(), g.flatten())
    temp3 = np.reshape((temp3/(np.multiply(visited.flatten(), visited.flatten())+1)), (len(grid),len(grid)))
    arr = np.unravel_index(np.argmax(temp3), temp3.shape)
    """
    for x in range()

    fringe = np.empty(0, dtype='object')
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase[0])):
            v = visited[x][y]
            d = getDistance(current, Node(x,y,0))
            d = d*(len(grid)*len(grid))*.5
            if(d<1):
                d=1
            fringe = np.append(fringe, Node(x,y,(probFound(grid, Node(x,y,knowledgeBase[x][y]))/((v**v)+1.0))*((1.0/d))))
    fringe = sorted(fringe, key=operator.attrgetter("prob"))
    fringe.reverse()
    """
    return Node(arr[0],arr[1],temp3[arr[0]][arr[1]])

def updateFringe1(knowledgeBase, grid, visited):
    fringe = []
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase[0])):
            v = visited[x][y]
            fringe.append(Node(x,y,knowledgeBase[x][y]/((v**v)+1)))
    fringe = sorted(fringe, key=operator.attrgetter("prob"))
    fringe.reverse()
    return fringe

def updateFringe2(knowledgeBase, grid, visited):
    fringe = []
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase[0])):
            v = visited[x][y]
            fringe = np.append(fringe, Node(x,y,probFound(grid, Node(x,y,knowledgeBase[x][y]))/((v**v)+1)))
    fringe = sorted(fringe, key=operator.attrgetter("prob"))
    fringe.reverse()
    return fringe

def getTotalBelief(knowledgeBase):
    sum = 0
    for x in range(len(knowledgeBase)):
        for y in range(len(knowledgeBase)):
            sum = sum + knowledgeBase[x][y]
    return sum

def printNodes(nodes):#prints all nodes by coordinates for debug purposes
    for i in range(len(nodes)):
        print("Coord: "+str(nodes[i].row)+", "+str(nodes[i].col)+" - Probability of target: "+str(nodes[i].prob))

def numpyUpdate(knowledgeBase, fn, node):#knowledgeBase, fn, node
    tempNode = Node(node.row, node.col, knowledgeBase[node.row][node.col])
    tempBase = knowledgeBase
    tempBase = (1.0/(len(knowledgeBase)*len(knowledgeBase)))/((1.0-knowledgeBase)+(knowledgeBase*fn))
    tempBase[node.row][node.col] = fn*(1.0/(len(knowledgeBase)*len(knowledgeBase)))/((1.0-tempNode.prob)+tempNode.prob*fn)
    return tempBase

def testAccuracy(trials):
    sum = 0.0
    for x in range(trials):
        print(x)
        sum = sum + improvedAgent()
    return sum/trials

def findOptima():
    for x in range():
        if(x!=0):
            print("Working: "+str(x))
            plt.plot(x, testAccuracy(1000, x/10.0), "ob")
    print("here")
    plt.title("Optima")
    plt.xlabel("Proportion")
    plt.ylabel("Avg times")
    plt.show()


print(testAccuracy(1000))
np.set_printoptions(threshold=np.inf)
#print(improvedAgent())
#print(improvedAgent())
"""
g, t = generateGrid(10)
a = np.array([[1,2],[3,4]])
b = np.array([[1,2,3],[1,2,3]])
print(np.unravel_index(np.argmax(a), a.shape))

print(a)
print(a.flatten())
print(np.reshape(a, (2,2)))

v = np.empty(shape=(10,10), dtype='object')
v = np.append(v,Node(1,2,3))
v = np.append(v,Node(3,2,1))
v = np.append(v,Node(4,5,6))
"""

#itesta()
#improvedAgent()
#print(testAccuracy(1000))
