# Objective of this python script is to plot occupancy map data



# Import Requied packages -------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os

plt.close('all')

# Set File to Read the Map  --------------------------------------------------
fileName = "houseMulti" #the name of the map files in the format "fileName_XXXXXX.txt"
fileDirectory = os.path.join('C:\\','Users','nadur','Downloads','submap_houseMultiLoop','submap_houseMultiLoop')
mapNum = 178 #the number of the submap to display
# -----------------------------------------------------------

yFileName = os.path.join(fileDirectory , fileName + "_Ygrid.txt")
xFileName = os.path.join(fileDirectory , fileName + "_Xgrid.txt")

#pFileName = fileDirectory + fileName + "_Pgrid.txt" Used for occupancy grid maps
submapNumber = str(mapNum)
pFileName = os.path.join(fileDirectory , fileName +"_Submap_" + submapNumber + ".txt") #Used for submaps


Ygrid = np.loadtxt(yFileName)
Xgrid = np.loadtxt(xFileName)
Pgrid = np.loadtxt(pFileName)

figure(num=None, figsize=(9, 8), dpi=80, facecolor='w', edgecolor='k')

marker_size = 10
plt.scatter(Xgrid,Ygrid, marker_size, c = Pgrid, cmap='Blues') #Blues, Greys
#plt.title("Occupancy Grid Map from file: %s" %fileName)
plt.title("Submap %s from file: %s" %(mapNum,fileName))
plt.xlabel("Distance [cm]")
plt.ylabel("Distance [cm]")
plt.show() #display plot


from PIL import Image
import numpy as np

w, h = 70, 70
data = np.zeros((h, w), dtype=np.uint8)
data[:, :] = (Pgrid*255).astype(np.uint8) # red patch in upper left
# img = Image.fromarray(data)
# img.show()

from matplotlib import pyplot as plt
plt.figure()
plt.imshow(data, interpolation='nearest')
plt.show()
plt.imsave('img3.png',data)

# plt.figure()
# da =  Image.open("Figure_1.png")
# plt.imshow(da, interpolation='nearest')
# plt.show()
