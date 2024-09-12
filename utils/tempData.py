import numpy as np
import torch
import os

class storingData:
    def __init__(self):
        self.dirPath = './tempData/'
        self.dirForResetting = self.dirPath + 'resetParams/'
        self.createDir(self.dirPath)
        self.createDir(self.dirForResetting)

    def createDir(self, dirName):
        if os.path.exists(dirName):
            # If the directory already exists, remove its contents
            for filename in os.listdir(dirName):
                file_path = os.path.join(dirName, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            # If the directory doesn't exist, create it
            os.mkdir(dirName)

    def appendToFile(self, array, file_path, dirPath=None):
        if dirPath is None:
            file_path = self.dirPath + str(file_path)
        else:
            file_path = dirPath + str(file_path)
        with open(file_path, 'a') as file:
            # Save the tensor array to the file
            np.savetxt(file, array, delimiter=',')

    def writeToFile(self, array, file_path, dirPath=None):
        if dirPath is None:
            file_path = self.dirPath + str(file_path)
        else:
            file_path = dirPath + str(file_path)
        with open(file_path, 'w') as file:
            # Save the tensor array to the file
            np.savetxt(file, array, delimiter=',')

    def readArray(self, file_path, dirPath=None):
        if dirPath is None:
            file_path = self.dirPath + str(file_path)
        else:
            file_path = dirPath + str(file_path)
        data = np.loadtxt(file_path, delimiter=',')
        return np.transpose(data)