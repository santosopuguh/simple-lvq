#!/usr/bin/python
'''
Simple LVQ code using NumPy
'''
import numpy as np

class LVQTrain():
    '''
    Class for train the weight
    '''
    def __init__(self):
        self.train_data = np.transpose(np.matrix([[1,0,0,1],[1,1,1,0],[0,0,0,1],[0,1,1,1]])) #Train Data
        self.weight = np.transpose(np.matrix([[1,0,1,0],[0,1,0,1]])) #Weight
        print 'Weight: \n', self.weight
        self.train_T =  np.matrix([0,0,1,1]) #Actual Class
        self.r_alpha = 0.9 #r-learning rate
        self.alpha = 0.001 #learning rate
        self.epoch = 10 #epoch
        
    def getWeight(self):
        for i in range(0, self.epoch): #loop from 1 - epoch
            for j in range(0, self.train_data.shape[1]):
                temp = np.zeros((1, self.weight.shape[1]))
                for k in range(0, self.weight.shape[1]):
                    for l in range(0, self.weight.shape[0]):
                        temp[0, k] = temp [0, k] + (self.weight[l, k] - self.train_data[l, j])**2
                
                Cj = np.argmin(temp)
                if Cj == self.train_T[0, j]:
                    temp2 = self.weight[:, Cj]
                    temp2 = temp2 + self.alpha * (self.train_data[:, j] - temp2)
                    self.weight[:, Cj] = temp2
                else:
                    temp2 = temp2[:, Cj]
                    temp2 = temp2 - self.alpha * (self.train_data[:, j] - temp2)
                    self.weight[:, Cj] = temp2
                    
            self.alpha = self.r_alpha * self.alpha
        
        print 'Updated Weight: \n', self.weight
                                          
                
class LVQTest(LVQTrain):
    '''
    class for classification
    '''
    def getClassification(self, test_data, test_T):
        self.getWeight()
        self.test_data = np.transpose(test_data)
        self.test_T = test_T
        self.classification = np.zeros((1, self.test_data.shape[1]))
        self.conf_matrix = np.zeros((2, 2))

        for j in range(0, self.test_data.shape[1]):
                temp = np.zeros((1, self.weight.shape[1]))
                for k in range(0, self.weight.shape[1]):
                    for l in range(0, self.weight.shape[0]):
                        temp[0, k] = temp [0, k] + (self.weight[l, k] - self.test_data[l, j])**2
                
                Cj = np.argmin(temp)
                self.classification[0, j] = Cj
                self.conf_matrix[self.test_T[0, j], Cj] = self.conf_matrix[self.test_T[0, j], Cj] + 1
                
        self.accuration = sum(np.diag((self.conf_matrix)))/sum(sum(self.conf_matrix))
        print'\nClassification Result:\n', self.classification
        print'Confussion Matrix:\n', self.conf_matrix
        print'Accuration:\n', self.accuration*100,'%'


if __name__ == '__main__':
    test_data = np.matrix([[0,1,1,0],[0,0,1,1]]) #Data Test
    print 'Test Data: \n', np.transpose(test_data)
    test_Cj = np.matrix([0,1]) #Data Test Target Class
    objTrain = LVQTest()
    objTrain.getClassification(test_data, test_Cj)
    