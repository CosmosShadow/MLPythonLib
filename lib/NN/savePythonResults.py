# -*- coding: utf-8 -*-
import csv

def savecsv(path, data):
    csvfile = file(path, 'wb')
    writer = csv.writer(csvfile)
    for row in data:
        writer.writerow(['{:.6f}'.format(x) for x in row])
    csvfile.close()
 
def MatToList(matrix):
    outputList = []
    for i in range(matrix.shape[0]):
        outputList += list(matrix[i])
    return outputList
    
def saveResult(path, weights, biases):
    for i in range(len(weights)):
        w = MatToList(weights[i])
        b = MatToList(biases[i])
        savecsv(path + 'w' + str(i + 1) + '.csv', [w])
        savecsv(path + 'b' + str(i + 1) + '.csv', [b])
    
     
        
if __name__ == '__main__':
    data = [[1111.23456789, 1.23456789], [1.23456789, 1.23456789]]
    savecsv('test.csv', data)