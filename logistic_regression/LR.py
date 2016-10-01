from datetime import datetime
from csv import DictReader
from math import sqrt
import json
from Util import Util

class LR:
    def __init__(self, D, alpha=.1):
        self.D = D
        self.alpha = alpha
        self.w = [0.] *D  
        self.n = [0.] *D
        
    def get_features(self, data_row):
        x = [0]  
        for key, value in data_row.items():
            index = int(value + key[1:], 16) % self.D  
            x.append(index)
            
        return x
    
    def get_features_conjectured(self, data_row):
        x = [0]  
        for key1, value1 in data_row.items():
            index = int(value1 + key1[1:], 16) % self.D  
            x.append(index)
            for key2, value2 in data_row.items():
                if(key1 != key2):
                    index = int(value1 + key1[1:] + value2, 16) % self.D  
                    x.append(index)
            
        return x
        
    def get_prediction(self, x):
        wTx = 0.
        for i in x:  
            wTx += self.w[i] * 1. 
        
        return Util.sigmoid(wTx)
        
        
    def update_weight(self, x, y_hat, y):
        for i in x:
            g_i = (y-y_hat)*1
            self.n[i] += g_i*g_i
            step = self.alpha / (sqrt(self.n[i]) + 1.)
            self.w[i] = self.w[i] + g_i *step
        
    
    def train(self, train):
        loss = 0
        for t, row in enumerate(DictReader(open(train))):
            y = 1. if row['Label'] == '1' else 0.
            del row['Label'] 
            del row['Id']  
            x = self.get_features(row)
            y_hat =self. get_prediction(x)
            loss += Util.logloss(y_hat, y)
            if t % 1000000 == 0 and t > 1:
                print('%s\tencountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), t, loss/t))
            self.update_weight(x, y_hat, y)
            
    def train_conjecture(self, train):
        loss = 0
        for t, row in enumerate(DictReader(open(train))):
            y = 1. if row['Label'] == '1' else 0.
            del row['Label'] 
            del row['Id']  
            x = self.get_features_conjectured(row)
            y_hat =self.get_prediction(x)
            loss += Util.logloss(y_hat, y)
            if t % 1000000 == 0 and t > 1:
                print('%s\tencountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), t, loss/t))
            self.update_weight(x, y_hat, y)
            
            
    def saveWeight(self, weight_filename):
        with open(weight_filename, 'w') as out:
            json.dump(self.w, out)
            
    def readWeight(self, weight_filename):
        with open(weight_filename) as inp:
            self.w = json.load(inp);
            
