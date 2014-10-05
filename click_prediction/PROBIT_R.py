from csv import DictReader
from scipy.stats import norm
from Util import Util
import numpy as np
import json
from datetime import datetime

MAX_ABS_SURPRISE = 5.0

class PROBIT_R:
    
    def __init__(self, D, prior_probability, beta, epsilon):
        self.D = D        
        self.beta = beta
        bias_mean = norm.ppf(prior_probability) * (beta ** 2 + D)
        self.mean = [bias_mean] *D  
        self.variance = [1.] *D  
        self.epsilon = epsilon

    def get_features(self, data_row):
        x = [0]  
        for key, value in data_row.items():
            index = int(value + key[1:], 16) % self.D  
            x.append(index)
        return x
        
    def total_mean(self, x):
        return sum(self.mean[i] for i in x)
 
    def total_variance(self, x):
        return sum(self.variance[i] for i in x) + self.beta ** 2

    def predict(self, x):
        return norm.cdf(self.total_mean(x) / self.total_variance(x))
                             
    def gaussian_corrections(self, t):
        # Clipping avoids numerical issues from ~0/~0.
        t = np.clip(t, -MAX_ABS_SURPRISE, MAX_ABS_SURPRISE)
        v = norm.pdf(t) / norm.cdf(t)
        w = v * (v + t)
        return v, w
                             
    def train(self, trainfile):
        loss = 0
        for t, row in enumerate(DictReader(open(trainfile))):
            y = 1. if row['Label'] == '1' else 0.
            del row['Label'] 
            del row['Id']  

            x = self.get_features(row)
            sigma_squared = self.total_variance(x)
            totalMean = self.total_mean(x)
            y_hat = y * totalMean / sigma_squared
            loss += Util.logloss(y_hat, y)
            if t % 1000000 == 0 and t > 1:
                print('%s\tencountered: %d\tcurrent logloss: %f' % (
                    datetime.now(), t, loss/t))
                    
            g = y * totalMean / np.sqrt(sigma_squared)
            v, w = self.gaussian_corrections(g)
 
            for i in x:
                mean_delta = y * self.variance[i] / np.sqrt(sigma_squared) * v
                variance_multiplier = 1.0 - self.variance[i] / sigma_squared * w
                self.mean[i]=self.mean[i] + mean_delta
                self.variance[i]=self.variance[i] * variance_multiplier
                prior_variance = 1.0
                prior_mean = 0.0
                adjusted_variance = self.variance[i] * prior_variance /((1.0 - self.epsilon) * prior_variance +
                    self.epsilon * self.variance[i])
                adjusted_mean = adjusted_variance * ((1.0 - self.epsilon) * self.mean[i] / self.variance[i] +
                    self.epsilon * prior_mean / prior_variance)
                    
                self.mean[i] = adjusted_mean
                self.variance[i] = adjusted_variance

    def saveWeight(self, weight_filename):
        with open("mean_" + weight_filename, 'w') as out:
            json.dump(self.mean, out)
        with open("var_" + weight_filename, 'w') as out:
            json.dump(self.variance, out)