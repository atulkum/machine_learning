import math
import numpy

class HMM_mine(object):
    def __init__(self, A, E, I):
        self.state = {}
        self.symbol = {}
        self.state_rev = {}
        self.symbol_rev = {}
        i = 0
        for a, prob in A.iteritems():
            src, dest = a[0], a[1]
            if src not in self.state:
                self.state[src] = i
                self.state_rev[i] = src
                i += 1
            if dest not in self.state:
                self.state[dest] = i
                self.state_rev[i] = dest
                i += 1  
        j = 0
        for e, prob in E.iteritems():
            hidden, observation = e[0], e[1]
            if hidden not in self.state:
                self.state[hidden] = i
                self.state_rev[i] = hidden
                i += 1  

            if observation not in self.symbol:
                self.symbol[observation] = j
                self.symbol_rev[j] = observation
                j += 1
        n_state = len(self.state)
        n_symbol = len(self.symbol)

        self.A = numpy.zeros(shape=(n_state, n_state), dtype=float)
        for a, prob in A.iteritems():
            asrc, adst = a[0], a[1]
            self.A[self.state[asrc], self.state[adst]] = prob
        self.A /= self.A.sum(axis=1)[:, numpy.newaxis]

        self.E = numpy.zeros(shape=(n_state, n_symbol), dtype=float)
        for e, prob in E.iteritems():
            hidden, observation = e[0], e[1]
            self.E[self.state[hidden], self.symbol[observation]] = prob
        self.E /= self.E.sum(axis=1)[:, numpy.newaxis]

        self.I = [ 0.0 ] * n_state
        for a, prob in I.iteritems():
            self.I[self.state[a]] = prob
        self.I = numpy.divide(self.I, sum(self.I))
        
        self.ALog = numpy.log2(self.A)
        self.ELog = numpy.log2(self.E)
        self.ILog = numpy.log2(self.I)

    def jointProb(self, p, x):
        s = map(self.state.get, p) 
        x = map(self.symbol.get, x) 
        s0 = self.I[s[0]] 
        total_prob = s0
        for i in xrange(1, len(s)):
            total_prob *= self.A[s[i-1], s[i]] 
        for i in xrange(0, len(p)):
             total_prob *= self.E[s[i], x[i]] 
        return total_prob 


    def viterbi(self, x):
        x = map(self.symbol.get, x) 
        nrow, ncol = len(self.state), len(x)
        mat   = numpy.zeros(shape=(nrow, ncol), dtype=float) 
        matTb = numpy.zeros(shape=(nrow, ncol), dtype=int)  
        for i in xrange(0, nrow):
            mat[i, 0] = self.E[i, x[0]] * self.I[i]
        for j in xrange(1, ncol):
            for i in xrange(0, nrow):
                ep = self.E[i, x[j]]
                mx, mxi = mat[0, j-1] * self.A[0, i] * ep, 0
                for i2 in xrange(1, nrow):
                    pr = mat[i2, j-1] * self.A[i2, i] * ep
                    if pr > mx:
                        mx, mxi = pr, i2
                mat[i, j], matTb[i, j] = mx, mxi
        omx, omxi = mat[0, ncol-1], 0
        for i in xrange(1, nrow):
            if mat[i, ncol-1] > omx:
                omx, omxi = mat[i, ncol-1], i
        i, p = omxi, [omxi]
        for j in xrange(ncol-1, 0, -1):
            i = matTb[i, j]
            p.append(i)
        p = ''.join(map(lambda x: self.state_rev[x], p[::-1]))
        return omx, p 

    def jointProbL(self, p, x):
        s = map(self.state.get, p) 
        x = map(self.symbol.get, x) 
        s0 = self.ILog[s[0]] 
        total_prob = s0
        for i in xrange(1, len(s)):
            total_prob += self.ALog[s[i-1], s[i]] 
        for i in xrange(0, len(p)):
             total_prob += self.ELog[s[i], x[i]] 
        return total_prob 


    def viterbiL(self, x):
        x = map(self.symbol.get, x) 
        nrow, ncol = len(self.state), len(x)
        mat   = numpy.zeros(shape=(nrow, ncol), dtype=float) 
        matTb = numpy.zeros(shape=(nrow, ncol), dtype=int)  
        for i in xrange(0, nrow):
            mat[i, 0] = self.ELog[i, x[0]] + self.ILog[i]
        for j in xrange(1, ncol):
            for i in xrange(0, nrow):
                ep = self.ELog[i, x[j]]
                mx, mxi = mat[0, j-1] + self.ALog[0, i] + ep, 0
                for i2 in xrange(1, nrow):
                    pr = mat[i2, j-1] + self.ALog[i2, i] + ep
                    if pr > mx:
                        mx, mxi = pr, i2
                mat[i, j], matTb[i, j] = mx, mxi
        omx, omxi = mat[0, ncol-1], 0
        for i in xrange(1, nrow):
            if mat[i, ncol-1] > omx:
                omx, omxi = mat[i, ncol-1], i
        i, p = omxi, [omxi]
        for j in xrange(ncol-1, 0, -1):
            i = matTb[i, j]
            p.append(i)
        p = ''.join(map(lambda x: self.state_rev[x], p[::-1]))
        return omx, p 


