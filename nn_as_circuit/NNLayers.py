
from __future__ import division
import math
import random

class Unit(object):    
    def __init__(self, value, grad):
	self.value = value
	self.grad = grad


class Mult(object):
    def forward(self, u0, u1):
	self.u0 = u0
	self.u1 = u1
        self.utop = Unit(u0.value*u1.value, 0.0)
	return self.utop

    def backward(self):
	self.u0.grad += self.u1.value*self.utop.grad
	self.u1.grad += self.u0.value*self.utop.grad


class Add(object):
    def forward(self, u0, u1):
	self.u0 = u0
	self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0.0)
	return self.utop

    def backward(self):
	self.u0.grad += 1 *self.utop.grad
	self.u1.grad += 1 *self.utop.grad


class Sigmoid(object):
    
    def forward(self, u0):
	self.u0 = u0
	self.sig = (1/(1+math.exp(-u0.value)))
        self.utop = Unit(self.sig, 0.0)
	return self.utop

    def backward(self):
	self.u0.grad += (self.sig*(1 - self.sig)) *self.utop.grad

def gradientCheck():
	def forwardCircuitFast(a,b,c,x,y): 
		return 1/(1 + math.exp( - (a*x + b*y + c))) 
	
	a = 1
        b = 2
        c = -3
 	x = -1
	y = 3
	h = 0.0001
	a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
	b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
	c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h
	x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h
	y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h
	
	print a_grad, b_grad, c_grad, x_grad, y_grad 


class Circuit(object):
    def __init__(self):
	self.mulg0 = Mult()
	self.mulg1 = Mult()
	self.addg0 = Add()
	self.addg1 = Add()

    def forward(self, x,y,a,b,c):
	self.ax = self.mulg0.forward(a, x) # a*x
	self.by = self.mulg1.forward(b, y) # b*y
    	self.axpby = self.addg0.forward(self.ax, self.by) # a*x + b*y
    	self.axpbypc = self.addg1.forward(self.axpby, c) # a*x + b*y + c
    	return self.axpbypc

    def backward(self, gradient_top):
	self.axpbypc.grad = gradient_top
    	self.addg1.backward() # sets gradient in axpby and c
    	self.addg0.backward() # sets gradient in ax and by
    	self.mulg1.backward() # sets gradient in b and y
    	self.mulg0.backward() # sets gradient in a and x


class SVM(object):
    def __init__(self):
	self.a = Unit(1.0,0.0)
	self.b = Unit(-2.0, 0.0)
	self.c = Unit(-1.0, 0.0)
	self.circuit = Circuit()

    def forward(self, x,y):
	self.unit_out = self.circuit.forward(x, y, self.a, self.b, self.c)
	return self.unit_out 

    def backward(self, label):
	self.a.grad = 0.0 
    	self.b.grad = 0.0 
    	self.c.grad = 0.0

    	# compute the pull based on what the circuit output was
    	pull = 0.0
    	if (label == 1 and self.unit_out.value < 1): 
      		pull = 1 # the score was too low: pull up
    	
    	if(label == -1 and self.unit_out.value > -1):
      		pull = -1 # the score was too high for a positive example, pull down
    	
    	self.circuit.backward(pull) # writes gradient into x,y,a,b,c

    	# add regularization pull for parameters: towards zero and proportional to value
    	self.a.grad += -self.a.value
    	self.b.grad += -self.b.value

    def learnFrom(self, x, y, label):
    	self.forward(x, y) # forward pass (set .value in all Units)
    	self.backward(label) # backward pass (set .grad in all Units)
    	self.parameterUpdate() # parameters respond to tug
  	
    def parameterUpdate(self):
    	step_size = 0.01
    	self.a.value += step_size * self.a.grad
    	self.b.value += step_size * self.b.grad
    	self.c.value += step_size * self.c.grad
  
def main():
	data = []
	labels = []
	data.append([1.2, 0.7]) 
	labels.append(1)
	data.append([-0.3, -0.5])
	labels.append(-1)
	data.append([3.0, 0.1]) 
	labels.append(1)
	data.append([-0.1, -1.0]) 
	labels.append(-1)
	data.append([-1.0, 1.1]) 
	labels.append(-1)
	data.append([2.1, -3]) 
	labels.append(1)
	'''
	svm = SVM()

	# a function that computes the classification accuracy
	def evalTrainingAccuracy():
  		num_correct = 0
  		for i in range(0, len(data)):
    			x = Unit(data[i][0], 0.0)
    			y = Unit(data[i][1], 0.0)
    			true_label = labels[i]

    			# see if the prediction matches the provided label
    			predicted_label = -1
			if svm.forward(x, y).value > 0:
    				predicted_label = 1
    			
			if predicted_label == true_label:
      				num_correct = num_correct + 1
    			
  		return num_correct / len(data)

	# the learning loop
	for iter in range(0, 400): 
  		# pick a random data point
  		i = int(math.floor(random.random() * len(data)))
  		x = Unit(data[i][0], 0.0)
  		y = Unit(data[i][1], 0.0)
  		label = labels[i]
  		svm.learnFrom(x, y, label)

  		if(iter % 25 == 0): # every 10 iterations... 
    			print 'training accuracy at iter ', iter, ': ',evalTrainingAccuracy()
  	'''
	a = 1
	b = -2
	c = -1 # initial parameters
	def evalTrainingAccuracy1():
  		num_correct = 0
  		for i in range(0, len(data)):
  			x = data[i][0];
  			y = data[i][1];
    			true_label = labels[i]

    			# see if the prediction matches the provided label
    			predicted_label = -1
  			score = a*x + b*y + c;
  			if score > 0:
    				predicted_label = 1
    			
			if predicted_label == true_label:
      				num_correct = num_correct + 1
  		return num_correct / len(data)

	for iter in range(0, 400): 
  		# pick a random data point
  		i = int(math.floor(random.random() * len(data)))
  		x = data[i][0];
  		y = data[i][1];
  		label = labels[i];

  		# compute pull
  		score = a*x + b*y + c;
  		pull = 0.0;
  		if label == 1 and score < 1:
			pull = 1;
  		if label == -1 and score > -1:
			pull = -1;

  		# compute gradient and update parameters
  		step_size = 0.01;
  		a += step_size * (x * pull - a); # -a is from the regularization
  		b += step_size * (y * pull - b); # -b is from the regularization
  		c += step_size * (1 * pull);
  		
		if(iter % 25 == 0): # every 10 iterations... 
    			print 'training accuracy at iter ', iter, ': ',evalTrainingAccuracy1()
		

def main1():
	a = Unit(1.0, 0.0)
	b = Unit(2.0, 0.0)
	c = Unit(-3.0, 0.0)
	x = Unit(-1.0, 0.0)
	y = Unit(3.0, 0.0)

	# create the gates
	mulg0 = Mult()
	mulg1 = Mult()
	addg0 = Add()
	addg1 = Add()
	sg0 = Sigmoid()
	
	def forwardNeuron():
 		ax = mulg0.forward(a, x) # a*x = -1
  		by = mulg1.forward(b, y) # b*y = 6
  		axpby = addg0.forward(ax, by) # a*x + b*y = 5
  		axpbypc = addg1.forward(axpby, c) # a*x + b*y + c = 2
  		return sg0.forward(axpbypc) # sig(a*x + b*y + c) = 0.8808
	
	s = forwardNeuron()
	
	print 'circuit output: ', s.value # prints 0.8808

	s.grad = 1.0
	sg0.backward() # writes gradient into axpbypc
	addg1.backward() # writes gradients into axpby and c
	addg0.backward() # writes gradients into ax and by
	mulg1.backward() # writes gradients into b and y
	mulg0.backward() # writes gradients into a and x
		
	step_size = 0.01
	a.value += step_size * a.grad # a.grad is -0.105
	b.value += step_size * b.grad # b.grad is 0.315
	c.value += step_size * c.grad # c.grad is 0.105
	x.value += step_size * x.grad # x.grad is 0.105
	y.value += step_size * y.grad # y.grad is 0.210

	print a.grad, b.grad, c.grad, x.grad, y.grad 
	
	s = forwardNeuron()
	
	print 'circuit output after one backprop: ', s.value # prints 0.8825
	
	gradientCheck()

if __name__ == '__main__':
	main()
