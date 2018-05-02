
from __future__ import division
import math
import random

# random initial parameters
a1 = random.random() - 0.5; # a random number between -0.5 and 0.5
# ... similarly initialize all other parameters to randoms
for iter in range (0, 400): 
  # pick a random data point
  i = int(math.floor(random.random() * len(data)))
  x = data[i][0];
  y = data[i][1];
  label = labels[i];

  # compute forward pass
  n1 = max(0, a1*x + b1*y + c1); # activation of 1st hidden neuron
  n2 = max(0, a2*x + b2*y + c2); # 2nd neuron
  n3 = max(0, a3*x + b3*y + c3); # 3rd neuron
  score = a4*n1 + b4*n2 + c4*n3 + d4; # the score

  # compute the pull on top
  pull = 0.0;
  if(label == 1 && score < 1) pull = 1; # we want higher output! Pull up.
  if(label == -1 && score > -1) pull = -1; # we want lower output! Pull down.

  # now compute backward pass to all parameters of the model

  # backprop through the last "score" neuron
  dscore = pull;
  da4 = n1 * dscore;
  dn1 = a4 * dscore;
  db4 = n2 * dscore;
  dn2 = b4 * dscore;
  dc4 = n3 * dscore;
  dn3 = c4 * dscore;
  dd4 = 1.0 * dscore; # phew

  # backprop the ReLU non-linearities, in place
  # i.e. just set gradients to zero if the neurons did not "fire"
  dn3 = (0 if n3 == 0 else dn3)
  dn2 = (0 if n2 == 0 else dn2)
  dn1 = (0 if n1 == 0 else dn1)

  # backprop to parameters of neuron 1
  da1 = x * dn1;
  db1 = y * dn1;
  dc1 = 1.0 * dn1;

  # backprop to parameters of neuron 2
  da2 = x * dn2;
  db2 = y * dn2;
  dc2 = 1.0 * dn2;

  # backprop to parameters of neuron 3
  da3 = x * dn3;
  db3 = y * dn3;
  dc3 = 1.0 * dn3;

  # phew! End of backprop!
  # note we could have also backpropped into x,y
  # but we do not need these gradients. We only use the gradients
  # on our parameters in the parameter update, and we discard x,y

  # add the pulls from the regularization, tugging all multiplicative
  # parameters (i.e. not the biases) downward, proportional to their value
  da1 += -a1; da2 += -a2; da3 += -a3;
  db1 += -b1; db2 += -b2; db3 += -b3;
  da4 += -a4; db4 += -b4; dc4 += -c4;

  # finally, do the parameter update
  step_size = 0.01;
  a1 += step_size * da1; 
  b1 += step_size * db1; 
  c1 += step_size * dc1;
  a2 += step_size * da2; 
  b2 += step_size * db2;
  c2 += step_size * dc2;
  a3 += step_size * da3; 
  b3 += step_size * db3; 
  c3 += step_size * dc3;
  a4 += step_size * da4; 
  b4 += step_size * db4; 
  c4 += step_size * dc4; 
  d4 += step_size * dd4;
  # wow this is tedious, please use for loops in prod.
  # we're done!


X = [ [1.2, 0.7], [-0.3, 0.5], [3, 2.5] ] # array of 2-dimensional data
y = [1, -1, 1] # array of labels
w = [0.1, 0.2, 0.3] # example: random numbers
alpha = 0.1; # regularization strength

def cost(X, y, w):
	total_cost = 0.0; # L, in SVM loss function above
        N = X.length;
        for i in range(0, N):
	# loop over all data points and compute their score
     		xi = X[i];
     		score = w[0] * xi[0] + w[1] * xi[1] + w[2];
    
    		# accumulate cost based on how compatible the score is with the label
     		yi = y[i]; # label
     		costi = Math.max(0, - yi * score + 1);
    		print('example ' + i + ': xi = (' + xi + ') and label = ' + yi);
    		print '  score computed to be ' + score.toFixed(3);
    		print '  => cost computed to be ' + costi.toFixed(3);
    		total_cost += costi;
  
            
	# regularization cost: we want small weights
                reg_cost = alpha * (w[0]*w[0] + w[1]*w[1])
                print 'regularization cost for current model is ' + reg_cost.toFixed(3);
                total_cost += reg_cost;
                            
                print 'total cost is ' + total_cost.toFixed(3);
        return total_cost;


