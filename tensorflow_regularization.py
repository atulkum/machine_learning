class inference_hidden0(BaseTensorFlow):
    def __init__(self):
        BaseTensorFlow.__init__(self)
        self.SEED = 66478 

    def model(self, images, input_size, output_size, isEval=None):

        with tf.variable_scope('softmax_linear', reuse=isEval):
            weights = tf.get_variable("weights", [input_size, output_size],
                initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(input_size)),
                          seed=self.SEED))
        
            biases = tf.get_variable("biases", [output_size],
                initializer=tf.constant_initializer(0.0))
    
            logits = tf.matmul(images, weights) + biases
            reg_linear = tf.nn.l2_loss(weights) 
    
            if isEval:  
                return logits
            else:
                regularizers = reg_linear
                return (logits, regularizers)
        
class inference_hidden1(BaseTensorFlow):
    def __init__(self):
        BaseTensorFlow.__init__(self)
        self.hidden1_units = 256
        self.SEED = 66478 

    def model(self, images, input_size, output_size, isEval=None):
        with tf.variable_scope('hidden1', reuse=isEval):
            weights = tf.get_variable("weights", [input_size, self.hidden1_units],
                initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(input_size)),
                          seed=self.SEED))    
            biases = tf.get_variable("biases", [self.hidden1_units],
                initializer=tf.constant_initializer(0.0))
    
        reg_hidden1 = tf.nn.l2_loss(weights) 
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
   
        with tf.variable_scope('softmax_linear', reuse=isEval):
            weights = tf.get_variable("weights", [self.hidden1_units, output_size],
                initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(self.hidden1_units)),
                          seed=self.SEED))
        
            biases = tf.get_variable("biases", [output_size],
                initializer=tf.constant_initializer(0.0))
    
            logits = tf.matmul(hidden1, weights) + biases
            reg_linear = tf.nn.l2_loss(weights) 
        
            if isEval:  
                return logits
            else:
                regularizers = (reg_hidden1 + reg_linear)
                return (logits, regularizers)
                
    def inference_hidden2(images, isEval=None):
  # Hidden 1
    with tf.variable_scope('hidden1', reuse=isEval):
        weights = tf.get_variable("weights", [IMAGE_PIXELS, hidden1_units],
            initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(IMAGE_PIXELS)),
                          seed=SEED))    
        biases = tf.get_variable("biases", [hidden1_units],
            initializer=tf.constant_initializer(0.0))
    
        reg_hidden1 = tf.nn.l2_loss(weights) 
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        if not isEval:
            print 'drop out added'
            hidden1 = tf.nn.dropout(hidden1, 0.5, seed=SEED)
  # Linear
    with tf.variable_scope('softmax_linear', reuse=isEval):
        weights = tf.get_variable("weights", [hidden1_units, NUM_CLASSES],
            initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(hidden1_units)),
                          seed=SEED))
        
        biases = tf.get_variable("biases", [NUM_CLASSES],
            initializer=tf.constant_initializer(0.0))
    
        logits = tf.matmul(hidden1, weights) + biases
        reg_linear = tf.nn.l2_loss(weights) 
    
        if isEval:  
            return logits
        else:
            regularizers = (reg_hidden1 + reg_linear)
            return (logits, regularizers)
    
hidden1_units = 256
hidden2_units = 128
def inference_hidden3(images, isEval=None):
    # Hidden 1
    with tf.variable_scope('hidden1', reuse=isEval):
        weights = tf.get_variable("weights", [IMAGE_PIXELS, hidden1_units],
            initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(IMAGE_PIXELS)),
                          seed=SEED))    
        biases = tf.get_variable("biases", [hidden1_units],
            initializer=tf.constant_initializer(0.0))
    
        reg_hidden1 = tf.nn.l2_loss(weights) 
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        if not isEval:
            hidden1 = tf.nn.dropout(hidden1, 0.5, seed=SEED)
    # Hidden 2
    with tf.variable_scope('hidden2', reuse=isEval):
        weights = tf.get_variable("weights", [hidden1_units, hidden2_units],
            initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(hidden1_units)),
                          seed=SEED))    
        biases = tf.get_variable("biases", [hidden2_units],
            initializer=tf.constant_initializer(0.0))
    
        reg_hidden2 = tf.nn.l2_loss(weights) 
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        if not isEval:
            hidden2 = tf.nn.dropout(hidden2, 0.5, seed=SEED)
    # Linear
    with tf.variable_scope('softmax_linear', reuse=isEval):
        weights = tf.get_variable("weights", [hidden2_units, NUM_CLASSES],
            initializer=tf.random_normal_initializer(0.0, 1.0 / math.sqrt(float(hidden2_units)),
                          seed=SEED))
        biases = tf.get_variable("biases", [NUM_CLASSES],
            initializer=tf.constant_initializer(0.0))
    
        logits = tf.matmul(hidden2, weights) + biases
        reg_linear = tf.nn.l2_loss(weights) 
    
    if isEval:  
        return logits
    else:
        regularizers = (reg_hidden1 +reg_hidden2+ reg_linear)
        return (logits, regularizers)
        
        
