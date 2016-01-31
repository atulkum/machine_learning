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
        
