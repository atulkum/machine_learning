import cPickle as pickle
import numpy as np
import tensorflow as tf
import time
import math

class BaseTensorFlow:
    def __init__(self):
        self.batch_size = 256
        self.NUM_CLASSES = 10
        self.starting_learning_rate = 0.01
        self.train_dir = './'
        self.num_steps = 50001
        self.IMAGE_PIXELS = 784
        
    def model(self, images, input_size, output_size, isEval=None):
        raise Exception('Error', 'Not implemented')
    
    def loadData(self):
        pickle_file = 'notMNIST.pickle'

        with open(pickle_file, 'rb') as f:
          save = pickle.load(f)
          self.train_dataset = save['train_dataset']
          self.train_labels = save['train_labels']
          self.valid_dataset = save['valid_dataset']
          self.valid_labels = save['valid_labels']
          self.test_dataset = save['test_dataset']
          self.test_labels = save['test_labels']
          del save  # hint to help gc free up memory
    
          self.train_dataset = self.train_dataset.reshape((-1, self.IMAGE_PIXELS)).astype(np.float32)
          self.valid_dataset = self.valid_dataset.reshape((-1, self.IMAGE_PIXELS)).astype(np.float32)
          self.test_dataset = self.test_dataset.reshape((-1, self.IMAGE_PIXELS)).astype(np.float32)

          print 'Training set', self.train_dataset.shape, self.train_labels.shape
          print 'Validation set', self.valid_dataset.shape, self.valid_labels.shape
          print 'Test set', self.test_dataset.shape, self.test_labels.shape

    def loss_function(self,logits, labels):
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(
              concated, tf.pack([self.batch_size, self.NUM_CLASSES]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self,loss):
        train_size = self.train_dataset.shape[0]
        tf.scalar_summary(loss.op.name, loss)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.starting_learning_rate,      
            global_step * self.batch_size,  
            train_size,          
            0.95,                
            staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self,logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def preapare_placeholder_inputs(self):
        self.images_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size,self.IMAGE_PIXELS))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))

    def fill_feed_dict(self, dataset, labels, step):
        if labels.shape[0] - self.batch_size > 0:
            offset = (step * self.batch_size) % (labels.shape[0] - self.batch_size)
        else:
            offset = 0
        images_feed = dataset[offset:(offset + self.batch_size), :]
        labels_feed = labels[offset:(offset + self.batch_size)]
        feed_dict = {
            self.images_placeholder: images_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict
    
    def do_eval(self,sess,
            eval_correct,
            dataset, 
            labels):
        true_count = 0  
        steps_per_epoch = labels.shape[0] // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in xrange(steps_per_epoch):
            feed_dict = self.fill_feed_dict(dataset, labels, step)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        
        precision = 1.0*true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))
    
    def run_training(self,sess, eval_correct, train_op, loss):
        
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=sess.graph_def)
        saver = tf.train.Saver()
    
        feed_dict = self.fill_feed_dict(self.train_dataset, self.train_labels, 0)
    
        for step in xrange(self.num_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)
            
            feed_dict = self.fill_feed_dict(self.train_dataset, self.train_labels, step+1)
           
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                
            if (step + 1) % 1000 == 0 or (step + 1) == self.num_steps:
                saver.save(sess, self.train_dir, global_step=step)
                print('Training Data Eval:')
                self.do_eval(sess, eval_correct,
                    feed_dict[self.images_placeholder], feed_dict[self.labels_placeholder])
                print('Validation Data Eval:')
                self.do_eval(sess, eval_correct, self.valid_dataset, self.valid_labels)
        
    def process(self):
        with tf.Graph().as_default():
            self.preapare_placeholder_inputs()
        
            logits_train, regularizer = self.model(self.images_placeholder, 
                                    self.IMAGE_PIXELS, self.NUM_CLASSES)
            loss = self.loss_function(logits_train, self.labels_placeholder)
            loss += 5e-4 * regularizer
            train_op = self.training(loss)
        
            logits_eval = self.model(self.images_placeholder, 
                                     self.IMAGE_PIXELS, self.NUM_CLASSES, isEval=True)
            eval_correct = self.evaluation(logits_eval, self.labels_placeholder)
    
            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)
            
                self.run_training(sess, eval_correct, train_op, loss)
                print('Test Data Eval:')
                self.do_eval(sess,
                    eval_correct,
                    self.test_dataset, self.test_labels)

###########Sample example class for Logistic Regression example ##########

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

############main method#########################
if __name__ == '__main__':
    model0 = inference_hidden0()
    model0.loadData()
    model0.process()
