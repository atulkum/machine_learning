import numpy as np
import tensorflow as tf
import time
import math

batch_size = 128
NUM_CLASSES = 10
learning_rate = 0.5
train_dir = './'
num_steps = 3001
SEED = 66478  # Set to None for random seed.

def loss_function(logits, labels):
  batch_size = tf.size(labels)
  labels = tf.expand_dims(labels, 1)
  indices = tf.expand_dims(tf.range(0, batch_size), 1)
  concated = tf.concat(1, [indices, labels])
  onehot_labels = tf.sparse_to_dense(
      concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def training(loss):
  tf.scalar_summary(loss.op.name, loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def placeholder_inputs():
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def fill_feed_dict(dataset, labels, images_placeholder, labels_placeholder, step):
  if labels.shape[0] - batch_size > 0:
    offset = (step * batch_size) % (labels.shape[0] - batch_size)
  else:
    offset = 0
  images_feed = dataset[offset:(offset + batch_size), :]
  labels_feed = labels[offset:(offset + batch_size)]
  feed_dict = {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed,
  }
  return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            dataset, 
            labels):
  true_count = 0  
  steps_per_epoch = labels.shape[0] // batch_size
  num_examples = steps_per_epoch * batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(dataset, labels,
                               images_placeholder,
                               labels_placeholder, step)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = 1.0*true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
    
def run_training(get_inference, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
  with tf.Graph().as_default():
    images_placeholder, labels_placeholder = placeholder_inputs()
    
    logits_train, rgularizer = get_inference(images_placeholder, train=True)
    loss = loss_function(logits_train, labels_placeholder)
    loss += 5e-4 * rgularizer
    train_op = training(loss)
    
    logits_eval = get_inference(images_placeholder)
    eval_correct = evaluation(logits_eval, labels_placeholder)
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
    
        summary_writer = tf.train.SummaryWriter(train_dir,
                                            graph_def=sess.graph_def)
    
        feed_dict = fill_feed_dict(train_dataset, train_labels,
                                 images_placeholder,
                                 labels_placeholder, 0)
    
        for step in xrange(num_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss],
                                   feed_dict=feed_dict)
            
            feed_dict = fill_feed_dict(train_dataset, train_labels,
                                 images_placeholder,
                                 labels_placeholder, step+1)
           
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                
            if (step + 1) % 1000 == 0 or (step + 1) == num_steps:
                saver.save(sess, train_dir, global_step=step)
                print('Training Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    feed_dict[images_placeholder], feed_dict[labels_placeholder])
                print('Validation Data Eval:')
                do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    valid_dataset, valid_labels)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            test_dataset, test_labels)
            
            
  hidden1_units = 128
hidden2_units = 64

def inference_hidden3(images,train=False):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS)),
                          seed=SEED),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    reg_hidden1 = tf.nn.l2_loss(weights) 
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    if train:
        hidden1 = tf.nn.dropout(hidden1, 0.5, seed=SEED)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units)),
                          seed=SEED),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    reg_hidden2 = tf.nn.l2_loss(weights) 
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    if train:
        hidden2 = tf.nn.dropout(hidden2, 0.5, seed=SEED)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units)),
                          seed=SEED), name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits = tf.matmul(hidden2, weights) + biases
    reg_linear = tf.nn.l2_loss(weights) 
    
  if train:  
    regularizers = (reg_hidden1 + reg_hidden2 + reg_linear)
    return (logits, regularizers)
  else:
    return logits
    

run_training(inference_hidden3, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
