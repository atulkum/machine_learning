from tensorflow.python.ops import candidate_sampling_ops
import tensorflow as tf
import math
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops

vocab_size = 5000
embd_size = 100
batch_size = 1

#randomly intialize context feature batch size 1
inputs = tf.Variable(tf.truncated_normal([batch_size, embd_size], stddev=1.0 / math.sqrt(embd_size)))
labels = tf.Variable(tf.constant([[23]], dtype=tf.int64))

sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=1,
          num_sampled=5,
          unique=True,
          range_max=vocab_size,
          seed=None)

sampled_id, true_expected_count, sampled_expected_count = sampled_values

labels_flat = array_ops.reshape(labels, [-1])
all_ids = array_ops.concat([labels_flat, sampled_id], 0)

weights = tf.Variable(tf.truncated_normal([vocab_size, embd_size], stddev=1.0 / math.sqrt(embd_size)))
biases = tf.Variable(tf.zeros([vocab_size]))

all_w = embedding_ops.embedding_lookup(weights, all_ids)
all_b = embedding_ops.embedding_lookup(biases, all_ids)

true_w = array_ops.slice(all_w, [0, 0], array_ops.stack([array_ops.shape(labels_flat)[0], -1]))
sampled_w = array_ops.slice(all_w, array_ops.stack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])

true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))
sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True)
true_logits = math_ops.matmul(inputs, true_w, transpose_b=True)

true_logits += true_b
sampled_logits += sampled_b

#true_logits is log (P_p(trule labele | h)) , h = inputs here
#sampled_logits is k of log ( P_n(sampleed label | h)) , h = inputs here
#

out_logits = array_ops.concat([true_logits, sampled_logits], 1)

out_labels = array_ops.concat([
        array_ops.ones_like(true_logits),
        array_ops.zeros_like(sampled_logits)
    ], 1)


sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=out_labels, logits=out_logits, name="sampled_losses")


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
print sess.run([sampled_losses])
sess.close()