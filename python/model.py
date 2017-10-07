import tensorflow as tf
import numpy as np
import os

class DeepCoder:
    def __init__(self, 
                 num_input = 3, 
                 embedding_size = 20, 
                 integer_range = 201, 
                 num_example = 5, 
                 max_list_len = 10,
                 num_hidden_layer = 3,
                 hidden_layer_size = 256, 
                 attribute_size = 34, 
                 batch_size = 100, 
                 num_epoch = 100,
                 learning_rate = 0.01,
                 save_dir = os.path.dirname(os.path.realpath(__file__))
                 ):
        
        """
        Args:
            num_input: Number of inputs(parameters) of a program
            embedding_size: Dimension of the integer embedding
            integer_range: Range of all possible outputs of a program
            num_example: Number of example provided per attribute
            max_list_len: Maximum length of a input/output list
            num_hidden_layer: Number of hidden layers
            hidden_layer_size: Dimension of hidden layer
            attribute_width: Number of functions in DSL
            batch_size: Number of samples per batch
            num_epoch: Number of epoch
            learning_rate: Learning rate
        """
        
        self._num_input = num_input
        self._embedding_size = embedding_size
        self._integer_range = integer_range
        self._num_example = num_example
        self._max_list_len = max_list_len
        self._num_hidden_layer = num_hidden_layer
        self._hidden_layer_size = hidden_layer_size
        self._attribute_size = attribute_size
        self._batch_size = batch_size
        self._num_epoch = num_epoch
        self._learning_rate = learning_rate
        self._save_dir = save_dir
        
        self._build_model()
        
    def _build_model(self):
        
        # Placeholders
        
        # data (x)
        # num_input + num_output ==  num_input + 1
        # max_list_len + type_vec_len == max_list_len + 2
        self._prog_data = tf.placeholder(tf.int32, shape=[None, 
                                                            self._num_example, 
                                                            self._num_input + 1, 
                                                            self._max_list_len + 2])
        
        # target (y)
        self._attribute = tf.placeholder(tf.float32, shape=[None, self._attribute_size])
        
        # Trainable variables
        
        # integer embedding
        self._integer_embedding = tf.Variable(tf.random_normal([self._integer_range + 1, self._embedding_size]), name='int_embed')
        
        # main network
        self._encoded = self._encoder(self._prog_data)
        self._decoded = self._decoder(self._encoded)
        
        self._loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self._decoded, 
                                                                            labels=self._attribute), 
                                    name='loss')
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._train_op = self._optimizer.minimize(self._loss)
        self._predict_op = tf.sigmoid(self._decoded)
        
        self._summary_writer = tf.summary.FileWriter(logdir=os.path.join(self._save_dir, 'log'), 
                                                     graph=tf.get_default_graph())
        
        s1 = tf.summary.histogram('Integer_Embedding', self._integer_embedding)
        s2 = tf.summary.scalar('Loss', self._loss)
        self._summary_op = tf.summary.merge([s1, s2])
         
        
    def _encoder(self, x):
        
        # Split type tensor and value tensor
        # Input: [batch_size, num_example, num_input + 1, max_list_len + 2]
        # Output: type: [batch_size, num_example, num_input + 1, 2]
        #         values: [batch_size, num_example, num_input + 1, max_list_len]
        types, values = tf.split(x, [2, self._max_list_len], axis=3)
        
        # Output: [batch_size, num_example, num_input + 1, max_list_len, embedding_size]
        values_embeded = tf.nn.embedding_lookup(self._integer_embedding, values)
        
        values_embeded_reduced = tf.reshape(values_embeded, [-1, 
                                                self._num_example, 
                                                self._num_input + 1, 
                                                self._max_list_len * self._embedding_size])
        
        types = tf.cast(types, tf.float32)
        
        # Output: [batch_size, num_example, num_input + 1, max_list_len*embedding_size + 2]
        x_embeded = tf.concat([types, values_embeded_reduced], axis=3)
        
        output = x_embeded
        
        # Hidden dense layers
        # Output: [batch_size, num_example, num_input + 1, hidden_layer_size]
        for i in range(self._num_hidden_layer):
            with tf.variable_scope('layer_{}'.format(i)) as scope:
                output = tf.layers.dense(inputs=output, 
                                         units=self._hidden_layer_size, 
                                         activation=tf.sigmoid)
        
        return output
        
        
        
    def _decoder(self, encoded):
        
        # Average pooling by reducing in examples
        # Input: [batch_size, num_example, num_input + 1, hidden_layer_size]
        # Output: [batch_size, num_example, (num_input + 1) * hidden_layer_size]
        reduced = tf.reshape(encoded, [-1, self._num_example, (self._num_input + 1) * self._hidden_layer_size])

        # Output: [batch_size, (num_input + 1) * hidden_layer_size]
        pooled = tf.reduce_mean(reduced, axis=1, name='pooling')
        result = tf.layers.dense(inputs=pooled, 
                                 units=self._attribute_size, 
                                 #activation=tf.sigmoid, 
                                 name='decoded')

        return result
    

    def train(self, data, target):
        
        # Ratio of train test split
        split_idx = len(data)//10*9
        
        # Shuffle training data, unfortunately shuffle cannot take two arrays
        # A random seed is created randomly and is used to see both shuffles
        some_seed = np.random.randint(1000000)
        np.random.seed(some_seed)
        np.random.shuffle(data)
        np.random.seed(some_seed)
        np.random.shuffle(target)

        # Generate train and test data
        train_data, train_target= data[:split_idx], target[:split_idx]
        test_data, test_target= data[split_idx:], target[split_idx:]
        
        # Random batch
        def get_batch(d, t):
            idx = np.random.choice(d.shape[0], size=self._batch_size)
            return d[idx], t[idx]
        
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            for ep in range(1, self._num_epoch + 1):
                for _ in range(len(data)//self._batch_size):
                    train_data_batch, train_target_batch = get_batch(train_data, train_target)
                    feed = {self._prog_data: train_data_batch, self._attribute: train_target_batch}
                    sess.run(self._train_op, feed)
                loss, summary = sess.run([self._loss, self._summary_op], feed)

                # Report loss every epoch
                print('Epoch: ', ep, 'Loss: ',loss)

                # Evaluate model every 10 epoches
                if ep%10 == 0:

                    test_data_batch, test_target_batch = get_batch(test_data, test_target)
                    test = sess.run(self._loss, feed_dict={self._prog_data: test_data_batch, 
                                                           self._attribute: test_target_batch})
                    self._summary_writer.add_summary(summary, ep)
                    print('Epoch: ', ep, 'Test loss:',test)

                # Save model every 20 epoches
                if ep%20 == 0:
                    saver.save(sess, os.path.join(self._save_dir, 'model/model'), global_step=ep)
                    
    def predict(self, data):

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver = tf.train.import_meta_graph(os.path.join(self._save_dir, 
                                                            'model/model-{}.meta'.format(self._num_epoch)))
            saver.restore(sess,tf.train.latest_checkpoint(os.path.join(self._save_dir, 'model/')))
            return sess.run(self._predict_op, feed_dict={self._prog_data: data})
            
  