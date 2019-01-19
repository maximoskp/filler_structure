import numpy as np
import tensorflow as tf
import copy

class PolyFiller():
    ''' initilises polyphonic model and can generate compositions with specific length
    and specific density/register constraints. It can also fill the next N notes of an
    input/given composition and return the new density/register values '''

    def __init__(self):
        print('Initialising PolyFiller')
        # retrieve saved data
        d = np.load('saved_data/training_data.npz')
        self.combined_matrix = d['combined_matrix']
        self.seed = d['seed']
        self.m = self.combined_matrix[2:, :]
        self.s = self.combined_matrix[:2, :]
        self.notes_range = self.m.shape[0]
        self.seed = d['seed']
        # test batch generation example
        self.max_len = d['max_len']
        self.composition_length = 32+32+32
        # composition structure
        self.composition_structure = np.zeros( (2, self.composition_length) )
        # 24x[0,0] , 16x[1,0] , 48x[1,1] , 16x[0,1] , 24x[0,0]
        self.composition_structure[0,16:(self.composition_length-40)]
        self.composition_structure[1,40:(self.composition_length-24)]
        self.batch_size = d['batch_size']
        self.step = 1
        self.input_rows = self.combined_matrix.shape[0]
        self.output_rows = self.m.shape[0]
        self.num_units = d['num_units']
        self.learning_rate = 0.001
        self.epochs = 5000
        self.temperature = 0.5
        # make initial piano roll empty matrix
        self.matrix = np.zeros( (self.notes_range, self.composition_length) )
        # initialise model
        tf.reset_default_graph()
        self.x = tf.placeholder("float", [None, self.max_len, self.input_rows])
        self.y = tf.placeholder("float", [None, self.output_rows])
        self.weight_out = tf.Variable(tf.random_normal([2*self.num_units[-1], self.output_rows]))
        self.bias_out = tf.Variable(tf.random_normal([self.output_rows]))
        self.prediction = self.rnn(self.x, self.weight_out, self.bias_out, self.input_rows)
        self.dist = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        self.cost = tf.reduce_mean(self.dist)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, 'saved_model/file.ckpt')
        # keep the predictions matrix - same size as matrix but with 0s
        # in the place of existing notes and prob values to all others
        self.predictions = np.zeros( (self.notes_range, self.composition_length) )
    # end constructor
    def fill_notes_in_matrix(self, matrix_in=[], num_notes=1):
        ' samples num_notes in given matrix_in with given density and register values '
        ' returns new matrix and new density and register values '
        if len(matrix_in) > 0 and np.sum(matrix_in) > 0:
            self.matrix = matrix_in
        else:
            self.matrix = np.zeros( (self.notes_range, self.composition_length) )
        for i in range(num_notes):
            print('filling note numbered: ', i)
            self.fill_single_note()
    # end fill_notes_in_matrix
    def fill_single_note(self):
        ' fills the next most probable note in self.matrix '
        # initially, update entire predictions matrix
        self.update_predictions()
        '''
        # find maximum element
        r = np.where( self.predictions == np.max(self.predictions) )
        # place note
        self.matrix[r[0][0], r[1][0]] = 1
        '''
        # sampling approach
        # sharpen large values
        self.predictions = np.power(self.predictions, 8)
        self.predictions = self.predictions/np.sum(self.predictions)
        # make predictions a 1D array
        tmpPredictions = np.reshape(self.predictions, ( self.predictions.size ))
        selection = np.random.multinomial(1, tmpPredictions, size=1)
        selection = np.reshape(selection, (self.predictions.shape[0], self.predictions.shape[1]))
        # find maximum element
        r = np.where( selection == np.max(selection) )
        # place note
        self.matrix[r[0][0], r[1][0]] = 1
    # end fill_single_note
    def update_predictions(self):
        ' runs from seed to the end of matrix -1column and updates all predictions '
        # scanning from seed to matrix
        # composition = np.array(self.seed[0,:,2:]).transpose()
        # tmpMat = copy.deepcopy(self.seed)
        # construct seed
        tmp_compact = np.vstack( (self.composition_structure[:,:32], self.matrix[:,:32]) )
        tmpMat = np.reshape( tmp_compact, [1, tmp_compact.shape[1], tmp_compact.shape[0]] )
        # for each matrix column, do prediction
        for i in range(32, self.matrix.shape[1], 1):
            # roll tmpMat according to matrix
            if i > 0:
                # remove_fist_char = self.seed[:,1:,:]
                remove_fist_char = tmpMat[:,1:,:]
                new_input = np.append(np.array(self.composition_structure[:,i]), self.matrix[:,i])
                tmpMat = np.append(remove_fist_char, np.reshape(new_input, [1, 1, self.input_rows]), axis=1)
            # make next prediction
            predicted = self.sess.run([self.prediction], feed_dict = {self.x:tmpMat})
            # currate predictions
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            # fit prediction to predictions matrix
            self.predictions[:,i] = predicted
        # zero-out predictions that correspond to starting and ending parts
        self.predictions[:,:32] = 0
        self.predictions[:,64:] = 0
        # to cdf for sampling
        self.prediction_to_cdf()
        # zero-out notes that already exist
        self.predictions[ self.matrix>0 ] = 0
    # end update_predictions
    def rnn(self, x, weight, bias, input_rows):
        '''
        define rnn cell and prediction
        '''
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self.input_rows])
        x = tf.split(x, self.max_len, 0)
        
        fw_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in self.num_units]
        # fw_stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
        bw_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n) for n in self.num_units]
        # bw_stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(fw_cells, bw_cells, x, dtype='float32')
        # outputs, states = tf.contrib.rnn.static_rnn(stacked_rnn_cell, x, dtype=tf.float32)
        prediction = tf.matmul(outputs[-1], self.weight_out) + self.bias_out
        return prediction
    # end rnn
    def prediction_to_cdf(self):
        ' converts predictions array to CDF '
        if np.sum(self.predictions) != 0:
            self.predictions = self.predictions/np.sum(self.predictions)