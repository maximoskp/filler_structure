from flask import Flask, render_template, g, current_app
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
# __NEW
import random
import NN_polyphonic as nnp

print('--- --- --- before')

def serialize_response_matrix(m):
    r = []
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            r.append( m[i,j] )
    return r
# end makeSequencerMatrix
def deserialise_input_matrix(m, r, c):
    mnp = np.array( m )
    m_out = np.reshape(mnp, (r,c))
    return m_out
# end deserialise_input_matrix

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# __NEW random number
with app.app_context():
    current_app.rr = []
    current_app.matrix = np.zeros( (34,16) )
    current_app.input_matrix = []
    current_app.input_rows = []
    current_app.input_columns = []
    current_app.num_notes = 1

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send matrix', namespace='/test')
def test_message(mat):
    print('--- --- --- test message')
    # __NEW
    with app.app_context():
        print('rr: ', current_app.rr)
    print('mat: ', mat)
    current_app.input_matrix = mat['matrix']
    current_app.input_rows = mat['rows']
    current_app.input_columns = mat['columns']
    current_app.num_notes = mat['num_notes']
    print('current_app.input_rows: ', current_app.input_rows)
    tmpMat = deserialise_input_matrix(current_app.input_matrix, current_app.input_rows, current_app.input_columns)
    current_app.matrix[:current_app.input_rows, :current_app.input_columns] = tmpMat
    print('current_app.matrix: ', current_app.matrix)
    with app.app_context():
        current_app.model.fill_notes_in_matrix(matrix_in=current_app.matrix, num_notes=current_app.num_notes)
        current_app.matrix = current_app.model.matrix
        emit('send matrix', {'matrix': serialize_response_matrix(current_app.matrix), 'rows': current_app.model.matrix.shape[0], 'columns': current_app.model.matrix.shape[1]})

@socketio.on('my broadcast event', namespace='/test')
def test_message(message):
    emit('my response', {'data': message['data']}, broadcast=True)

@socketio.on('connect', namespace='/test')
def test_connect():
    print('--- --- --- connect')
    # __NEW
    with app.app_context():
        current_app.rr.append( random.random() )
    emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

@socketio.on('initialise', namespace='/test')
def initialise_variables():
    print('--- --- --- initialise')
    # session['seed']

if __name__ == '__main__':
    print('--- --- --- main')

    # GENERATE
    # generate seed
    with app.app_context():
        current_app.model = nnp.PolyFiller()

    socketio.run(app, host='0.0.0.0', port=8880)
    app.run(threaded=True)
