import numpy as np
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from helper import get_stop_words, get_metrics, save_model, load_model, get_metrics
from config import FACEBOOK_POSTS_CSV, TWITTER_POSTS_CSV, TWITTER_DATA_DIR, FACEBOOK_DATA_DIR, STOP_WORDS_PATH, SENTIMENT_RNN_CHECKPOINT
from Initialize_Data import Initialize_Data
from Visualize import Visualize
from Post_Cleanuper import Posts_Cleansing, Text_Cleanuper

def get_without_stop_words(data, stop_words):
    newList = [w for w in data if not w in stop_words]
    return "".join(newList)

def lstm_cell(lstm_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    # Add dropout to the cell
    return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

def get_batches(x, y, batch_size=100):
    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

def build_rnn(inputType, labelType, n_words, lstm_layers, batch_size, learning_rate, lstm_size):
    tf.reset_default_graph()
    with tf.name_scope('inputs'):
        inputs_ = tf.placeholder(inputType, [None, None], name="inputs")
        labels_ = tf.placeholder(labelType, [None, None], name="labels")
        keep_prob= tf.placeholder(tf.float32, name="keep_prob")


    # Size of the embedding vectors (number of units in the embedding layer)
    embed_size = 300 

    with tf.name_scope("Embeddings"):
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

    with tf.name_scope("RNN_layers"):
        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size, keep_prob) for _ in range(lstm_layers)])
        
        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)
        
    with tf.name_scope("RNN_forward"):
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)


    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        tf.summary.histogram('predictions', predictions)
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels_, predictions)
        tf.summary.scalar('cost', cost)

    with tf.name_scope('validation'):
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    merged = tf.summary.merge_all()

    return inputs_, labels_, keep_prob, embedding, embed, cell, initial_state, outputs, final_state, predictions, cost, correct_pred, accuracy, optimizer, merged

def train(x_train, y_train, checkpoint, epochs, batch_size, initial_state, inputs_, labels_, keep_prob, merged, accuracy, cost, final_state, optimizer, saver):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('./logs/tb/train', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/tb/test', sess.graph)
        iteration = 1

        for e in range(epochs):
            state = sess.run(initial_state)
            train_acc = []

            for ii, (x, y) in enumerate(get_batches(x_train, y_train, batch_size), 1):

                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 0.5,
                        initial_state: state}
                summary, batch_acc, loss, state, _ = sess.run([merged, accuracy, cost, final_state, optimizer], feed_dict=feed)
                train_acc.append(batch_acc)
                train_writer.add_summary(summary, iteration)
            
                if iteration%20==0:
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.5f}".format(loss))
                    print("Accuracy: {:.5f}".format(np.mean(train_acc)))

                iteration +=1
                test_writer.add_summary(summary, iteration)
                saver.save(sess, checkpoint)
        saver.save(sess, checkpoint) 


def main():    
    stop_words = get_stop_words(STOP_WORDS_PATH)
    data = Initialize_Data();
    visualizer = Visualize();

    data.initialize_twitter_posts(TWITTER_POSTS_CSV, TWITTER_DATA_DIR)
    data.initialize_facebook_posts(FACEBOOK_POSTS_CSV, FACEBOOK_DATA_DIR)

    # Cleanup posts
    text_Cleanuper = Posts_Cleansing(data)
    text_Cleanuper.cleanup(Text_Cleanuper())

    train_data = []
    text_lengths = []

    for idx, text in enumerate(data.posts):
        new_text = get_without_stop_words(text, stop_words)
        text_lengths.append(len(new_text))
        train_data.append(new_text)

    labels = np.array([1 if l == "positive" else 0 for l in data.labels])
    print('Labels')

    # Text vectorization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    word_index = tokenizer.word_index
    n_words = len(word_index) + 1

    print("Fitting is complete.")

    train_seq = tokenizer.texts_to_sequences(train_data)
    print("Train_seq is complete.")

    max_sentence_length = int(round(np.percentile(text_lengths, 80)))
    train_data = pad_sequences(train_seq, maxlen = max_sentence_length)
    print("train_data is complete.")

    # Training and testing
    x_train, x_test, y_train, y_test = train_test_split(train_data, labels, test_size = 0.15, random_state = 2)

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(x_train.shape), 
          "\nTest set: \t\t{}".format(x_test.shape))
    print("label set: \t\t{}".format(y_train.shape), 
          "\nTest label set: \t\t{}".format(y_test.shape))

    # RNN hyperparameters
    lstm_size = 128
    lstm_layers = 1
    batch_size = 96
    learning_rate = 0.01
    epochs = 1

    # Create the graph object
    tf.reset_default_graph()
    inputs_, labels_, keep_prob, embedding, embed, cell, initial_state, outputs, final_state, predictions, cost, correct_pred, accuracy, optimizer, merged = build_rnn(tf.int32, tf.int32, n_words, lstm_layers, batch_size, learning_rate, lstm_size)

    checkpoint = SENTIMENT_RNN_CHECKPOINT

    # Train network
    saver = tf.train.Saver()
    
    train(x_train, y_train, checkpoint, epochs, batch_size, initial_state, inputs_, labels_, keep_prob, merged, accuracy, cost, final_state, optimizer, saver)

    test_acc = []
    test_predictions = []

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))
        for ii, (x, y) in enumerate(get_batches(x_test, y_test, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}
            batch_acc, test_pred, test_state = sess.run([accuracy, correct_pred, final_state], feed_dict=feed)
            print(test_pred)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


    # Get scores
    y_test_reshaped = y_test.reshape(y_test.shape[0], 1)
    print(y_test_reshaped)

    accuracy, precision, recall, f1 = get_metrics(y_test_reshaped, test_pred)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    cm = confusion_matrix(y_test_reshaped, test_pred)
    visualizer.plot_confusion_matrix(cm, classes=['positive','negative'], normalize=False, title='Confusion matrix')


if __name__ == '__main__':
    main()