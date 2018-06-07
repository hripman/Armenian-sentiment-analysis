from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

from Initialize_Data import Initialize_Data
from config import FACEBOOK_POSTS_CSV, TWITTER_POSTS_CSV, TWITTER_DATA_DIR, FACEBOOK_DATA_DIR, STOP_WORDS_PATH, LOG_REG_MODEL_PATH, WORD_VEC_FILE
from Visualize import Visualize
from Post_Cleanuper import Posts_Cleansing, Text_Cleanuper
from Vectorizer import Bag_Of_Words, Word_2_Vec
from helper import get_stop_words, get_metrics, save_model, load_model, get_most_important_features, get_positive_negative_words

def get_tokens_without_stop_words(data, stop_words):
    newList = [(word) for word in data if word not in stop_words]
    return newList

def get_labeled_list(data, labels, stop_words):
    labeled_list = []
    for index, item in enumerate(data):
        if(labels[index] != 'neutral'):
            row = {}
            row["text"] = item
            row["label"] = labels[index]
            row["tokens"] = get_tokens_without_stop_words(item.split(" "), stop_words)

            labeled_list.append(row)

    return labeled_list

def main():    
    stop_words = get_stop_words(STOP_WORDS_PATH)
    data = Initialize_Data();
    visualizer = Visualize();

    data.initialize_twitter_posts(TWITTER_POSTS_CSV, TWITTER_DATA_DIR)
    data.initialize_facebook_posts(FACEBOOK_POSTS_CSV, FACEBOOK_DATA_DIR)

    # Cleanup posts
    text_Cleanuper = Posts_Cleansing(data)
    text_Cleanuper.cleanup(Text_Cleanuper())

    tokenidez_list = get_labeled_list(data.posts, data.labels, stop_words)

    # Divide data into test and train set

    X_train, X_test, Y_train, Y_test = train_test_split(data.posts, data.labels, test_size=0.2, random_state=40)

    # Bag of Words model vectorization
    bag_of_words_model = Bag_Of_Words(X_train)
    bag_of_words_model.build_vectorizer(stop_words)

    X_train_counts = bag_of_words_model.data_counts
    X_test_counts = bag_of_words_model.vectorizer.transform(X_test)

    # Visualize vectorized data
    visualizer.plot_vectorized_data(X_train_counts, np.array(Y_train) == 'positive')

    # Logistic Regression model
    clf = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', 
                             multi_class='ovr', n_jobs=-1, random_state=40)
    clf.fit(X_train_counts, Y_train)

    print('Saving model')
    save_model(clf, LOG_REG_MODEL_PATH)

    print('Load model')
    trained_model = load_model(LOG_REG_MODEL_PATH)

    # Predict on text labels
    y_predicted_counts = trained_model.predict(X_test_counts)
    
    # Get model scorce     

    accuracy, precision, recall, f1 = get_metrics(Y_test, y_predicted_counts)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    # Print words importance matrix
    importance = get_most_important_features(bag_of_words_model.vectorizer.vocabulary_.items(), trained_model, 10)

    # Visualize important features   
    top_scores = [a[0] for a in importance[0]['tops']]
    top_words = [a[1] for a in importance[0]['tops']]
    bottom_scores = [a[0] for a in importance[0]['bottom']]
    bottom_words = [a[1] for a in importance[0]['bottom']]

    visualizer.plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

    # Word2Vec vectorization
    word2vecVectorization = Word_2_Vec(WORD_VEC_FILE)
    word2vecVectorization.build_vectorizer(tokenidez_list)

    X_train_word2vec, X_test_word2vec, Y_train_word2vec, Y_test_word2vec = train_test_split(word2vecVectorization.embeddings, data.labels, test_size=0.2, random_state=40)
    
    # Visualize data
    visualizer.plot_vectorized_data(word2vecVectorization.embeddings, np.array(data.labels) == 'positive')

    clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                             multi_class='multinomial', random_state=40)
    clf_w2v.fit(X_train_word2vec, Y_train_word2vec)

    print('Saving model')
    save_model(clf_w2v, LOG_REG_MODEL_PATH)

    print('Load model')
    clf_w2v = load_model(LOG_REG_MODEL_PATH)

    Y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

    accuracy_word2vec, precision_word2vec, recall_word2vec, f1_word2vec = get_metrics(Y_test_word2vec, Y_predicted_word2vec)

    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy_word2vec, precision_word2vec, 
                                                                           recall_word2vec, f1_word2vec))

    testLabel = "Խնդիրը վատ է գրված"

    word2vecVectorization.build_vectorizer([{"tokens": get_tokens_without_stop_words(testLabel.split(" "), stop_words)}])
    Y_predicted_word2vec = clf_w2v.predict(word2vecVectorization.embeddings)
    print(Y_predicted_word2vec)
    
if __name__ == '__main__':
    main()