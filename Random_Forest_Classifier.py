from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from Initialize_Data import Initialize_Data
from config import FACEBOOK_POSTS_CSV, TWITTER_POSTS_CSV, TWITTER_DATA_DIR, FACEBOOK_DATA_DIR, STOP_WORDS_PATH, RANDOM_FOREST_MODEL_PATH
from Visualize import Visualize
from Post_Cleanuper import Posts_Cleansing, Text_Cleanuper
from Vectorizer import Bag_Of_Words;
from helper import get_stop_words, get_metrics, save_model, load_model

def main():    
    stop_words = get_stop_words(STOP_WORDS_PATH)
    data = Initialize_Data();
    visualizer = Visualize();

    data.initialize_twitter_posts(TWITTER_POSTS_CSV, TWITTER_DATA_DIR)
    data.initialize_facebook_posts(FACEBOOK_POSTS_CSV, FACEBOOK_DATA_DIR)

    # Cleanup posts
    text_Cleanuper = Posts_Cleansing(data)
    text_Cleanuper.cleanup(Text_Cleanuper())

    # Divide data into test and train set

    X_train, X_test, Y_train, Y_test = train_test_split(data.posts, data.labels, test_size=0.2, random_state=40)

    # Bag of Words model vectorization
    bag_of_words_model = Bag_Of_Words(X_train)
    bag_of_words_model.build_vectorizer(stop_words)

    X_train_counts = bag_of_words_model.data_counts
    X_test_counts = bag_of_words_model.vectorizer.transform(X_test)

    forest = RandomForestClassifier(n_estimators = 100) 
    forest = forest.fit(X_train_counts, Y_train)
    
    y_predicted_counts_train = forest.predict(X_train_counts)

    accuracy, precision, recall, f1 = get_metrics(Y_train, y_predicted_counts_train)
    print("Train accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    y_predicted_counts = forest.predict(X_test_counts)
    
    accuracy, precision, recall, f1 = get_metrics(Y_test, y_predicted_counts)
    print("Test accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    # Find best hyperparams

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # First create the model to tune
    rf = RandomForestClassifier()

    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train_counts, Y_train)
    print('Get Best Params')
    print(rf_random.best_params_)


    print('Saving model')
    save_model(rf_random, RANDOM_FOREST_MODEL_PATH);

    print('Load model')
    trained_model = load_model(RANDOM_FOREST_MODEL_PATH)
    y_predicted_counts_train = trained_model.predict(X_train_counts)

    accuracy, precision, recall, f1 = get_metrics(Y_train, y_predicted_counts_train)
    print("Train accuracy = %.3f, precisionս = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    y_predicted_counts = trained_model.predict(X_test_counts)
    
    accuracy, precision, recall, f1 = get_metrics(Y_test, y_predicted_counts)
    print("Test accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

if __name__ == '__main__':
    main()
