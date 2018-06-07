import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
from sklearn.cross_validation import ShuffleSplit 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer 

from Initialize_Data import Initialize_Data
from config import FACEBOOK_POSTS_CSV, TWITTER_POSTS_CSV, TWITTER_DATA_DIR, FACEBOOK_DATA_DIR, STOP_WORDS_PATH, NAIVE_BAYES_MODEL_PATH
from Visualize import Visualize
from Post_Cleanuper import Posts_Cleansing, Text_Cleanuper
from helper import get_stop_words, get_metrics, save_model, load_model, get_most_important_features

def create_ngram_model(stop_words = 'english', mingram = 1, maxgram = 3, useBynary = False):
    tfidf_ngrams = TfidfVectorizer(ngram_range=(mingram, maxgram), analyzer="word", binary=useBynary, stop_words=stop_words)
    clf = MultinomialNB() 
    return Pipeline([('vect', tfidf_ngrams), ('clf', clf)])

# Use grid search for different params
def grid_search_model(clf_factory, X, Y, stop_words):
    cv = ShuffleSplit(n=len(X), test_size=0.3, random_state=0)
    param_grid = dict(
        vect__min_df = [1, 2],
        vect__smooth_idf = [False, True],
        vect__use_idf = [False, True],
        vect__sublinear_tf = [False, True],
        vect__binary = [False, True],
        clf__alpha = [0, 0.01, 0.05, 0.1, 0.5, 1],
    )
    
    grid_search = GridSearchCV(clf_factory(stop_words), param_grid = param_grid, cv = cv, scoring = make_scorer(f1_score), verbose=1)
    grid_search.fit(X, Y)
    
    return grid_search.best_estimator_

# Treain model
def train_test_model(clf_factory, X, Y):
    cv = ShuffleSplit(n=len(X), test_size=0.2, random_state=0)
    scores = []
    pr_scores = []
    precions = []
    recalls = []
    accuracies = []
    f1scores = []
    
    for train, test in cv:
        X_train = X[train]
        Y_train = Y[train]
        X_test = X[test]
        Y_test = Y[test]
        
        clf = clf_factory;
        clf.fit(X_train, Y_train)

        train_score = clf.score(X_train, Y_train)
        test_score = clf.score(X_test, Y_test)
        scores.append(test_score)
        probability = clf.predict(X_test)
        accuracy, precision, recall, f1 = get_metrics(Y_test, probability)
        accuracies.append(accuracy)
        precions.append(precision)
        recalls.append(recall)
        f1scores.append(f1)
        
    summary = (np.mean(accuracy), np.mean(precision), np.mean(recall), np.mean(f1))
    print("summary")
    print ("{}".format(summary)) 

    return clf;


def main():    
    stop_words = get_stop_words(STOP_WORDS_PATH)
    data = Initialize_Data();
    visualizer = Visualize();

    data.initialize_twitter_posts(TWITTER_POSTS_CSV, TWITTER_DATA_DIR)
    data.initialize_facebook_posts(FACEBOOK_POSTS_CSV, FACEBOOK_DATA_DIR)

    # Visalize daya
    df = np.array(data.posts);
    lf= np.array(data.labels);

    pos_ind = lf == "positive";
    neg_ind = lf == "negative"

    pos = df[pos_ind]
    neg = df[neg_ind]

    visualizer.plot_data_distibution([pos.shape[0], neg.shape[0]], ["positive", "negative"], "Training set distribution")

    # Cleanup posts
    text_Cleanuper = Posts_Cleansing(data)
    text_Cleanuper.cleanup(Text_Cleanuper())

    # Train and Test Model
    clf = train_test_model(create_ngram_model(frozenset(stop_words)), np.array(data.posts), np.array(data.labels) == "positive")

    # Find best Model params and train
    clf = grid_search_model(create_ngram_model, np.array(data.posts), np.array(data.labels) == "positive", frozenset(stop_words))

    print('Saving model')
    save_model(clf, NAIVE_BAYES_MODEL_PATH);

    print('Loading model')
    trained_model = load_model(NAIVE_BAYES_MODEL_PATH)

    train_test_model(trained_model, np.array(data.posts), np.array(data.labels) == "positive")

    importance = get_most_important_features(trained_model.named_steps['vect'].vocabulary_.items(), trained_model.named_steps['clf'], 10)

    top_scores = [a[0] for a in importance[0]['tops']]
    top_words = [a[1] for a in importance[0]['tops']]
    bottom_scores = [a[0] for a in importance[0]['bottom']]
    bottom_words = [a[1] for a in importance[0]['bottom']]

    visualizer.plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")


    Y_predicted_word2vec = trained_model.predict(["Նա վատ աղջիկ է"])
    print(Y_predicted_word2vec)

if __name__ == '__main__':
    main()
