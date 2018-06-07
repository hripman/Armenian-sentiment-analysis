import csv
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pickle

def get_stop_words(stop_words_scv):
    words = []

    with open(stop_words_scv, encoding='utf-8') as csvfile:
        metareader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in metareader:
            word = line[0]
            words.append(word)

    return np.asarray(words)

def get_positive_negative_words(pos_file, neg_file):
    positive_words = [] 
    negative_words = []

    with open(pos_file, encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            positive_words.append(line.strip())

    with open(neg_file, encoding='utf-8') as f:
        lines = f.read().splitlines()
        for line in lines:
            positive_words.append(line.strip())

    return {"positive_words": np.asarray(positive_words), "negative_words": np.asarray(negative_words)}

def get_metrics(y_test, y_predicted):  
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')             
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')
    
    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')
    
    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def load_model(file):
    return pickle.load(open(file, 'rb'))

def get_most_important_features(vocabulary, model, n=5):
	index_to_word = {v:k for k,v in vocabulary}
    
    # loop for each class
	classes ={}
	for class_index in range(model.coef_.shape[0]):
		word_importances = [(el, index_to_word[i]) for i,el in enumerate(model.coef_[class_index])]
		sorted_coeff = sorted(word_importances, key = lambda x : x[0], reverse=True)
		tops = sorted(sorted_coeff[:n], key = lambda x : x[0])
		bottom = sorted_coeff[-n:]
		classes[class_index] = {
			'tops':tops,
			'bottom':bottom
		}
	return classes