from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
import numpy as np
from abc import ABC

class Vectorizer(ABC):
    def build_vectorizer(self):
        pass

class Bag_Of_Words(Vectorizer):
    def __init__(self, texts):
        self.processed_data = texts

    def build_vectorizer(self, stop_words = 'english', mingram = 1, maxgram = 3, useBynary = False):
        tfid_vectorizer = TfidfVectorizer(ngram_range=(mingram, maxgram), analyzer="word", binary=useBynary, stop_words=frozenset(stop_words))

        emb = tfid_vectorizer.fit_transform(self.processed_data)

        num_samples, num_features = emb.shape
        self.data_counts = emb
        self.vectorizer = tfid_vectorizer



class Word_2_Vec():
    def __init__(self, wordVecFile):
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(wordVecFile, binary=True)

    def build_vectorizer(self, words, generate_missing=False): 
        self.embeddings = []   
        for item in words:
            self.embeddings.append(self.get_average_word2vec(item['tokens'], self.word2vec, generate_missing=generate_missing))
                                                                                   
        self.embeddings = list(self.embeddings)

    def get_average_word2vec(self, tokens_list, word2vec, generate_missing=False, k=100):
        if len(tokens_list)<1:
            return np.zeros(k)
        if generate_missing:
            vectorized = [word2vec[word] if word in word2vec else np.random.rand(k) for word in tokens_list]
        else:
            vectorized = [word2vec[word] if word in word2vec else np.zeros(k) for word in tokens_list]
        length = len(vectorized)    
        summed = np.sum(vectorized, axis=0)
        averaged = np.divide(summed, length)
        return averaged
