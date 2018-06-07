import plotly
from plotly import graph_objs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import itertools

plotly.offline.init_notebook_mode()


class Visualize():

    def plot_data_distibution(self, datas, labels, title):
        dist = [
                graph_objs.Bar(
                    x=labels,
                    y=datas,
                )]

        plotly.offline.iplot({"data":dist, "layout":graph_objs.Layout(title=title)})

    def plot_important_words(self, top_scores, top_words, bottom_scores, bottom_words, name):
        y_pos = np.arange(len(top_words))
        top_pairs = [(a,b) for a,b in zip(top_words, top_scores)]
        top_pairs = sorted(top_pairs, key=lambda x: x[1])
        
        bottom_pairs = [(a,b) for a,b in zip(bottom_words, bottom_scores)]
        bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
        
        top_words = [a[0] for a in top_pairs]
        top_scores = [a[1] for a in top_pairs]
        
        bottom_words = [a[0] for a in bottom_pairs]
        bottom_scores = [a[1] for a in bottom_pairs]
        
        fig = plt.figure(figsize=(10, 10))  

        plt.subplot(121)
        plt.barh(y_pos,bottom_scores, align='center', alpha=0.5)
        plt.title('Negative', fontsize=20)
        plt.yticks(y_pos, bottom_words, fontsize=14)
        plt.suptitle('Key words', fontsize=16)
        plt.xlabel('Importance', fontsize=20)
        
        plt.subplot(122)
        plt.barh(y_pos,top_scores, align='center', alpha=0.5)
        plt.title('Positive', fontsize=20)
        plt.yticks(y_pos, top_words, fontsize=14)
        plt.suptitle(name, fontsize=16)
        plt.xlabel('Importance', fontsize=20)
        
        plt.subplots_adjust(wspace=0.8)
        plt.show()

    def plot_vectorized_data(self, test_data, test_labels):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        color_mapper = {label:idx for idx,label in enumerate(set(test_labels))}
        color_column = [color_mapper[label] for label in test_labels]

        colors = ['orange','blue']
        plt.scatter(lsa_scores[:,0], lsa_scores[:,1], s=8, alpha=.8, c=test_labels, cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Negative')
        green_patch = mpatches.Patch(color='blue', label='Positive')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 10})
        plt.show()

    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=30)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=20)
        plt.yticks(tick_marks, classes, fontsize=20)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 
                     color="white" if cm[i, j] < thresh else "black", fontsize=40)
        
        plt.tight_layout()
        plt.ylabel('True label', fontsize=30)
        plt.xlabel('Predicted label', fontsize=30)

        plt.show()

