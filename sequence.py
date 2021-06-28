'''
    Artathon - hackathon on modeling Immune cells evolution data.
    Immune cells evolution is represented as a graph of mutations present in 
    each clone. Clone is defined as a set of almost similarly evolved cells.
    
    This program takes as an input a data which is a list of evolution graphs,
    and uses unsupervised Doc2Vec model to get numeric vector representation
    for each evolution graph.
    
    This numeric representation is used to cluster the graphs and to get
    a tSNE embedding for visualization.
    
    Separate Jupyter notebooks contain additional analysis of the output data
    from this program.
    
    We use the gensim Doc2Vec implementation for the initial experiments.
    This implementation treats each mutation as a set of words.
    Mutation is represented as three words: "X_from, N_Position, Y_to"
    where X is the name of the element which is mutated, N is the position,
    and Y is the element to which X is mutated.
    
    One of possible improvements is to use a specific Doc2Vec model
    where additional numerical description of each mutation in a graph 
    could be added (such as treating N as a number instead of a word)
'''

import pickle
import pandas as pd
import numpy as np
from ast import literal_eval 
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def get_data():
    '''
        Reads cells evolution data.
        Data consists of clone ID and a textual representation of the 
        evolution graph (tree) for each clone.
        The textual representation of a tree consists of a set of all 
        mutations in each paths from root to each leaf in the evolution tree.
    '''
    filepath = "data/D207_mutations_per_branch(1).csv"
    print("Reading data...")
    data = pd.read_csv(filepath, converters={2:literal_eval})
    return data

def get_embeddings_doc(data):
    '''
    Trains and resurns Doc2Vec embedding of each evolutionary graph.
    Training is done on full data. Predictions are done only on 
    C20 (largest) clones in the dataset for the speed of subsequent 
    analysis steps.
    '''
    # Training on full data        
    print("Transforming data...")
    clones = list(data["mutations"])
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(clones)]
    print("Training doc2vec...")
    model = Doc2Vec(documents, vector_size=5, window=10, min_count=1, workers=8)
    # inference only on C20 data
    c20 = pickle.load(open("data/D207_c20_clones_matching_paper.pkl","rb"))
    data = data[data["clone_id"].isin(c20)]
    print("Predicting...")
    vectors = np.asarray([model.infer_vector(clone) for clone in clones])
    return vectors

def get_clustering(embeddings):
    '''
    k-Means clustering on the Doc2Vec embeddings
    Clustering is done before applying tSNE on the original embeddings.
    '''
    print("Clustering...")
    model = KMeans(n_clusters=5, verbose=1, n_jobs=10)
    clusters = model.fit_predict(embeddings)
    return clusters

def get_tsne(embeddings):
    '''
    tSNE embedding of the Doc2Vec embeddings.
    '''
    print("tSNE...")
    model = TSNE(verbose=1, n_jobs = 10)
    tsne = model.fit_transform(embeddings)
    return tsne

if __name__ == "__main__":
        data = get_data()    
        embeddings = get_embeddings_doc(data)
        pickle.dump(embeddings, open("embeddings.p", "wb"))
        clusters = get_clustering(embeddings)
        pickle.dump(embeddings, open("clusters.p", "wb"))
        tsne = get_tsne(embeddings)
        pickle.dump(tsne, open("tsne.p", "wb"))
        
        # saving combined data for use in Jupyter notebooks
        df = data[["clone_id",]]
        df = pd.concat([df, pd.DataFrame(embeddings)], axis=1)
        df["cluster"] = clusters
        df["x"] = tsne[:,0]
        df["y"] = tsne[:,1]
        pickle.dump(df, open("s100_combined_train_all_predict_c20.p", "wb"))
    
        # simple visualization
        plt.scatter(tsne[:,0], tsne[:,1], c=clusters, s=1)
