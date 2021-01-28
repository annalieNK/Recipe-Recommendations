# import modules
import numpy as np 
import pandas as pd 
from collections import Counter 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx 
import plotly.offline as py
import plotly.io as pio
import sys
import argparse

from plot import plot_network_graph

def compute_similarities(df, THRESHOLD):

    #### Create a document-term-matrix
    vectorizer = CountVectorizer(lowercase=True, min_df=1, analyzer='word', stop_words=None)

    #### one dtm with matching unique words
    onewordingredients = [["".join(i.split()) for i in inner] for inner in list(df['ingredients'])]
    original_ingredient_corpus = [" ".join(i) for i in onewordingredients]
    dtm_orignal_ingredient = vectorizer.fit_transform(original_ingredient_corpus)

    #### And another dtm where each word is its own token
    separate_words_corpus = [" ".join(i) for i in list(df['ingredients'])]
    dtm_separate_words = vectorizer.fit_transform(separate_words_corpus)

    # concatenate matrices
    dtm = np.concatenate((dtm_orignal_ingredient.toarray(), dtm_separate_words.toarray()), axis=1)

    #### Compute similarity between any two recipes
    similarity_csr = cosine_similarity(dtm, dense_output=False)

    # get similar recipes by index
    sim_recipes = np.argwhere(similarity_csr > THRESHOLD)
    sim_recipes = sim_recipes[sim_recipes[:, 0] != sim_recipes[:, 1]]
    
    return similarity_csr, sim_recipes

def build_graph(sim_recipes, RECIPE_INDEX, similarity_csr, THRESHOLD):
    
    first_order = [i[1] for i in sim_recipes if i[0] in [RECIPE_INDEX]]

    second_order = list(set([i[1] for i in sim_recipes if i[0] in first_order]))
    # remove original recipe
    if RECIPE_INDEX in second_order:
        second_order.remove(RECIPE_INDEX)
    second_order = [x for x in second_order if x not in first_order]

    third_order = list(set([i[1] for i in sim_recipes if i[0] in second_order]))
    # remove original recipe
    if RECIPE_INDEX in third_order:
        third_order.remove(RECIPE_INDEX)
    third_order = [x for x in third_order if x not in first_order+second_order]

    # get list of all recommended recipes by index
    all_recommendations = list(set([RECIPE_INDEX] + first_order + second_order + third_order))
    all_recommendations.sort()

    # keep only those recipes of interest 
    # - note that a new matrix will change the index number of the recommended recipes
    row_idx = np.array(all_recommendations)
    col_idx = np.array(all_recommendations)
    recommendation_csr = similarity_csr[row_idx[:, None], col_idx]

    # for the connected nodes keep only those pairs that have a similarity > THRESHOLD
    direct_recommendation_csr = (recommendation_csr > THRESHOLD) 

    # return the new indices of the narrowed matrix containing only the recommendations
    new_indices = [i for i in enumerate(all_recommendations)]

    # get the new index of the original recipe
    original_recipe_idx = [i[0] for i in new_indices if i[1]==RECIPE_INDEX][0]

    # get new indices of the recommendations
    first = []
    second = []
    third = []
    for idx,i in new_indices:
        if i in first_order:
            first.append(idx)
        if i in second_order:
            second.append(idx)
        if i in third_order:
            third.append(idx)

    # convert adjacency recommendation matrix to graph
    G = nx.from_numpy_matrix(direct_recommendation_csr)
    
    return all_recommendations, original_recipe_idx, first, second, third, G

# map a color to the recommendation level
def create_visualization(original_recipe_idx, first, second, third, df, all_recommendations, G, THRESHOLD):
    d = {}
    d[original_recipe_idx] = 0
    d.update({i: 1 for i in first})
    d.update({j: 2 for j in second})
    d.update({k: 3 for k in third})

    node_colors_by_position = [d[i] for i in sorted(d)]
    node_text_by_position = list(df.loc[all_recommendations]['id'].values)#list(pos.keys())

    fig = plot_network_graph(G, TITLE="Recommended recipes by distance with threshold of {}".format(THRESHOLD), list_of_colors_by_order_of_nodes=node_colors_by_position, list_of_text_by_order_of_nodes=node_text_by_position)
        
    pio.write_html(fig, '../figures/raw_code_graph_output.html') 
    
    return py.iplot(fig)

# main
def recommend_recipes(args):
    
    df = pd.read_json('../data/train.json')
    df = df.head(1000)

    THRESHOLD = (args.threshold)
    RECIPE = (args.recipe)
    RECIPE_INDEX = df[df['id']==RECIPE].index.values[0]
    
    similarity_csr, sim_recipes = compute_similarities(df, THRESHOLD)
    all_recommendations, original_recipe_idx, first, second, third, G = build_graph(sim_recipes, RECIPE_INDEX, similarity_csr, THRESHOLD)
    fig = create_visualization(original_recipe_idx, first, second, third, df, all_recommendations, G, THRESHOLD)
    return fig

# Add argument parser
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('threshold', type=float, help='Threshold to compute similarity.')
    parser.add_argument('recipe', type=int, help='Recipe ID.')

    return parser.parse_args(argv)


if __name__ == '__main__':
    # start_time = timeit.default_timer()
    recommend_recipes(parse_arguments(sys.argv[1:]))  
    # print(timeit.default_timer() - start_time) 


#### Parameters
# Run it like this: python raw_code.py .5 41995
