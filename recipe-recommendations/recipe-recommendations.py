import numpy as np 
import pandas as pd 
import streamlit as st
from collections import Counter 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import networkx as nx 
import plotly.offline as py
import plotly.graph_objects as go
import plotly

pd.set_option("display.max_colwidth", None)

#### Introduction
st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
st.markdown('<style>h2{color: blue;}</style>', unsafe_allow_html=True)

st.title("Recipe inspirations")

st.write("Often recipe websites contain a filter to look for recipes. Sometimes when you look for a recipe you don't necessarily know what filter you want to apply exactly. Do you want to filter on recipes with zucchini or carrots? Maybe you don't mind what kind of vegetable is used when the taste of the dishes are sort of similar.")
st.write("The goal of this recommendation system is to find recipes that are considered similar in taste and composition. And, to challenge your comfortzone with new recipes.")
st.write("In order to do that we use the ingredients of each recipe as the features and calculate how similar to another recipe each recipe is.")
st.write("The foundation of this recommendation system is a network graph. A network graph will provide a representation of how connected, or similar, recipes are.")
st.write("In this example the relations between recipes are explored. However, the concept described here can be applied to many other type of relations between entities or actors.")
st.write("Ok, let's continue with an example. For example, try recipe 41995 in the Mexican kitchen.")

# Load the data
df = pd.read_json('/Users/annalie/Dev/poetry-demo-2/data/train.json')
df = df.head(1000)

#### Choose your recipe of reference
st.header("Recommend recipes")

# Choose a kitchen category
category = st.selectbox(label='Select a kitchen', options= df['cuisine'].unique())

# Choose a recipe
category_subset = df[df['cuisine']==category]
recipe = st.selectbox(label='Select a recipe', options= sorted(category_subset['id'].unique()))

# get index by recipe ID
RECIPE_INDEX = df[df['id']==recipe].index.values[0]

#### Define helper functions

# create a new list where ingredients are one word
def one_word_ingredients(ingredients):
    return "".join(ingredients.split())

# convert list of lists to list of strings
def convert_tokens_to_string(tokens):
    return " ".join(tokens)

def multiple_words_ingredients(ingredients):
    return " ".join(ingredients)

def sim_recipes_by_idx(idx):
    return [i[1] for i in sim_recipes if i[0]==idx]

def get_new_index(all_recommendations):
    return [i for i in enumerate(all_recommendations)]


#### Preprocess the data and compute similarities between recipes
if recipe is not None:

    #### Create a document-term-matrix
    vectorizer = CountVectorizer(lowercase=True, min_df=1, analyzer='word', stop_words=None)

    #### one dtm with matching unique words
    onewordingredients = [[one_word_ingredients(i) for i in inner] for inner in list(df['ingredients'])]
    original_ingredient_corpus = [convert_tokens_to_string(i) for i in onewordingredients]
    dtm_orignal_ingredient = vectorizer.fit_transform(original_ingredient_corpus)


    #### And another dtm where each word is its own token
    separate_words_corpus = [multiple_words_ingredients(i) for i in list(df['ingredients'])]
    dtm_separate_words = vectorizer.fit_transform(separate_words_corpus)

    # concatenate matrices
    dtm = np.concatenate((dtm_orignal_ingredient.toarray(), dtm_separate_words.toarray()), axis=1)

    #### Compute similarity between any two recipes
    similarity_csr = cosine_similarity(dtm, dense_output=False)

    # get similar recipes by index
    sim_recipes = np.argwhere(similarity_csr > .5)
    sim_recipes = sim_recipes[sim_recipes[:, 0] != sim_recipes[:, 1]]


st.write('When you hit any of the button below it will show you a dataframe with the recipe index, the kitchen of the recipe, and the corresponding ingredients per recipe.')


#### Return similar recipes 
first_order = sim_recipes_by_idx(RECIPE_INDEX)

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


first_order_output = st.button('Find most similar recipes')
if first_order_output:
    st.dataframe(df.loc[first_order].set_index('id'))

# Find new recipes that are similar to the recipes similar to the reference recipe
second_order_output = st.button('Show me more recipes')
if second_order_output:
    st.dataframe(df.loc[second_order].set_index('id'))

# repeat
third_order_output = st.button('Let me be inspired')
if third_order_output:
    st.dataframe(df.loc[third_order].set_index('id'))


#### Build the network graph
st.header("Visualization of the recommendation system")
st.write("If you're curious what lies underneath this recommendation system, hit the button below and it will show you how this recommendation system is structured.")

model_run = st.button('Visualize the output')

if model_run:

    # get list of all recommended recipes by index
    all_recommendations = list(set([RECIPE_INDEX] + first_order + second_order + third_order))
    all_recommendations.sort()

    # keep only those recipes of interest 
    # - note that a new matrix will change the index number of the recommended recipes
    row_idx = np.array(all_recommendations)
    col_idx = np.array(all_recommendations)
    recommendation_csr = similarity_csr[row_idx[:, None], col_idx]

    # for the connected nodes keep only those pairs that have a similarity > .5
    direct_recommendation_csr = (recommendation_csr > .5) 

    # convert adjacency matrix to graph
    G = nx.from_numpy_matrix(direct_recommendation_csr)

    # return the new indices of the narrowed matrix containing only the recommendations
    new_indices = get_new_index(all_recommendations)

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

    #### Create the visualization

    # map a color to the recommendation level
    d = {}
    d[original_recipe_idx] = 0

    for i in first:
        d[i] = 1
    for j in second:
        d[j] = 2
    for k in third:
        d[k] = 3

    node_colors_by_position = [d[i] for i in sorted(d)]

    # Nodes
    # color nodes by recommendation level
    # get the location of the nodes
    pos = nx.spring_layout(G)   

    Xn = [pos[k][0] for k in pos.keys()]
    Yn = [pos[k][1] for k in pos.keys()]
    trace_nodes = dict(type='scatter',
                    x=Xn, 
                    y=Yn,
                    mode='markers',
                    marker=dict(size=10, color=node_colors_by_position),
                    hoverinfo='text',
                    text=list(df.loc[all_recommendations]['id'].values)#list(pos.keys()),
                    )

    # Edges
    Xe=[]
    Ye=[]
    for e in G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])
    trace_edges=dict(type='scatter',
                    mode='lines',
                    x=Xe,
                    y=Ye,
                    line=dict(width=1, color='#555555'),
                    hoverinfo='none'
                    )

    # visualize the network
    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )
    layout=dict(title= 'Recommended recipes by distance',  
                width=600,
                height=600,
                autosize=False,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(l=40,r=40,b=85,t=100,pad=0,),
        hovermode='closest',
        plot_bgcolor='#ffffff', #set background color            
        )


    fig = dict(data=[trace_edges, trace_nodes], layout=layout)
    
    st.plotly_chart(fig, use_container_width=True, sharing='streamlit')
    

st.write("The idea behind this recommendation system is to look for recipes that are most similar in terms of their cosine distance. Each recipe is converted to a vector throug a bag of words matrix. The cosine similarity is calculated between any two vectors. The advantage of using a cosine similarity metric is to outweigh the fact that some recipes contain very few ingredients and others contain many. By converting recipes to vectors we only consider the angel between two vectors and not the lenght. The result is a matrix where the columns and rows represent the recipes and the values the value of the similarity ranging from 0.0 to 1.0. Here, any two recipes with a threshold of more than 0.5 are considered similar.")
st.write("This method is repeated three times. The first iteration returns recipes similar to the reference recipe. The second iteration returns recipes similar to the recipes that are similar to the reference recipe. Repeated three times in total.")
st.write("Next, a network is constructed of only those recipes.")
st.write("The nodes in the network are colored by the level of similarity. The reference recipe is colored in blue; its most direct recommendations are purple; the most similar recipes to those are orange; and the furthest recommendations are yellow.")
st.write("Based on your curiousity to try out recipes with a new taste, but that show some familiarity, you may grow the network.")
st.write("Enjoy!")  
