Recipe Recommendations
======================

Description
-----------

Often recipe websites contain a filter to look for recipes. Sometimes
when you look for a recipe you don’t necessarily know what filter you
want to apply exactly. Do you want to filter on recipes with zucchini or
carrots? Maybe you don’t mind what kind of vegetable is used when the
taste of the dishes are sort of similar. The goal of this recommendation
system is to find recipes that are considered similar in taste and
composition. And, to challenge your comfortzone with new recipes. In
order to do that we use the ingredients of each recipe as the features
and calculate how similar to another recipe each recipe is. To build the
features we use the bag of words technique. For the similarity metric we
use the cosine similarity. The advantage of using a cosine similarity
metric is to outweigh the fact that some recipes contain very few
ingredients and others contain many. By converting recipes to vectors we
only consider the angel between two vectors and not the lenght. Hence,
the cosine similarity works well with sparse matrices. The foundation of
this recommendation system is a network graph. A network graph will
provide a representation of how connected, or similar, recipes are. The
resulting network consists only of recipes that exceed the threshold of
a similarity score of 0.5. In this example the relations between recipes
are explored. However, the concept described here can be applied to many
other type of relations between entities or actors.

The result of this recommendation system is presented with the Streamlit
application and contains an interactive network made with Plotly and
Networkx.

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

Make sure Streamlit is installed and a python version between 3.6 and
3.8 is running. The python version of this application python 3.8.6. For
all prerequisites see the dependencies in the file pyproject.toml.

Installation
~~~~~~~~~~~~

To run the application clone this repository. 

Download the data with DVC using the following line of code in the same directory in your terminal:

::

    dvc pull


To run the streamlit application and view the recommendation system run:

::

   streamlit run recipes-recommendatons.py

Usage
-----

The recommendation system created here is an example of exploring
relations between between entities. The data used here consists of text,
but the concept can be used for many other applications. Think of a
network of people and their behaviors, or a network of connected roads.

| The below figure shows an example of the output through the Streamlit
  application.
| |Recipe Recommendation Network example|

Authors and acknowledgment
--------------------------

Annalie Kruseman 
Dataset downloaded from Kaggles ‘What’s Cooking’.

.. |Recipe Recommendation Network example| image:: https://github.com/annalieNK/Recipe-Recommendations/blob/main/example.png?raw=true