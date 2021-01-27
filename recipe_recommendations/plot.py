import matplotlib.pyplot as plt
import networkx as nx 
import plotly.offline as py
import plotly.graph_objects as go
import plotly


def plot_network_graph(G, list_of_colors_by_order_of_nodes, list_of_text_by_order_of_nodes, TITLE):

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
                    marker=dict(size=10, color=list_of_colors_by_order_of_nodes),
                    hoverinfo='text',
                    text=list_of_text_by_order_of_nodes,
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
    layout=dict(title=TITLE,  
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

    return fig