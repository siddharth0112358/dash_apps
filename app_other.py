import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import base64
import io
import os
import time
from dash.dependencies import Input, Output, State
import numpy as np
import flask
from flask_cors import CORS
import pandas as pd
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from umap import UMAP
from gensim.models import Word2Vec, FastText, KeyedVectors

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

models = ['autos_w2v_sg_5.bin', 'autos_w2v_sg_2.bin', 'autos_w2v_sg_30.bin', 'autos_w2v_cbow_30.bin', 'autos_w2v_cbow_5.bin', 'autos_w2v_cbow_2.bin', 'autos_ft_sg_5.bin']

model = 'autos_ft_sg_5.bin'
count = 1000
model_format = '1 if binary, 0 if text format'
w2v_model = KeyedVectors.load_word2vec_format(model, binary=bool(model_format))
word_vectors_matrix = np.ndarray(shape=(count, 100), dtype='float64')
word_list = []
i = 0
for word in w2v_model.wv.vocab:
    word_vectors_matrix[i] = w2v_model[word]
    word_list.append(word)
    i = i+1
    if i == count:
        break
tsne = TSNE(n_components= 3, random_state = 0, perplexity = 30, learning_rate= 50, n_iter = 250)
word_vectors_matrix_dimesions = tsne.fit_transform(word_vectors_matrix)
points = pd.DataFrame([(word, coords[0], coords[1], coords[2]) for word, coords in [(word, word_vectors_matrix_dimesions[word_list.index(word)]) for word in word_list]], columns=["word", "x", "y", "z"])
data = []
scatter =  go.Scatter3d(
#            name=points["word"],
    x=points['x'],
    y=points['y'],
    z=points['z'],
    text=points["word"],
    mode='markers',
    marker=dict(
        size=2.5,
        symbol='circle-open'
    )
)
data.append(scatter)

# Layout for the t-SNE graph
tsne_layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )


app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='model',
                options=[{'label': i, 'value': i} for i in models],
                value='autos_ft_sg_5.bin'
            ),
            NamedSlider(
                name="Perplexity",
                short="perplexity",
                min=3,
                max=100,
                step=None,
                val=30,
                marks={
                    i: str(i) for i in [3, 10, 30, 50, 100]
                },
            ),
            NamedSlider(
               name="Number of Neighbors",
               short="neighbors",
               min=2,
               max=200,
               step=None,
               val=50,
               marks={
                   i: str(i) for i in [2, 10, 50, 100, 200]
               },
           )
        ],
        style={'width': '33.33%', 'float': 'left', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='dimensions',
                options=[{'label': i, 'value': i} for i in [2, 3]],
                value= 3
            ),
            NamedSlider(
               name="Number Of Iterations",
               short="iterations",
               min=250,
               max=1000,
               step=None,
               val=500,
               marks={
                   i: str(i) for i in [250, 500, 750, 1000]
               },
           )

        ],
        style={'width': '33.33%', 'float': 'middle', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='clustering type',
                options=[{'label': i, 'value': i} for i in ['UMAP', 'TSNE']],
                value='TSNE'
            ),
            NamedSlider(
               name="Learning Rate",
               short="learning-rate",
               min=10,
               max=200,
               step=None,
               val=100,
               marks={
                   i: str(i) for i in [10, 50, 100, 200]
               },
           ),
            NamedSlider(
               name="Minium distance",
               short="min-dist",
               min=0.0,
               max=0.99,
               step=None,
               val=0.5,
               marks={
                   i: str(i) for i in [0.0, 0.25, 0.5, 0.99]
               },
            )
        ],
        style={'width': '33.33%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '70px 35px'
    }),
    html.Div([
        html.Div([
            # Data about the graph
            html.Div(
                id="kl-divergence",
                style={'display': 'none'}
            ),

            html.Div(
                id="end-time",
                style={'display': 'none'}
            ),

            html.Div(
                id="error-message",
                style={'display': 'none'}
            ),

            # The graph
            dcc.Graph(
                id='tsne-3d-plot',
                figure={
                    'data': data,
                    'layout': tsne_layout
                },
                style={
                    'height': '80vh',
                },
            )
        ],
            id="plot-div",
            className="eight columns"
        )
    ])
 ])


# Button Click --> Update graph with states
@app.callback(Output('plot-div', 'children'),
              [Input('model', 'value'),
               Input('slider-perplexity', 'value'),
               Input('slider-neighbors', 'value'),
               Input('dimensions', 'value'),
               Input('slider-iterations', 'value'),
               Input('clustering type', 'value'),
               Input('slider-learning-rate', 'value'),
               Input('slider-min-dist', 'value')
               ])

def get_vocab_dataframe_plot(model, perplexity, n_neighbors, dimensions, iterations, clustering, learning_rate, min_dist):
    count = 1000
    model_format = '1 if binary, 0 if text format'
    w2v_model = KeyedVectors.load_word2vec_format(model, binary=bool(model_format))
    word_vectors_matrix = np.ndarray(shape=(count, 100), dtype='float64')
    word_list = []
    i = 0
    for word in w2v_model.wv.vocab:
        word_vectors_matrix[i] = w2v_model[word]
        word_list.append(word)
        i = i+1
        if i == count:
            break
    if clustering == 'TSNE':
        tsne = TSNE(n_components= dimensions, random_state = 0, perplexity = perplexity, learning_rate=learning_rate, n_iter = iterations)
        word_vectors_matrix_dimesions = tsne.fit_transform(word_vectors_matrix)
    elif clustering == 'UMAP':
        umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=dimensions)
        word_vectors_matrix_dimesions = umap.fit_transform(word_vectors_matrix)
    if dimensions == 2:
        points = pd.DataFrame([(word, coords[0], coords[1]) for word, coords in [(word, word_vectors_matrix_dimesions[word_list.index(word)]) for word in word_list]], columns=["word", "x", "y"])
        data = []
        scatter = go.Scatter(
    #        name=points["word"],
            x=points['x'],
            y=points['y'],
            text=points["word"],
            mode='markers',
            marker=dict(
                size=2.5,
                symbol='circle-open'
            )
        )
        data.append(scatter)
    elif dimensions == 3:
        points = pd.DataFrame([(word, coords[0], coords[1], coords[2]) for word, coords in [(word, word_vectors_matrix_dimesions[word_list.index(word)]) for word in word_list]], columns=["word", "x", "y", "z"])
        data = []
        scatter = scatter = go.Scatter3d(
#            name=points["word"],
            x=points['x'],
            y=points['y'],
            z=points['z'],
            text=points["word"],
            mode='markers',
            marker=dict(
                size=2.5,
                symbol='circle-open'
            )
        )
        data.append(scatter)
    return [
        # Data about the graph
        html.Div(
            id="kl-divergence",
            style={'display': 'none'}
        ),

        html.Div(
            id="end-time",
            style={'display': 'none'}
        ),

        html.Div(
            id="error-message",
            style={'display': 'none'}
        ),

        # The graph
        dcc.Graph(
            id='tsne-3d-plot',
            figure={
                'data': data,
                'layout': tsne_layout
            },
            style={
                'height': '80vh',
            },
        )
    ]


if __name__ == '__main__':
    app.run_server(debug=True)
