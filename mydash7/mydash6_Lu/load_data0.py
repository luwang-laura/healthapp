import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression #Logic Regression Model
from sklearn import svm
from sklearn.svm import SVC #SVC
from sklearn.ensemble import RandomForestClassifier #Random Forest Model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os as os
import csv, sqlite3
import plotly.graph_objects as go
import copy
import dash
import dash_core_components as dcc
import dash_html_components as html
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
import dash_bootstrap_components as dbc

def clean_data(data):
    subset_varible=["SEQN","ECD010","ECQ020","ECQ060","ECD070A","MCQ080E","WHQ030E"]  # Filter specified variables
    #data=pd.read_csv(file_name)
    data1 = data[subset_varible]
    data1 = data1[data1.ECD010 <= 60] ; data1 = data1[data1.ECQ060 <= 2]
    data1 = data1[data1.ECD070A <= 20] ; data1 = data1[data1.WHQ030E <= 3]
    data1 = data1[data1.MCQ080E <= 2] ; data1 = data1[data1.ECQ020 <= 2]#Remove abnormal data
    data1.MCQ080E[data1['MCQ080E'] == 2] = 0 #Change value 2 to 0 which means overweight
    data1 = data1.dropna() # Remove null value
    data1.columns = ["ID","Mother's age when born", "Mother smoked when pregnant",
                   "Receive newborn care at health facility", "Weight at birth, pounds",
                     "Doctor confirmed overweight", "How do you consider weight"]
    return data1


def ETL(data1:pd.DataFrame):
    conn = sqlite3.connect('database.db')
    try:
        conn.execute('DROP TABLE IF EXISTS `tan2345` ')
    except Exception as e:
        raise(e)
    finally:
        print('Table dropped')
    def create_table():
        conn = sqlite3.connect("database.db")
        try:
            create_tb_cmd='''
           create table tan2345(ID integer,Mother's age when born integer,Mother smoked when pregnant integer,Receive newborn care at health facility integer,How do you consider weight integer,Weight at birth, pounds integer,Doctor confirmed overweight integer);
            '''
            conn.execute(create_tb_cmd)
        except:
            print("Create table failed")
            return False
        #conn.execute(insert_dt_cmd)
        conn.commit()
    create_table()
    conn = sqlite3.connect("database.db")
    cu=conn.cursor()
    #Insert the newly fetched data into the database's table
    data1.to_sql('tan2345',conn, if_exists='append', index=False)
    conn.commit()
    ##Read the newest data from the database
    conn = sqlite3.connect("database.db")
    #print(conn)
    sql="SELECT * from tan2345"
    data2=pd.read_sql(sql,conn)
    return data2

def Scatter_data(data2:pd.DataFrame, x_name:str, y_name:str):
    l = data2.shape[0]
    x = list(data2[x_name].values + np.random.randint(-500, 500, l) / 1000)
    x_model = [[i] for i in x]
    y = list(data2[y_name].values + np.random.randint(-500, 500, l) / 1000)
    y_model = [[i] for i in y]
    model = linear_model.LinearRegression()
    model.fit(x_model, y_model)
    min_pre = model.predict([[min(x)]])[0][0]
    max_pre = model.predict([[max(x)]])[0][0]
    scatter = {"x":x, "y":y, "trend_lines":[[min(x), max(x)], [min_pre, max_pre]]}
    return scatter


def Correlation_data(data2:pd.DataFrame, variables:list):
    corr = data2[variables].corr(method='kendall')   # Calculate the correlation coefficient of 1 to 3 columns of variables
    return corr



def get_show_scatter():
    fig = go.Figure(
    )
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    fig.update_layout(
        #autosize=False,
        paper_bgcolor="LightSteelBlue",
    )


    row1 = html.Div([
        html.H2(
            children='Data Distribution',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])


    row2 = html.Div([
        html.Div(children='The blue line shows the average weight change with mother\'s age.', style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    ])

    row3 = html.Div([
        dcc.Graph(
            #id='example-graph-2',
            figure=fig
        )
    ])

    return [row1, row2, row3]


def get_heatmap():

    fig = go.Figure(
    )
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    fig.update_layout(
        #autosize=False,
        paper_bgcolor="LightSteelBlue",
    )



    row3 = html.Div([
        dcc.Graph(
            #id='heamp-2',
            figure=fig
        )
    ])

    row1 = html.Div([
        html.H2(
            children='The Association',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='MAP: Mother\'s age when born. MSP: Mother smoked when pregnant. WBP: Weight at birth, pounds',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    return [row1,row2,row3]


def get_norm():
    fig = go.Figure(
    )
    fig.update_layout(
        #autosize=False,
        paper_bgcolor="LightSteelBlue",
    )

    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    #fig.update_layout(l=50, r=50, t=50, b=50, title_text="Side By Side Subplots")
    row1 = html.Div([
        html.H2(
            children='The Significance',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='Normal distribution.',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])

    row3 = html.Div([
        dcc.Graph(
            #id='disss-2',
            figure=fig
        )
    ])

    return [row1,row2,row3]

def get_data0():
    return get_show_scatter(),get_heatmap(),get_norm()


if __name__=="__main__":

    file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = ETL(data1)
    x_scatter = "Mother's age when born"
    y_scatter = "Weight at birth, pounds"
    scatter = Scatter_data(data2, x_scatter, y_scatter)
    scatter['x_label'] = x_scatter
    scatter['y_label'] = y_scatter

    corr_vars2 = ["Mother's age when born", "Mother smoked when pregnant", "Weight at birth, pounds"]
    corr2 = Correlation_data(data2, corr_vars2)
    data_dict = {'data': corr2}
    data_dict['x'] = corr_vars2
    data_dict['y'] = corr_vars2

    norm_data = {}
    factor_name = 'Mother smoked when pregnant'; x_name = 'Weight at birth, pounds'
    smoke = data2[data2[factor_name]==1][x_name] #母亲吸烟婴儿体重
    nosmoke = data2[data2[factor_name]==2][x_name] #母亲不吸烟婴儿体重
    x1 = smoke.values.tolist()
    x2 = nosmoke.values.tolist()
    norm_data['label_1'] = "x"
    norm_data['label_2'] = "y"
    norm_data['x1'] = x1
    norm_data['x2'] = x2

    app.layout = html.Div([
        dcc.Graph(
            id='show_scatter',
            figure=get_show_scatter(scatter)
        ),
        dcc.Graph(
            id='show_heatmap',
            figure=get_heatmap(data_dict)
        ),

        dcc.Graph(
            id='show_dis',
            figure= get_norm(norm_data)
        ),


    ], style={'margin': 100})
    app.run_server(debug=True)



