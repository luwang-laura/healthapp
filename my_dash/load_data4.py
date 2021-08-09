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
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def clean_data(data):
    subset_varible=["SEQN","ECD010","ECQ020","ECQ060","ECD070A","MCQ080E","WHQ030E"]  # Filter specified variables
    #data=pd.read_csv(file_name)
    data1 = data[subset_varible]
    data1 = data1[data1.ECD010 <= 60] ; data1 = data1[data1.ECQ060 <= 2]
    data1 = data1[data1.ECD070A <= 20] ; data1 = data1[data1.WHQ030E <= 3]
    data1 = data1[data1.MCQ080E <= 2] ; data1 = data1[data1.ECQ020 <= 2]#Remove abnormal data
    data1.MCQ080E[data1['MCQ080E'] == 2] = 0 #Change value 2 to 0 which means overweight
    data1 = data1.dropna() # 去除含有空值的行
    data1.columns = ["ID","Mother's age when born", "Mother smoked when pregnant",
                   "Receive newborn care at health facility", "Weight at birth, pounds",
                     "Doctor confirmed overweight", "How do you consider weight"]
    #print(data1.shape) #Check data dimensions
    #data1.head()
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

def Pie_data(data2:pd.DataFrame, x_name:str):
    numbers = list(data2[x_name].value_counts()) #获取列ECQ020数据
    names = list(data2[x_name].value_counts().index) #获取列ECQ020有哪些值
    names = ['Mother not smoked when pregnant','Mother smoked when pregnant']
    pie = {"numbers":numbers, "names":names}
    return pie


def Correlation_data(data2:pd.DataFrame, variables:list):
    corr = data2[variables].corr(method='kendall')   # Calculate the correlation coefficient of 1 to 3 columns of variables
    return corr



def get_heatmap(data_dictx):
    data_dictx = copy.deepcopy(data_dictx)
    for k in data_dictx:
        print(k,data_dictx[k])
    x = data_dictx['a']
    y = data_dictx['b']
    data = data_dictx['data']
    print(data_dictx)

    #data = [[1, 25, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, 5, 20]]
    fig = px.imshow(data,
                    labels=dict(x="Day of Week", y="Time of Day", color="Productivity"),
                    x=x,
                    y=y,
                    )
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    colors = {
        'background': '#111111',
        'text': '#111111'
    }
    row1 = html.Div([
        html.H2(
            children='HeatMap',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='HeatMap',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )
    row3 = html.Div([
        dcc.Graph(
            # id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]

def get_norm(data):
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    return ff.create_distplot([data['x1']], group_labels=[data['label_1']], show_hist=False)
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Mother smokes', 'Mother not smokes'))
    fig.add_trace(
        ff.create_distplot([data['x1']], group_labels=[data['label_1']],show_hist=False), row=1, col=1
    )

    fig.add_trace(
        ff.create_distplot([data['x2']], group_labels=[data['label_2']],show_hist=False),
        row=1, col=2
    )

    fig.update_xaxes(title_text="Mother smoked when pregnant", row=1, col=1)
    fig.update_xaxes(title_text="Mother smoked when pregnant", row=1, col=2)

    fig.update_yaxes(title_text="Weight at birth, pounds", row=1, col=1)
    fig.update_yaxes(title_text="Weight at birth, pounds", row=1, col=2)

    fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
    return fig

def Box_data(data2:pd.DataFrame, factor_name: str, x_name:str):
    factor = data2[factor_name].value_counts().index
    box = {}
    for i in factor:
        box.update({i : list(data2[data2[factor_name] == i][x_name])})
    return box

def get_pie(pie):
    fig = go.Figure(data=[go.Pie(labels=pie["names"], values=pie["numbers"])])
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    colors = {
        'background': '#111111',
        'text': '#111111'
    }
    row1 = html.Div([
        html.H2(
            children='Pie Plot',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='Mother smoked when pregnant',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )
    row3 = html.Div([
        dcc.Graph(
            # id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]

def get_box(data):
    fig = go.Figure()
    fig.add_trace(go.Box(y=list(data.values())[0],name="Receive newborn care at health facility"))
    fig.add_trace(go.Box(y=list(data.values())[1],name="Weight at birth, pounds"))
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    colors = {
        'background': '#111111',
        'text': '#111111'
    }
    row1 = html.Div([
        html.H2(
            children='Box Plot',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='Receive newborn care at health facility # Weight at birth, pounds',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )
    row3 = html.Div([
        dcc.Graph(
            # id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]


def get_A(data_dict):
    data_dictx = copy.deepcopy(data_dict)
    fig = get_heatmap(data_dictx)



def get_B(pie_data):
    fig = get_pie(pie_data)
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    colors = {
        'background': '#111111',
        'text': '#111111'
    }
    row1 = html.Div([
        html.H2(
            children='Suggest ML Model',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='Mother smoked when pregnant',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )
    row3 = html.Div([
        dcc.Graph(
            # id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]

def get_C(box_data):
    fig = get_box(box_data)
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }

    row1 = html.Div([
        html.H2(
            children='Suggest ML Model',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='Suggest ML Model',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )
    row3 = html.Div([
        dcc.Graph(
            # id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]

def get_data4(file_name):
    #file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = data1

    x_pie = "Mother smoked when pregnant"
    pie_data = Pie_data(data2, x_pie)
    box_data = Box_data(data2, "Receive newborn care at health facility", "Weight at birth, pounds")

    corr_vars1 = ["Mother's age when born", "Mother smoked when pregnant", "Receive newborn care at health facility",
                  "Weight at birth, pounds", "Doctor confirmed overweight", "How do you consider weight"]
    corr1 = Correlation_data(data2, corr_vars1)
    data_dict = {'data': corr1}
    data_dict['a'] = corr_vars1
    data_dict['b'] = corr_vars1


    A = get_heatmap(data_dict)

    B = get_pie(pie_data)

    C = get_box(box_data)

    return A,B,C

