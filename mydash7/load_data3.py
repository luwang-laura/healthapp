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
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
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


def Line_data(data2:pd.DataFrame, x_name:str, y_name:str):
    count_mean = data2.mean()[y_name]  # Calculate the overall average value
    # Calculate the mean of different age groups
    mean_result = data2.groupby(x_name).mean()[y_name]
    x_data = [str(int(i)) for i in mean_result.index]
    y_data = mean_result.values
    line = {"x":x_data, "y":y_data, "px":[x_data[0], x_data[-1]], "py":[count_mean,count_mean]}
    return line



def get_show_scatter(scatter):
    #sctx = ['2019/1/1', '2019/1/2', '2019/1/3', '2019/1/4', '2019/1/5']
    #scty = [3607, 3834, 3904, 4080, 3997]
    trace = go.Scatter(
        x=scatter['x'],
        y=scatter['y'],
        mode='markers',
        name='points',
        #labels = {'x': scatter['x_label'], 'y': scatter['y_label']}
    )

    trace1 = go.Scatter(
        x=scatter["trend_lines"][0],
        y=scatter["trend_lines"][1],
        name='count_mean'
    )

    fig = go.Figure(
        data = [trace,trace1],
    )

    fig.update_xaxes(
        # tickangle=90,
        title_text="Mother's age when born",
        #title_font={"size": 20},
        #title_standoff=25
        )

    fig.update_yaxes(
        title_text="Weight at birth, pounds",
        # title_text="Temperature",
        #title_standoff=25
        #
        )


    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
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
            children='The Average',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='The Average',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])

    row3 = html.Div([
        dcc.Graph(
            #id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]



def get_line(data_dict):
    x_line = "Mother's age when born"
    y_line = "Weight at birth, pounds"
    trace = go.Scatter(
        x=data_dict['x'],
        y=data_dict['y'],
        name='mean_result',
        #labels = {'x': x_line, 'y': y_line}
    )

    trace1 = go.Scatter(
        x=data_dict["px"],
        y=data_dict["py"],
        name='count mean'
    )


    fig = go.Figure(
        data=[trace, trace1],
    )
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )

    fig.update_xaxes(
        #tickangle=90,
        title_text="Mother's age when born",
        #title_font={"size": 20},
        #title_standoff=25
        )

    fig.update_yaxes(
        title_text="Weight at birth, pounds",
        #title_text="Temperature",
        #title_standoff=25
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
            children='Line Chart',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='The red line shows the average weight changes with mother\'s age'+ '\nThe green line shows the average weight of babies of smoking mothers is lower in comparision.',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])

    row3 = html.Div([
        dcc.Graph(
            #id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]

def get_pie(des_data):
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig = go.Figure(
        data=[go.Bar(x=des_data.axes[1], y=des_data.loc['mean'],stacked=True)],
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
            children='Compare Mean',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='Compare Mean',
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
            #id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]


def get_norm(data):
    headerColor = 'grey'
    rowEvenColor = 'lightgrey'
    rowOddColor = 'white'
    values = [data.axes[0]]
    for i in range(len(data.columns)):
        values.append(list(data.iloc[:, i]))

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[['EXPENSES'] + data.columns],
            line_color='darkslategray',
            fill_color=headerColor,
            align=['left', 'center'],
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=values,
            line_color='darkslategray',
            # 2-D list of colors for alternating rows
            fill_color=[[rowOddColor, rowEvenColor, rowOddColor, rowEvenColor, rowOddColor] * 5],
            align=['left', 'center'],
            font=dict(color='darkslategray', size=11)
        ))
    ])

    return fig

def Describe_data(data2, factor_name = 'Mother smoked when pregnant', x_name = 'Weight at birth, pounds'):
    smoke = data2[data2[factor_name]==1][x_name] #母亲吸烟婴儿体重
    nosmoke = data2[data2[factor_name]==2][x_name] #母亲不吸烟婴儿体重
    des = pd.concat([smoke.describe(), nosmoke.describe()],axis=1)
    des.columns=['Mother smoked when preganent','Mother not smoke when preganent']
    return des


def get_box(data):
    fig = go.Figure()
    fig.add_trace(go.Box(y=list(data.values())[0], name="no"))
    fig.add_trace(go.Box(y=list(data.values())[1], name="yes"))
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

    fig.update_xaxes(
        # tickangle=90,
        title_text="Receive newborn care at health facility",
        # title_font={"size": 20},
        # title_standoff=25
    )

    fig.update_yaxes(
        title_text="Weight at birth, pounds",
        # title_text="Temperature",
        # title_standoff=25
    )

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
    fig.update_yaxes(
        title_text="Weight at birth, pounds",
        # title_text="Temperature",
        # title_standoff=25
    )

    row3 = html.Div([
        dcc.Graph(
            # id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]

def Box_data(data2:pd.DataFrame, factor_name: str, x_name:str):
    factor = data2[factor_name].value_counts().index
    box = {}
    for i in factor:
        box.update({i : list(data2[data2[factor_name] == i][x_name])})
    return box

def Hist_data(data2:pd.DataFrame, x_name:str):
    hist = list(data2[x_name])
    return hist

def get_Hist_data(des_data):
    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig = go.Figure(
        data=[go.Bar(y=des_data)],
    )
    fig = px.histogram(np.array(des_data), nbins=10, range_x=[0,20])
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    #fig.update_layout(bargap=0.2)
    colors = {
        'background': '#111111',
        'text': '#111111'
    }
    row1 = html.Div([
        html.H2(
            children='Histogram',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='It shows the frequency of the baby\'s weight, most babies weight at 6-7 pounds',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(
        xaxis_title='Weight at birth, pounds',
        yaxis_title='Frequency',
        # yaxis=dict(scaleanchor="x", scaleratio=1),
        # xaxis=dict(constrain='domain'),
    )

    row3 = html.Div([
        dcc.Graph(
            #id='MLreslut',
            figure=fig
        )
    ])

    return [row1, row2, row3]
def get_data3(file_name):
    #file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = ETL(data1)
    x_scatter = "Mother's age when born"
    y_scatter = "Weight at birth, pounds"
    scatter = Scatter_data(data2, x_scatter, y_scatter)
    scatter['x_label'] = x_scatter
    scatter['y_label'] = y_scatter

    x_line = "Mother's age when born"
    y_line = "Weight at birth, pounds"
    line = Line_data(data2, x_line, y_line)
    des_data = Describe_data(data2)
    box_data = Box_data(data2, "Receive newborn care at health facility", "Weight at birth, pounds")

    data2 = data1
    data22 = data2.copy()
    x_hist = "Weight at birth, pounds"
    hist = Hist_data(data22, x_hist)

    return get_line(line), get_box(box_data), get_Hist_data(hist)



if __name__=="__main__":

    file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = data1
    x_scatter = "Mother's age when born"
    y_scatter = "Weight at birth, pounds"
    scatter = Scatter_data(data2, x_scatter, y_scatter)
    scatter['x_label'] = x_scatter
    scatter['y_label'] = y_scatter

    x_line = "Mother's age when born"
    y_line = "Weight at birth, pounds"
    line = Line_data(data2, x_line, y_line)
    des_data = Describe_data(data2)
    import plotly.express as px

    app.layout = html.Div([
        dcc.Graph(
            id='show_scatter',
            figure=get_show_scatter(scatter)
        ),
        dcc.Graph(
            id='show_heatmap',
            figure=get_line(line)
        ),

        dcc.Graph(
            id='show_dis',
            figure= get_norm(des_data)
        ),

        dcc.Graph(
            id='show_dis2',
            figure= go.Figure(
                data=[go.Bar(x=des_data.axes[1], y=des_data.loc['mean'])],
            )
        ),


    ], style={'margin': 100})

    app.run_server(debug=True)


