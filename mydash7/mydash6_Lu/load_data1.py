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
import statsmodels.stats.weightstats as st
from scipy.stats import chi2_contingency
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os as os
import csv, sqlite3

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


def Correlation_data(data2:pd.DataFrame, variables:list):
    corr = data2[variables].corr(method='kendall')   # Calculate the correlation coefficient of 1 to 3 columns of variables
    return corr



def get_show_scatter(scatter):
    #sctx = ['2019/1/1', '2019/1/2', '2019/1/3', '2019/1/4', '2019/1/5']
    #scty = [3607, 3834, 3904, 4080, 3997]
    trace = go.Scatter(
        x=scatter['x'],
        y=scatter['y'],
        mode='markers',
        name='sample',
        #labels = {'x': scatter['x_label'], 'y': scatter['y_label']}
    )

    trace1 = go.Scatter(
        x=scatter["trend_lines"][0],
        y=scatter["trend_lines"][1],
        name='mean'
    )


    fig = go.Figure(
        data = [trace,trace1],
    )
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    colors = {
        'background': '#111111',
        'text': '#111111'
    }
    fig.update_layout(
        #autosize=False,
        paper_bgcolor="LightSteelBlue",
    )
    fig.update_xaxes(
        # tickangle=90,
        title_text="Mother's age when born",
        # title_font={"size": 20},
        # title_standoff=25
    )

    fig.update_yaxes(
        title_text="Weight at birth, pounds",
        # title_text="Temperature",
        # title_standoff=25
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
        html.Div(children='The red line shows the average weight waves with mother\'s age.'+ '\nThe green line shows the average weight of babies of smoking mothers is lower in comparision.', style={
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


def get_heatmap(data_dict):
    data = data_dict['data']
    import plotly.express as px

    label_list =['MAP','MSP','WBP']
    #MAP: Mother\'s age when born. MSP: Mother smoked when pregnant. WBP: Weight at birth, pounds


    #data = [[1, 25, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, 5, 20]]

    fig = px.imshow(data,
                    #labels=dict(x="Day of Week", y="Time of Day", color="Productivity"),
                    x=label_list,
                    y=label_list,
                    )
    fig.update_xaxes(side="top")
    colors = {
        'background': '#111111',
        'text': '#7FDBFF'
    }
    colors = {
        'background': '#111111',
        'text': '#111111'
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
            children="The correlation coefficient values of the mother's age at delivery, whether the mother smoked during pregnancy, and the weight of the newborn baby were all less than 0.3, which seems to indicate that the correlation between these variables is extremely weak.",
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])
    return [row1,row2,row3]


def Test(data2, x_name, y_name, x_type, y_type, groups, normal=''):
    # x_type, y_type set to 0(Numerical), 1(Categrical)
    # groups set to 0(=2), 1(>2)
    # normal distribution set to True(yes),False(no)
    try:
        if x_type + y_type == 1 and groups == 0 and normal == True:
            # Claim that one variable is Categrical; T-test
            cate = data2[x_name] if len(data2[x_name].value_counts().index) == 2 else data2[y_name]
            cate_types = data2[x_name].value_counts().index
            nums = data2[x_name] if len(data2[x_name].value_counts().index) > 2 else data2[y_name]
            nums0 = nums[cate == cate_types[0]]
            nums1 = nums[cate == cate_types[1]]
            t, p_two, df = st.ttest_ind(nums0, nums1)
            words = 'T test：' + 't=' + str(round(t,2)) + ',P value=' + str(round(p_two,2)) + ',Freedom degree=' + str(round(df,2))
            alpha = 0.05
            if (p_two < alpha):
                words = words + '\nP<α，%s has significant difference with %s %s' % \
                        (cate.name + str(cate_types[0]), cate.name + str(cate_types[1]), nums.name)
            else:
                words = words + '\nP>α，%s has no significant difference with %s %s' % \
                        (cate.name + str(cate_types[0]), cate.name + str(cate_types[1]), nums.name)

        elif len(data2[x_name].value_counts().index) < 5 and len(data2[y_name].value_counts().index) < 5:
            #  # Claim that two variables are all Categrical; Che-test
            table = pd.crosstab(data2[x_name], data2[y_name])
            chi = chi2_contingency(table)
            words = 'Chi test：' + 'chi=' + str(round(chi[0],2)) + ',P value=' + str(round(chi[1],2)) + ',Freedom degree=' + str(round(chi[2],2))
            alpha = 0.05
            if (chi[1] < alpha):
                words = words + '\nP<α，%s has significant difference with %s.' % (x_name, y_name)
            else:
                words = words + '\nP>α，%s has no significant difference with %s.' % (x_name, y_name)
        elif x_type + y_type == 1 and groups == 1:
            #  # Claim that one variable is Categrical; Variance test
            cate = data2[x_name] if len(data2[x_name].value_counts().index) < \
                                    len(data2[y_name].value_counts().index) else data2[y_name]
            nums = data2[x_name] if len(data2[x_name].value_counts().index) > \
                                    len(data2[y_name].value_counts().index) else data2[y_name]
            aov = []
            for i in cate.value_counts().index:
                aov.append(nums[cate == i])
            aov_test = stats.f_oneway(*aov)
            words = 'Variance test：' + 'F=' + str(round(aov_test[0],2)) + ',P value=' + str(round(aov_test[1],2))
            alpha = 0.05
            if (aov_test[1] < alpha):
                words = words + '\nP<α，%s has significant difference with %s.' % (cate.name, nums.name)
            else:
                words = words + '\nP>α，%s has no significant difference with %s.' % (cate.name, nums.name)
        else:
            words = 'Error, Please re-select'
    except:
        words = 'Error, Please re-select'

    return words


def get_norm(data, all_result):
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    #return ff.create_distplot([data['x1']], group_labels=[data['label_1']], show_hist=False)
    '''
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Mother smokes', 'Mother not smokes'))
    trace1 = ff.create_distplot([data['x1']], group_labels=[data['label_1']],show_hist=True)
    trace2 = ff.create_distplot([data['x2']], group_labels=[data['label_2']], show_hist=True)
    fig = go.Figure(
        data=[trace1, trace2],
    )
    '''
    fig = ff.create_distplot([data['x2'],data['x1']], group_labels=[data['label_2'],data['label_1']], show_hist=True)
    fig.update_layout(
        #autosize=False,
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
            children=x_str,
            style={
                'textAlign': 'left',
                'color': colors['text']
            })
        for x_str in all_result.split("@#")
    ])

    row3 = html.Div([
        dcc.Graph(
            #id='disss-2',
            figure=fig
        )
    ])

    return [row1,row2,row3]

def get_data1(file_name):
    #file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = data1
    #data2 = ETL(data1)
    x_scatter = "Mother's age when born"
    y_scatter = "Weight at birth, pounds"
    scatter = Scatter_data(data2, x_scatter, y_scatter)
    scatter['x_label'] = x_scatter
    scatter['y_label'] = y_scatter

    data1.columns = ["ID", "Mother's age when born", "Mother smoked when pregnant",
                     "Receive newborn care at health facility", "Weight at birth, pounds",
                     "Doctor confirmed overweight", "How do you consider weight"]

    corr_vars2 = ["Mother's age when born", "Mother smoked when pregnant", "Weight at birth, pounds"]
    corr2 = Correlation_data(data2, corr_vars2)
    data_dict = {'data': corr2}
    data_dict['x'] = corr_vars2
    data_dict['y'] = corr_vars2

    norm_data = {}
    factor_name = 'Mother smoked when pregnant';
    x_name = 'Weight at birth, pounds'
    smoke = data2[data2[factor_name] == 1][x_name]  # 母亲吸烟婴儿体重
    nosmoke = data2[data2[factor_name] == 2][x_name]  # 母亲不吸烟婴儿体重
    x1 = smoke.values.tolist()
    x2 = nosmoke.values.tolist()
    norm_data['label_1'] = "Mother smokes"
    norm_data['label_2'] = "Mother not smokes"
    norm_data['x1'] = x1
    norm_data['x2'] = x2

    aov_test = Test(data2, 'How do you consider weight', 'Weight at birth, pounds', 1, 0, 1, normal=True)  # 方差检验
    print(aov_test)
    t_test = Test(data2, 'Mother smoked when pregnant', 'Weight at birth, pounds', 1, 0, 0, normal=True)  # T检验
    print(t_test)
    chi_test = Test(data2, 'Mother smoked when pregnant', 'Doctor confirmed overweight', 1, 1, 2)  # 卡方检验
    print(chi_test)
    all_result = "@#@#".join([aov_test, t_test, chi_test]).replace("\n", "@#")
    return get_show_scatter(scatter),get_heatmap(data_dict),get_norm(norm_data, all_result)


if __name__=="__main__":

    file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = data1
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



