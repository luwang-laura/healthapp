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

def KMeans_data(data2, variables = ["Mother's age when born", "Mother smoked when pregnant",
                                    "Receive newborn care at health facility", "Weight at birth, pounds",
                                    "Doctor confirmed overweight", "How do you consider weight"], k = 5):
    Kdata = data2[variables] #Remove first column
    Kdata = (Kdata - Kdata.mean(axis  =0))/(Kdata.std(axis = 0)) #z-score Strandardize
    kmodel = KMeans(n_clusters = k) #聚5类
    kmodel.fit(Kdata) #训练模型
    label = pd.Series(kmodel.labels_)  # 各样本的类别
    Kdata['Type'] = label.values
    return Kdata


def Model_data(data2: pd.DataFrame):
    # No risk factor:ECQ020
    vars1 = ["Mother's age when born", "Receive newborn care at health facility",
             "Weight at birth, pounds", "How do you consider weight"]
    data1_x = data2[vars1]  # Choose X variable
    data1_y = data2["Doctor confirmed overweight"]  # Choose Y variable
    train1_x, test1_x, train1_y, test1_y = train_test_split(data1_x, data1_y, train_size=0.8, random_state=123)
    # 20% Test; 80% Train Set

    # With Risk Factor: ECQ020
    vars2 = ["Mother's age when born", "Mother smoked when pregnant", "Receive newborn care at health facility",
             "Weight at birth, pounds", "How do you consider weight"]
    data2_x = data2[vars2]  # Choose X variable
    data2_y = data2['Doctor confirmed overweight']  # Choose Y variable
    train2_x, test2_x, train2_y, test2_y = train_test_split(data2_x, data2_y, train_size=0.8, random_state=123)

    # 20% Test; 80% Train Set

    def ModelLogic(train_x: pd.DataFrame, train_y: pd.DataFrame):  # LogicRegression
        x = train_x.values;
        y = train_y.values
        model = LogisticRegression()  # Build LogicRegression Model y = 1 / (1 + exp ** (-x))
        model.fit(x, y)  # Train Model
        return model

    def ModelSVC(train_x: pd.DataFrame, train_y: pd.DataFrame):  # SVC
        x = train_x.values;
        y = train_y.values
        model = SVC(C=2, kernel='sigmoid', probability=True)  # probability must be True
        model.fit(x, y)
        return model

    def ModelForest(train_x: pd.DataFrame, train_y: pd.DataFrame):  # RandomForest
        x = train_x.values;
        y = train_y.values
        model = RandomForestClassifier(max_depth=6, n_estimators=200, random_state=5)
        model.fit(x, y)
        return model

    def ModelTest(model, test_x, test_y):  # Define model checking function
        pred = list(model.predict(test_x))  # Predict test set data
        pd_rl = pd.DataFrame({'pred': pred, 'true': test_y.values})
        try:
            TP = pd.crosstab(pd_rl.true, pd_rl.pred)[1][
                1]  # True Positive example: the actual value is 1, the predicted value is also 1
        except:
            TP = 0
        try:
            FP = pd.crosstab(pd_rl.true, pd_rl.pred)[1][
                0]  # False positive example: actual value is 0, predicted value is 1
        except:
            FP = 0
        try:
            TN = pd.crosstab(pd_rl.true, pd_rl.pred)[0][
                0]  # True negative example: the actual value is 0, and the predicted value is also 0
        except:
            TN = 0
        try:
            FN = pd.crosstab(pd_rl.true, pd_rl.pred)[0][
                1]  # False Negative example: the actual value is 1, the predicted value is 0
        except:
            FN = 0
        # Model Accuracy、Precision、Sensitivity、Recall/Specificity、F1 value、cohen's kappa
        test_dict = {}
        # print('Accuracy:'+ accuracy_score(pd_rl.true, pd_rl.pred))
        test_dict.update({'Accuracy': model.score(test_x, test_y)})  # Model evaluation, accuracy
        try:
            test_dict.update({'Precision': TP / (TP + FP)})  # Model evaluation, accuracy rate
        except:
            test_dict.update({'Precision': 0})  # Model evaluation, accuracy rate
        try:
            # Model evaluation, sensitivity = number of true positives TP/(number of true positives TP + number of false negativesFN) * 100%.
            # Correctly judge the rate of patients;
            test_dict.update({'Sensitivity': TP / (TP + FN)})
        except:
            test_dict.update({'Sensitivity': 0})  # Model evaluation, recall/sensitivity
        # Specificity = number of true negatives TN/(number of true negatives TN + number of false positives FP))*100%.
        # Correctly judge the rate of non-patients
        test_dict.update({'Specificity': TN / (TN + FP)})
        test_dict.update({'F1': (2 * test_dict['Sensitivity'] * test_dict['Precision']) / (
                    test_dict['Sensitivity'] + test_dict['Precision'])})  # Model evaluation, F1 value
        # https://www.cnblogs.com/duoba/p/13344877.html
        p0 = (TP + TN) / (TP + FP + TN + FN)
        pe = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / (TP + FP + TN + FN) ** 2
        test_dict.update({"cohen's kappa": (p0 - pe) / (1 - pe)})  # Model evaluation, cohen's kappa coefficient value

        return test_dict

    modellog1 = ModelLogic(train1_x, train1_y)  # Build a logistic regression model
    modelsvc1 = ModelSVC(train1_x, train1_y)  # Build SVC Model1
    modelforest1 = ModelForest(train1_x, train1_y)  # Build a random forest model1
    log1 = ModelTest(modellog1, test1_x, test1_y)
    svc1 = ModelTest(modelsvc1, test1_x, test1_y)
    forest1 = ModelTest(modelforest1, test1_x, test1_y)

    modellog2 = ModelLogic(train2_x, train2_y)  # Build a logistic regression model
    modelsvc2 = ModelSVC(train2_x, train2_y)  # Build SVC Model1
    modelforest2 = ModelForest(train2_x, train2_y)  # Build a random forest model1
    log2 = ModelTest(modellog2, test2_x, test2_y)
    svc2 = ModelTest(modelsvc2, test2_x, test2_y)
    forest2 = ModelTest(modelforest2, test2_x, test2_y)

    # predict probabilities
    # pred_prob0 = modellinear1.predict_proba(test1_x)
    pred_prob1 = modellog1.predict_proba(test1_x)
    pred_prob2 = modelsvc1.predict_proba(test1_x)
    pred_prob3 = modelforest1.predict_proba(test1_x)

    pred_prob4 = modellog2.predict_proba(test2_x)
    pred_prob5 = modelsvc2.predict_proba(test2_x)
    pred_prob6 = modelforest2.predict_proba(test2_x)
    # pred_prob7 = modellinear2.predict_proba(test2_x)
    # roc curve for models
    # fpr0, tpr0, thresh0 = roc_curve(test1_y, pred_prob1[:,1], pos_label=1)
    fpr1, tpr1, thresh1 = roc_curve(test1_y, pred_prob1[:, 1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(test1_y, pred_prob2[:, 1], pos_label=1)
    fpr3, tpr3, thresh3 = roc_curve(test1_y, pred_prob3[:, 1], pos_label=1)
    fpr4, tpr4, thresh4 = roc_curve(test2_y, pred_prob4[:, 1], pos_label=1)
    fpr5, tpr5, thresh5 = roc_curve(test2_y, pred_prob5[:, 1], pos_label=1)
    fpr6, tpr6, thresh6 = roc_curve(test2_y, pred_prob6[:, 1], pos_label=1)
    # fpr7, tpr7, thresh7 = roc_curve(test2_y, pred_prob7[:,1], pos_label=1)
    # auc scores
    # auc_score0 = roc_auc_score(test0_y, pred_prob0[:,1])
    auc_score1 = roc_auc_score(test1_y, pred_prob1[:, 1])
    auc_score2 = roc_auc_score(test1_y, pred_prob2[:, 1])
    auc_score3 = roc_auc_score(test1_y, pred_prob3[:, 1])
    auc_score4 = roc_auc_score(test2_y, pred_prob4[:, 1])
    auc_score5 = roc_auc_score(test2_y, pred_prob5[:, 1])
    auc_score6 = roc_auc_score(test2_y, pred_prob6[:, 1])
    Roc_data = {"log_no": [fpr1, tpr1, auc_score1], "rft_no": [fpr2, tpr2, auc_score2],
                "svc_no": [fpr3, tpr3, auc_score3],
                "log": [fpr4, tpr4, auc_score4], "rft": [fpr5, tpr5, auc_score5], "svc": [fpr6, tpr6, auc_score6]}
    test_dict = {i: [log1[i], log2[i], svc1[i], svc2[i], forest1[i], forest2[i]] for i in log1.keys()}
    test_df = pd.DataFrame.from_dict(test_dict, orient='index')
    test_df.columns = [' LogicRisk', ' LogicNoRisk', ' SVCRisk', ' SVCNoRisk', ' ForestRisk', ' ForestNoRisk']

    name_list = ['LogicRegression(No Risk) (area = %0.2f)' % auc_score1,
                 'RandomForest(No Risk) (area = %0.2f)' % auc_score2,
                 'SVC(No Risk) (area = %0.2f)' % auc_score3,
                 'LogicRegression(Risk) (area = %0.2f)' % auc_score4,
                 'RandomForest(Risk)  (area = %0.2f)' % auc_score5,
                 'SVC(Risk)  (area = %0.2f)' % auc_score6]

    x_list = [fpr1,fpr2,fpr3,fpr4,fpr5,fpr6]
    y_list = [tpr1,tpr2,tpr3,tpr4,tpr5,tpr6]
    z_list = [auc_score1, auc_score2, auc_score3, auc_score4, auc_score5, auc_score6]

    max_roc = [[x,y] for x,y in zip(z_list,name_list)]
    max_roc.sort(key=lambda x:x[0])
    text = "Recommendation model : " + max_roc[-1][1]

    return Roc_data, test_df, x_list, y_list, name_list, text
    #return Roc_data, test_df


def get_show_KMeans(kmeans):
    #sctx = ['2019/1/1', '2019/1/2', '2019/1/3', '2019/1/4', '2019/1/5']
    #scty = [3607, 3834, 3904, 4080, 3997]
    trace_list = []
    kxn = "Mother's age when born"
    kyn = "Weight at birth, pounds"
    for i in range(5):
        trace = go.Scatter(
            x=list(kmeans[kxn][kmeans['Type'] == i]),
            y=list(kmeans[kyn][kmeans['Type'] == i]),
            mode='markers',
            name=str(i),)
        trace_list.append(trace)


    layout=go.Layout(
        yaxis={
            'hoverformat': '' #如果想显示小数点后两位'.2f'，显示百分比'.2%'
        },

    )

    fig = go.Figure(
        data = trace_list,
        layout = layout
    )
    fig.update_layout(
        # autosize=False,
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
            children='Clustering Analysis',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='Analysis Results: This shows the data distribution and data can be devided into 5 groups, majority group is type 4.',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])

    row3 = html.Div([
        dcc.Graph(
            #id='cluster5',
            figure=fig
        )
    ])

    return [row1,row2,row3]






def get_auc_plot(name_list, x_list, y_list, text):
    fig = go.Figure()
    '''
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    '''

    for i in range(len(name_list)):
        fig.add_trace(go.Scatter(x=x_list[i], y=y_list[i], name=name_list[i], mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        #yaxis=dict(scaleanchor="x", scaleratio=1),
        #xaxis=dict(constrain='domain'),
    )



    fig.update_layout(
        # autosize=False,
        paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(legend_x=1, legend_y=0)
    #fig.update_layout(legend_x=0, legend_y=0)
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
            children=text,
            style={
                'textAlign': 'center',
                'color': colors['text']
            }),
    ])

    row3 = html.Div([
        dcc.Graph(
            id='ML',
            figure=fig
        )
    ])

    return [row1, row2, row3]



def get_result_plot(data):
    fig = go.Figure()
    '''
    for i in range(1, data.shape[0]):
        fig.add_trace(go.Bar(
            x=data.columns[1:],
            y=data.iloc[i, 1:],
            name=data.columns[i],
            marker_color='indianred'
        ))
    '''
    import plotly.express as px
    color_list = px.colors.sequential.Plasma
    for i in range(len(data.columns)):
        y = data.iloc[:, i]
        my = np.array(y).max()
        if my>1:
            y = [k/my for k in y]
        fig.add_trace(go.Bar(
            x=data.axes[0],
            y=y,
            name=data.columns[i],
            marker_color=color_list[i%len(color_list)]
        ))


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
            children='Comapre ML Model',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
    ])

    row2 = html.Div([
        html.Div(
            children='This figure shows all the machine learning parameters in comparison',
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

def get_data2(file_name):
    #file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = data1
    kmeans = KMeans_data(data2)
    kxn = "Mother's age when born"
    kyn = "Weight at birth, pounds"

    roc, test_df, x_list, y_list, name_list,text = Model_data(data2)

    return get_show_KMeans(kmeans), get_auc_plot(name_list, x_list, y_list,text), get_result_plot(test_df)



if __name__=="__main__":

    file_name = r"ECQ_D.csv"
    data1 = clean_data(file_name)
    data2 = ETL(data1)
    kmeans = KMeans_data(data2)
    kxn = "Mother's age when born"
    kyn = "Weight at birth, pounds"

    roc, test_df, x_list, y_list, name_list = Model_data(data2)


    app.layout = html.Div([
        dcc.Graph(
            id='get_show_KMeans',
            figure= get_show_KMeans(kmeans)
        ),
        dcc.Graph(
            id='show_heatmap',
            figure=get_auc_plot(name_list, x_list, y_list)
        ),

        dcc.Graph(
            id='show_dis',
            figure= get_result_plot(test_df)
        ),


    ], style={'margin': 100})
    app.run_server(debug=True)



