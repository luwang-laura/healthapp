import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import datetime
import io

#--------------------------------------------------------------
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash
from dash.dependencies import Input, Output, State
import pandas as pd
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
import dash_bootstrap_components as dbc
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
from load_data0 import get_data0
from load_data1 import get_data1
from load_data2 import  get_data2
from load_data3 import  get_data3
from load_data4 import  get_data4
#a,b,c = get_data0()
global file_name
file_name = None
global button_id_all
button_id_all = None
#--------------------------------------------
'''
   
    #html.Hr(),
    html.Button('Statistical Approach', id='StatisticalApproach', n_clicks=0),
    #html.Hr(),
    html.Button('Machine Learning Models', id='MachineLearningModels', n_clicks=0),
    #html.Hr(),
    html.Button('Compare Average', id='CompareAverage', n_clicks=0),
    #html.Hr(),
    html.Button('Select Graphics', id='SelectGraphics', n_clicks=0),]
'''

select1 = dbc.Select(
    id="Xvariable",
    options=[
        {"label": "Option 1", "value": "1"},
        {"label": "Option 2", "value": "2"},
        {"label": "Disabled option", "value": "3", "disabled": True},
    ],
)

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    assert 'csv' in filename
    global file_name
    file_name = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    #a,b,c = get_data1(file_name)
    #return get_init(a, b, c)
    return file_name

@app.callback(
            #Output(component_id='Xvariable', component_property='options'),
            Output(component_id='b6', component_property='children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(a, b, c):
    options = [{"label": "Option 1", "value": "1"}, ]
    print(a, b, c)
    if a is None:
        return 'None'
    if b is not None:
        file_name = parse_contents(a,b,c)
        options = []
        print(file_name)
        name_list = ["Mother's age when born", "Mother smoked when pregnant",
        "Receive newborn care at health facility", "Weight at birth, pounds",
        "Doctor confirmed overweight", "How do you consider weight"]
        for name in name_list:
            options.append({"label": name, "value": name})
        return b[-15:]

    return "None"


up_file = dcc.Upload( id='upload-data',children=dbc.Button('Upload File',color="primary", block=True))
Button1 = dbc.Button("Statistical Approach", id='b1', color="primary", block=True)
Button2 = dbc.Button("Machine Learning Models", id='b2', color="secondary", block=True)
Button3 = dbc.Button("Compare average", id='b3', color="success", block=True)


#Button0 = dbc.Button("Select Graphics", id='b4', color="primary", block=True)

#Button4 = dbc.Button("X variable", color="secondary", block=True)
Button5 = dbc.Button("", id='b6', color="success", block=True)

#file_name = r"ECQ_D.csv"
#a,b,c = get_data1(file_name)
#a,b,c = get_data2()
#a,b,c = get_data3()
#a,b,c = get_data4()
global ga,gaa,gb,gc
ga = [html.Hr(),html.Hr(),html.Hr()]
gaa = [html.Hr(),html.Hr(),html.Hr()]
gb = [html.Hr(),html.Hr(),html.Hr()]
gc = [html.Hr(),html.Hr(),html.Hr()]
a,b,c = ga,gb,gc
colors = {
    'background': '#111111',
    'text': '#111111'
}

import dash_core_components as dcc

dx1 = dcc.Dropdown(
    options=[
        {'label': 'How do you consider weight', 'value': 'How do you consider weight'},
        {'label': 'Mother smoked when pregnant', 'value': 'Mother smoked when pregnant'},
    ],
    placeholder="Select x varible",
    id = "Dropdownx"
)

dy1 = dcc.Dropdown(
    options=[
        {'label': 'Weight at birth, pounds', 'value': 'Weight at birth, pounds'},
        {'label': 'Doctor confirmed overweight', 'value': 'Doctor confirmed overweight'},
    ],
    placeholder="Select y varible",
    id = "Dropdowny"
)


dz1 = dcc.Dropdown(
    options=[
        {'label': '2 or more', 'value': '2 or more'},
        {'label': 'Only 2', 'value': 'Only 2'},
    ],
    placeholder="Select group varible",
    id = "Dropdownz"
)


leftButton = html.Div(children=[


            html.H1(
                children='Interactive Health Analytic dashboard',
                style={
                    'textAlign': 'center',
                    'color': colors['text']
                }
            ),
        dbc.Row([

                dbc.Col(
                    html.Div([up_file]),
                    #width=2
                ),
                dbc.Col(Button5,
                    #width=1
                    ),


                dbc.Col(Button1,
                        #width=1,
                        ),

                dbc.Col(Button2,
                        #width=1,
                    #width={"size": "auto"},
                    ),

                dbc.Col(Button3,
                        #width=1,
                    #idth={"size": "auto"},
                    ),
               #dbc.Col(Button0,
                    #width=1,
                    # idth={"size": "auto"},
               #     ),

        ],
            no_gutters=True,
            ),


        dbc.Row([

                dbc.Col(
                    dx1
                ),
                dbc.Col(dy1,
                    #width=1
                    ),

                dbc.Col(dz1,
                        #width=1,
                        ),
        ],
            no_gutters=True,
            ),


        dbc.Row([
            dbc.Col(html.Div(id='11', children=[a[0]]),
                #width={"size": "auto"},
            ),
            dbc.Col(html.Div(id='12', children=[b[0]]),
                #width={"size": "auto"},
                    ),
            dbc.Col(html.Div(id='13', children=[b[0]]),
                #width={"size": "auto"},
                ),
        ]),



        dbc.Row([
            dbc.Col(html.Div(id='31', children=[a[2]]),
                    # width={"size": "auto"},
                    ),
            dbc.Col(html.Div(id='32', children=[b[2]]),
                    # width={"size": "auto"},
                    ),
            dbc.Col(html.Div(id='33', children=[c[2]]),
                    # width={"size": "auto"},
                    ),
        ]),
     dbc.Row([
        dbc.Col(html.Div(id='21', children=[a[1]]),
                # width={"size": "auto"},
                ),
        dbc.Col(html.Div(id='22', children=[b[1]]),
                # width={"size": "auto"},
                ),
        dbc.Col(html.Div(id='23', children=[c[1]]),
                # width={"size": "auto"},
                ),
    ]),


    ])




@app.callback(
    Output(component_id='11', component_property='children'),
    Output(component_id='12', component_property='children'),
    Output(component_id='13', component_property='children'),
    Output(component_id='21', component_property='children'),
    Output(component_id='22', component_property='children'),
    Output(component_id='23', component_property='children'),
    Output(component_id='31', component_property='children'),
    Output(component_id='32', component_property='children'),
    Output(component_id='33', component_property='children'),
     Input('b1', 'n_clicks'),
     Input('b2', 'n_clicks'),
     Input('b3', 'n_clicks'),
     Input('Dropdownx', "value"),
     Input('Dropdowny', "value"),
     Input('Dropdownz', "value"),
)
     #Input('Xvariable', 'value'))
def display(btn1, btn2, btn3, dx, dy, dz):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(button_id)
    global button_id_all

    global ga, gb, gc
    a, b, c = ga, gb, gc
    global file_name
    if file_name is not None:
        if button_id  in ('b1','b2','b3','b4'):
            button_id_all = button_id

            if button_id=='b1':
                a, b, c = get_data1(file_name, dx, dy, dz)
            if button_id == 'b2':
                a, b, c = get_data2(file_name)
            if button_id == 'b3':
                a, b, c = get_data3(file_name)
        else:
            a, b, c = get_data1(file_name, dx, dy, dz)

    return a[0],b[0],c[0],a[1],b[1],c[1],a[2],b[2],c[2]




app.layout = dbc.Container([leftButton], 'mian', fluid=True)

if __name__ == '__main__':
    #app.run_server(host='0.0.0.0')
    app.run_server(debug=True)