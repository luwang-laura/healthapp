from inspect import stack
import pandas as pd
import plotly.express as px

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)
server = app.server

#---------------------------------------------------------------
#Taken from https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases
df = pd.read_excel("mddr.xlsx")

dff = df.groupby("Login", as_index=False)[['Normalized Handle Resolves','Total EPT*','Total EPT (No Outliers)','IPH W/O Outliers','IPH Total','SOE Adoption']].sum()
print (dff[:5])
#---------------------------------------------------------------
app.layout = html.Div([
    html.Div([
        dash_table.DataTable(
            id='login_id',
            data=dff.to_dict('records'),
            columns=[
                {"name": i, "id": i, "deletable": False, "selectable": False} for i in dff.columns
            ],
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            row_selectable="multi",
            row_deletable=False,
            selected_rows=[],
            page_action="native",
            page_current= 0,
            page_size= 6,
            style_cell_conditional=[
                {'if': {'column_id': 'Login'},
                 'width': '5%', 'textAlign': 'left'},
                {'if': {'column_id': 'Normalized Handle Resolves'},
                 'width': '10%', 'textAlign': 'left'},
                {'if': {'column_id': 'Total EPT*'},
                 'width': '5%', 'textAlign': 'left'},
                {'if': {'column_id': 'Total EPT (No Outliers)'},
                 'width': '5%', 'textAlign': 'left'},
                {'if': {'column_id': 'IPH W/O Outliers'},
                 'width': '5%', 'textAlign': 'left'},
                {'if': {'column_id': 'IPH Total'},
                 'width': '5%', 'textAlign': 'left'},
                {'if': {'column_id': 'SOE Adoption'},
                 'width': '5%', 'textAlign': 'left'},
            ],
        ),
    ],className='row'),

    html.Div([
        html.Div([
             html.Div([
            dcc.Dropdown(id='bardropdown',
                options=[
                        {'label': 'Resolves', 'value': 'Resolves'},
                        {'label': 'Normalized Handle Resolves', 'value': 'Normalized Handle Resolves'}  
                ],
                value='Resolves',
                multi=True,
                #barmode=stack,
                clearable=False
            ),
        ],className='six columns'),
        #---------
         html.Div([
            dcc.Dropdown(id='bardropdown2',
                options=[
                         {'label': 'Total EPT*', 'value': 'Total EPT*'},
                         {'label': 'IPH Total', 'value': 'IPH Total'},
                         {'label': 'Total EPT (No Outliers)', 'value': 'Total EPT (No Outliers)'} 
                ],
                value='Total EPT*',
                multi=True,
                #barmode=stack,
                clearable=False
            ),
        ],className='six columns'),
        ],className='row2'),
        #--------Pie Dropdown------------
        html.Div([
              html.Div([
        dcc.Dropdown(id='piedropdown',
            options=[
                {'label': 'IPH W/O Outliers', 'value': 'IPH W/O Outliers'},
                {'label': 'IPH Total', 'value': 'IPH Total'},
                {'label': 'SOE Adoption', 'value': 'SOE Adoption'}
                ],
            value='SOE Adoption',
            multi=False,
            clearable=False
        ),
        ],className='pie columns'),
         
        #-----
        dcc.Dropdown(id='piedropdown2',
            options=[
                {'label': 'IPH W/O Outliers', 'value': 'IPH W/O Outliers'},
                {'label': 'IPH Total', 'value': 'IPH Total'},
                {'label': 'SOE Adoption', 'value': 'SOE Adoption'}
                ],
            value='SOE Adoption',
            multi=False,
            clearable=False
        ),
        ],className='pie columns'),
        #---

    ],className='row3'),

    html.Div([
        html.Div([
            dcc.Graph(id='barchart'),
        ],className='six columns'),
        html.Div([
            dcc.Graph(id='barchart2'),
        ],className='six columns'),
        ],className='row2'),

        html.Div([
            html.Div([
            dcc.Graph(id='piechart'),
        ],className='pie columns'),

            html.Div([
            dcc.Graph(id='emptychart'),
        ],className='six columns'),

        ],className='row3'),
])

#------------------------------------------------------------------
@app.callback(
    [Output('piechart', 'figure'),
     Output('barchart', 'figure'),
     Output('barchart2', 'figure')],
    [Input('login_id', 'selected_rows'),
     Input('piedropdown', 'value'),
     Input('bardropdown', 'value'),
     Input('bardropdown2', 'value')]
)
def update_data(chosen_rows,piedropval,bardropval,bardropval2):
    if len(chosen_rows)==0:
        df_filterd = dff[dff['Login'].isin(['ammarieg','begonp','briandjb','dongkexi','hzhengx','limzy','llormari','luwangm','tozhao'])]
    else:
        print(chosen_rows)
        df_filterd = dff[dff.index.isin(chosen_rows)]
    #----------------Pie Chart ----------
    pie_chosen_loginid=df_filterd['Login'].tolist()
    df_pie = df[df['Login'].isin(pie_chosen_loginid)]

    pie_chart=px.pie(
            data_frame=df_pie,
            names='Login',
            values=piedropval,
            hole=.3,
            labels={'Login':'Login'}
            )
    pie_chart.update_layout(uirevision='foo')
    #----------------Bar Chart 1----------
    list_chosen_loginid=df_filterd['Login'].tolist()
    df_bar = df[df['Login'].isin(list_chosen_loginid)]

    bar_chart = px.bar(
            data_frame=df_bar,
            x=bardropval,
            #y=bardropval,
            color='Login',
            labels={'SOE Adoption':'SOE Adoption', 'Normalized Handle Resolves':'Normalized Handle Resolves'},
            )
    bar_chart.update_layout(uirevision='foo')
 #----------------Bar Chart 2----------
     #extract list of chosen countries
    chosen_loginid=df_filterd['Login'].tolist()
    #filter original df according to chosen countries
    #because original df has all the complete dates
    df_bar2 = df[df['Login'].isin(chosen_loginid)]

    bar_chart2 = px.bar(
            data_frame=df_bar2,
            x=bardropval2,
            #y=bardropval,
            color='Login',
            labels={'SOE Adoption':'SOE Adoption', 'Normalized Handle Resolves':'Normalized Handle Resolves'},
            )
    bar_chart2.update_layout(uirevision='foo')

    return (pie_chart,bar_chart,bar_chart2)

#------------------------------------------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)