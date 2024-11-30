import dash
from dash import Dash, html, Input, Output, callback, dcc
from utils import modelo_sir_rumor


dash.register_page(
    __name__,
    path='/modelos/sir-model-rumores',
    name='Modelo SIR - Para rumores'
)

layout = html.Div(className='pages', children = [
    html.H2('Modelo epidemiologico SIR', className='page_title'),
    html.Div(className='page_box',children=[
        html.Div(className='div_parametros',children = [
            html.H2('PARÁMETROS',className='title'),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Tamaño de poblacion',className='subtitle'),
                    dcc.Input(type='number', value=250,min=5,step=5, id='pop_size'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Infectados iniciales',className='subtitle'),
                    dcc.Input(type='number', value=1,min=1, id='inf_0'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Recuperados iniciales',className='subtitle'),
                    dcc.Input(type='number', value=6,min=0, id='rec_0'),
                ], className='input-right-margin'),
            ]),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Ratio de infección',className='subtitle'),
                    dcc.Input(type='number', value=0.004, step=0.0001,min=0.0001, id='beta'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Ratio de recuperación',className='subtitle'),
                    dcc.Input(type='number', value=0.01, step=0.0001,min=0.0001, id='gamma'),
                ], className='input-right-margin'),
            ]),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Tiempo',className='subtitle'),
                    dcc.Input(type='number', value=25,step=5,min=5, id='time'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Precisión',className='subtitle'),
                    dcc.Input(type='number',min=1, value=100, id='n'),
                ], className='input-right-margin'),
            ]),

    ]),
        html.Div(className='div_grafica',children = [
            html.H2('GRÁFICA',className='title'),
            html.Div(className='grafica', children = [
                html.Div([dcc.Graph(id='figure_sir_rumor',style={'width': '100%', 'height': '700px'})]),
            ])
        ]),
    ])
])

@callback(
    Output('figure_sir_rumor', 'figure'),
    Input('pop_size', 'value'),
    Input('inf_0', 'value'),
    Input('rec_0', 'value'),
    Input('beta', 'value'),
    Input('gamma', 'value'),
    Input('time', 'value'),
    Input('n', 'value'),
)

def grafica_sir(N,I0,R0,beta,gamma,tiempo_total,n):
    fig = modelo_sir_rumor(N,I0,R0,beta,gamma,tiempo_total,n)
    return fig
