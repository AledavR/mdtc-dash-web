import dash
from dash import Dash, html, Input, Output, callback, dcc
from utils import modelo_sir


dash.register_page(
    __name__,
    path='/modelos/sir-model',
    name='Modelo SIR'
)

layout = html.Div(className='pages', children = [
    html.H2('Modelo epidemiologico SIR', className='page_title'),
    html.Div(className='page_box',children=[
        html.Div(className='div_parametros',children = [
            html.H2('PARÁMETROS',className='title'),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Tamaño de poblacion',className='subtitle'),
                    dcc.Input(type='number', value=150,min=100, id='pop_size'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Infectados iniciales',className='subtitle'),
                    dcc.Input(type='number', value=1,min=1, id='inf_0'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Recuperados iniciales',className='subtitle'),
                    dcc.Input(type='number', value=0,min=0, id='rec_0'),
                ], className='input-right-margin'),
            ]),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Ratio de infección',className='subtitle'),
                    dcc.Input(type='number', value=0.05, step=0.001,min=0.001, id='beta'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Ratio de recuperación',className='subtitle'),
                    dcc.Input(type='number', value=0.01, step=0.001,min=0.001, id='gamma'),
                ], className='input-right-margin'),
            ]),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Tiempo',className='subtitle'),
                    dcc.Input(type='number', value=1000,step=50,min=50, id='time'),
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
                html.Div([dcc.Graph(id='figure_sir',style={'width': '100%', 'height': '700px'})]),
            ])
        ]),
    ])
])

@callback(
    Output('figure_sir', 'figure'),
    Input('pop_size', 'value'),
    Input('inf_0', 'value'),
    Input('rec_0', 'value'),
    Input('beta', 'value'),
    Input('gamma', 'value'),
    Input('time', 'value'),
    Input('n', 'value'),
)

def grafica_sir(N,I0,R0,beta,gamma,tiempo_total,n):
    fig = modelo_sir(N,I0,R0,beta,gamma,tiempo_total,n)
    return fig
