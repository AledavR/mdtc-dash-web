import dash
from dash import Dash, html, Input, Output, callback, dcc
from utils import ecuacion_logistica_sympy

dash.register_page(
    __name__,
    path='/modelos/logistico-sympy',
    name='Modelo Logistico (sympy)'
)

 # Layout HTML

layout = html.Div(
    className='pages', children = [
        html.H2('Modelo Logistico - Sympy', className='page_title'),
        html.Div(className='page_box', children = [
            html.Div(className='div_parametros', children = [
                html.H2('PARÁMETROS',className='title'),
                html.Div(className='div_flex', children =[
                    html.Div([
                        html.H3('Población Inicial'),
                        dcc.Input(type='number', value=10, id='pob_ini'),
                    ], className='input-right-margin'),
                    html.Div([
                        html.H3('Tiempo inicial'),
                        dcc.Input(type='number', value=0, id='time_ini'),
                    ], className='input-right-margin'),
                    html.Div([
                        html.H3('Tiempo final'),
                        dcc.Input(type='number', value=60, id='time_fin'),
                    ], className='input-right-margin'),
                ]),
                html.Div(className='div_flex', children = [
                    html.Div([
                        html.H3('Tasa de crecimiento'),
                        dcc.Input(max=5, min=0, step=0.1, type='number', value=0.1, id='r'),
                    ], className='input-right-margin'),
                    html.Div([
                        html.H3('Capacidad de carga'),
                        dcc.Input(type='number',value=150, id='K'),
                    ], className='input-right-margin'),
                ]),
                html.H3('Malla para el Campo de Vectores'),
                dcc.Slider(min=1, max=40, step=1, value=15, marks=None,
                           tooltip={'placement':'bottom','always_visible':True},
                           id='mallado',),
                html.H3('Tamaño del Vector'),
                dcc.Slider(min=0.1, max=2, step=0.1, value=1, id='size_vec'),
                html.H3('Desactivar Mallado'),
                dcc.RadioItems(
                    ['Activado', 'Desactivado'],
                    'Activado',
                    id='no_arrows',
                    labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                )
                
    ]),
            html.Div(className='div_grafica', children = [
                html.H2('GRÁFICA',className='title'),
                html.Div(className='grafica', children = [
                    dcc.Loading(
                        type='dot',
                        children=dcc.Graph(
                            id='figure_log_sympy',
                            style={'width': '100%', 'height': '700px'})
                    ),
                ])
            ]),
        ]),
])

# Callbacks
@callback(
    Output('figure_log_sympy', 'figure'),
    Input('pob_ini', 'value'),
    Input('time_ini', 'value'),
    Input('time_fin', 'value'),
    Input('r', 'value'),
    Input('K', 'value'),
    Input('mallado', 'value'),
    Input('size_vec', 'value'),
    Input('no_arrows', 'value'),
)

def grafica_log_sympy(P0,t_i,t_f,r,k,mallado,size_vec,no_arrows):
    fig = ecuacion_logistica_sympy(k, P0, r, t_i, t_f,mallado, size_vec,no_arrows)
    return fig
