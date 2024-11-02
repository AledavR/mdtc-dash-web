import dash
from dash import Dash, html, Input, Output, callback, dcc
from utils import ecuacion_lv

dash.register_page(
    __name__,
    path='/modelos/lotka-volterra',
    name='Modelo Depredador-Presa'
)

 # Layout HTML

layout = html.Div(className='pages', children = [
    html.H2('Modelo Depredador-Presa (Lotka-Volterra)', className='page_title'),
    html.Div(className='page_box', children = [
        html.Div(className='div_parametros', children = [
            html.H2('PARÁMETROS',className='title'),
            html.Div([
                html.H3('Población Inicial Depredadores'),
                dcc.Input(type='number', value=12, id='D0'),
            ]),
            html.Div([
                html.H3('Población Inicial Presas'),
                dcc.Input(type='number', value=56, id='P0'),
            ]),
            html.Div([
                html.H3('Tiempo inicial'),
                dcc.Input(type='number', value=0, id='t_i'),
            ]),
            html.Div([
                html.H3('Tiempo final'),
                dcc.Input(type='number', value=50, id='t'),
            ]),
            html.H3('Tasa de nacimiento - Presas'),
            dcc.Input(max=2, min=0, step=0.1, type='number', value=1.0, id='a'),
            html.H3('Tasa de depredación'),
            dcc.Input(max=2, min=0, step=0.1, type='number', value=0.2, id='b'),
            html.H3('Tasa de nacimiento - Depredadores'),
            dcc.Input(max=2, min=0, step=0.1, type='number', value=1.0, id='c'),
            html.H3('Tasa de mortalidad de depredadores'),
            dcc.Input(max=2, min=0, step=0.1, type='number', value=0.1, id='d'),
            html.H3('Numero de muestras'),
            dcc.Slider(min=30, max=80, step=5, value=65, marks=None, tooltip={'placement':'bottom','always_visible':True}, id='mallado'),
            
    ]),
        html.Div(className='div_grafica', children = [
            html.H2('GRÁFICA',className='title'),
            html.Div(className='grafica', children = [
                dcc.Loading(
                    type='dot',
                    children=dcc.Graph(id='figure_lv',style={'width': '100%', 'height': '700px'})
                ),
            ])                        
            
    ]),
    ]),
])

# Callbacks

@callback(
    Output('figure_lv', 'figure'),
    Input('a', 'value'),
    Input('b', 'value'),
    Input('c', 'value'),
    Input('d', 'value'),
    Input('P0', 'value'),
    Input('D0', 'value'),
    Input('t', 'value'),
    Input('t_i', 'value'),
    Input('mallado', 'value'),
)

def grafica_lv(a,b,c,d,P0,D0,t,t_i,mallado):
    fig = ecuacion_lv(a,b,c,d,P0,D0,t,t_i,mallado)
    return fig
