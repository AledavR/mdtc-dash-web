import dash
from dash import Dash, html, Input, Output, callback, dcc
from utils import analisis_ode


dash.register_page(
    __name__,
    path='/analisis/ode-system',
    name='Sistema de EDOs'
)

layout = html.Div(className='pages', children = [
    html.H2('Sistema de EDOs', className='page_title'),
    html.Div(className='page_box',children=[
        html.Div(className='div_parametros',children = [
            html.H2('PARÁMETROS',className='title'),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Primera ecuacion',className='subtitle'),
                    dcc.Input(type='text', value='x*(5-y)', id='eq1'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Segunda ecuacion',className='subtitle'),
                    dcc.Input(type='text', value='y*(5-x)', id='eq2'),
                ], className='input-right-margin'),
            ]),
            html.Div(className='div_flex', children =[
                html.Div([
                    html.H3('Valor minimo',className='subtitle'),
                    dcc.Input(type='number', value=0, id='min'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Valor maximo',className='subtitle'),
                    dcc.Input(type='number', value=0, id='max'),
                ], className='input-right-margin'),
            ]),
            html.Button('Calcular puntos de equilibrio', id='calcular', n_clicks=0),
    ]),
        html.Div(className='div_grafica',children = [
            html.H2('RESULTADOS',className='html'),
            html.Div(id='puntos_criticos'),
            dcc.Markdown(id='matriz_jacobiana', mathjax=True),
            html.Div(id='analisis_puntos'),
            html.Div(className='grafica', children = [
                dcc.Loading(
                    type='dot',
                    children=dcc.Graph(id='figure_ode',style={'width': '100%', 'height': '700px'})
                ),
            ])
        ]),
    ])
])

@callback(
    Output('puntos_criticos', 'children'),
    Output('matriz_jacobiana', 'children'),
    Output('analisis_puntos', 'children'),
    Output('figure_ode', 'figure'),
    Input('eq1', 'value'),
    Input('eq2', 'value'),
    Input('min', 'value'),
    Input('max', 'value'),
    Input('calcular', 'n_clicks'),
)

def analisis_ode_page(eq1,eq2,min,max,n_clicks):
    puntos_criticos,matriz_jacobiana,analisis_puntos,figure_ode = analisis_ode(eq1,eq2,min,max,n_clicks)
    return puntos_criticos,matriz_jacobiana,analisis_puntos,figure_ode
