import dash
from dash import Dash, html, Input, Output, callback, dcc
from utils import modelo_lv_comp_intra

dash.register_page(
    __name__,
    path='/',
    # path='/modelos/lotka-volterra-comp',
    name='Modelo Lotka-Volterra - Competicion Intraespecifica'
)

 # Layout HTML

layout = html.Div(className='pages', children = [
    html.H2('Modelo Depredador-Presa (Lotka-Volterra)', className='page_title'),
    html.Div(className='page_box', children = [
        html.Div(className='div_parametros', children = [
            html.H2('PARÁMETROS',className='title'),
            html.H3('Poblaciones iniciales',className='input-label'),
            html.Div(className='div_flex', children = [
                html.Div([
                    html.H3('Presas',className='input-label'),
                    dcc.Input(type='number', value=0.4, id='X0', max=1.0, min=0.0, step=0.1),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Depr. Intermedio',className='input-label'),
                    dcc.Input(type='number', value=0.6, id='Y0', max=1.0, min=0.0, step=0.1),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('Depr. Superior', className='input-label'),
                    dcc.Input(type='number', value=0.8, id='Z0', max=1.0, min=0.0, step=0.1),
                ], className='input-right-margin'),
            ]),
            html.H3('Tiempo',className='input-label'),
            html.Div(className='div_flex', children = [
                # html.Div([
                #     html.H3('Tiempo inicial',className='input-label'),
                #     dcc.Input(type='number', value=0, id='t_i'),
                # ], className='input-right-margin'),
                html.Div([
                    dcc.Input(type='number', value=100, id='t',max=200,min=10),
                ], className='input-right-margin'),
            ]),
            html.H3('Tasas de Mortalidad',className='input-label'),
            html.Div(className='div_flex', children = [
                html.Div([
                    html.H3('d1', className='input-label'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.05, id='d1'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('d2', className='input-label'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.09, id='d2'),
                ], className='input-right-margin'),
            ]),
            html.H3('Parametros de Interacción - Depredador Intermedio',className='input-label'),
            html.Div(className='div_flex', children = [
                html.Div([
                    html.H3('a21'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.3, id='a21'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('a22'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.15, id='a22'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('a23'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.09, id='a23'),
                ], className='input-right-margin'),
            ]),
            html.H3('Parametros de Interacción - Depredador Superior',className='input-label'),
            html.Div(className='div_flex', children = [
                html.Div([
                    html.H3('a31'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.18, id='a31'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('a32'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.1, id='a32'),
                ], className='input-right-margin'),
                html.Div([
                    html.H3('a33'),
                    dcc.Input(max=1, min=0, step=0.01, type='number', value=0.08, id='a33'),
                ], className='input-right-margin'),
            ]),
            
            html.H3('Numero de muestras'),
            dcc.Slider(min=30, max=80, step=5, value=65, marks=None, tooltip={'placement':'bottom','always_visible':True}, id='cant'),
            
    ]),


        html.Div(className='div_grafica', children = [
            html.H2('GRÁFICA',className='title'),
            html.Div(className='grafica', children = [
                # dcc.Loading(
                # type='dot',
                # children=
                    dcc.Graph(id='figure_lv_cmp',style={'width': '100%', 'height': '700px'})
                    # ),
            ])                        
            
    ]),
    ]),
])

# Callbacks

@callback(
    Output('figure_lv_cmp', 'figure'),
    Input('X0', 'value'),
    Input('Y0', 'value'),
    Input('Z0', 'value'),
    Input('a21', 'value'),
    Input('a22', 'value'),
    Input('a23', 'value'),
    Input('a31', 'value'),
    Input('a32', 'value'),
    Input('a33', 'value'),
    Input('d1', 'value'),
    Input('d2', 'value'),
    Input('t', 'value'),
    Input('cant', 'value'),
)

def grafica_lv(X0,Y0,Z0,a21,a22,a23,a31,a32,a33,d1,d2,t,cant):
    fig = modelo_lv_comp_intra(X0,Y0,Z0,a21,a22,a23,a31,a32,a33,d1,d2,t,cant)
    return fig
