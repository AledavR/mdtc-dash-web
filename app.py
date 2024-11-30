"""
Alejandro Ramirez
2024
"""

from dash import Dash, html, dcc
import dash

external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Poiret+One&display=swap',
]

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.Div(
        className='header', children = [
            html.Img(className = 'sm_logo', src=dash.get_asset_url('sm_logo.png')),
            html.H1('Tecnicas de Modelamiento 2024', className = 'main'),
            html.Div(),
        ]
    ),
    html.Div(
        className='contenedor_navegacion', children = [
            html.Div(
                className='subnav',
                children = [
                    dcc.Link(html.Button('Modelos', className='boton menu'), href='/modelos/logistico-numpy'),
                    html.Div(
                        className='subnav-content',
                        children = [
                            dcc.Link(html.Button('Modelo Logistico', className='boton item'), href='/modelos/logistico-numpy'),
                            dcc.Link(html.Button('Modelo Logistico - Sympy', className='boton item'), href='/modelos/logistico-sympy'),
                            dcc.Link(html.Button('Crecimiento con Umbral', className='boton item'), href='/modelos/umbral'),
                            dcc.Link(html.Button('Modelo Lotka-Volterra', className='boton item'), href='/modelos/lotka-volterra'),
                            dcc.Link(html.Button('Modelo Lotka-Volterra - Comp. Intraespecie', className='boton item'), href='/'),
                            dcc.Link(html.Button('Modelo Lotka-Volterra - Comp. Intraespecie - 3D', className='boton item'), href='/modelos/lotka-volterra-comp-3d'),
                            dcc.Link(html.Button('Modelo SIR', className='boton item'), href='/modelos/sir-model'),
                            dcc.Link(html.Button('Modelo SIR', className='boton item'), href='/modelos/sir-model-rumores'),
                            dcc.Link(html.Button('Modelo Lotka-Volterra 3D', className='boton item'), href='/modelos/lotka-volterra-comp-3d'),
                        ])
                ]
            ),
            html.Div(
                className='subnav',
                children = [
                    dcc.Link(html.Button('Analisis', className='boton menu'), href='/analisis/edo-system'),
                    html.Div(
                        className='subnav-content',
                        children = [
                            dcc.Link(html.Button('Sistema de EDOs', className='boton item'), href='/analisis/ode-system'),
                        ])
                ]
            ),
        ]
    ),
    dash.page_container,
    "Autor: Alejandro Ramirez Ch."
])



if __name__ == '__main__':
    app.run(debug=True,port=8051)
