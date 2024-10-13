"""
Alejandro Ramirez
2024
"""

from dash import Dash, html, dcc
import dash

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

app.layout = html.Div(children=[
    html.Div(className='header', children = [
            html.Img(className = 'sm_logo', src='assets/sm_logo.png'),
            html.H1('Modelos Poblacionales - Alejandro Ramirez Chero', className = 'main')
    ]),
    html.Div(className='contenedor_navegacion', children =[
        dcc.Link(html.Button('Modelo Logistico', className='boton edo_1'), href='/'),
        dcc.Link(html.Button('Modelo Logistico - Sympy', className='boton edo_2'), href='/log-sym'),
        dcc.Link(html.Button('Crecimiento con Umbral', className='boton edo_2'), href='/umbral'),
        dcc.Link(html.Button('Modelo Lotka-Volterra', className='boton edo_2'), href='/lv')
    ]),
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True,port=8051)
