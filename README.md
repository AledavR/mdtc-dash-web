# Dash Web - MDTC

This repository contains a dash web page implementation of different mathematical models.

## Current models

+ Logarithmic growth function
+ Exponential growth with threshold
+ Lotka-Volterra

## Setup

To run the dash web page follow these steps:

1. Clone this repository

2. Visit the repository directory through console

3. Create a python virtual environment 

```python -m venv dash-web-venv```

4. Source the virtual environment

On windows: 
```.\venv\dash-web-venv\Scripts\activate ```
On Linux:
```source dash-web-venv/bin/activate```

5. Install dependencies:

```pip install numpy scipy dash sympy```

6. Run the server:

```python app.py```


## Contribution

- When pushing a commit be sure to be descriptive with your changes. 
- Make sure that your code works before pushing.
- When creating your `venv` use any name that ends in venv e.g. `dash-venv` or `.venv`
