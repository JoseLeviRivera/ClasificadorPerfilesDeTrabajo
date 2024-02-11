from sklearn.feature_selection import SequentialFeatureSelector

""""
Esta función utiliza el Sequential Feature Selector (SFS) para seleccionar un número específico
de características (n) utilizando el modelo y los datos proporcionados. El parámetro direction 
determina si se seleccionarán ('forward') o eliminarán ('backward') características. La función devuelve
un array booleano indicando qué características han sido seleccionadas.
Parámetros:
    - modelo (modelo de scikit-learn): El modelo utilizado para evaluar la importancia de las características.
    - X (similar a matriz): La matriz de características.
    - y (similar a matriz): Las etiquetas de destino.
    - n (int): Número de características a seleccionar.
    - dir (str): Dirección de la selección, puede ser 'forward' para selección hacia adelante o 'backward' para eliminación hacia atrás.
Resultados:
    Un array booleano indicando qué características han sido seleccionadas (True) y cuáles no (False).
"""


def generate_features_selector(modelo, X, y, n, dir):
    sfs = SequentialFeatureSelector(modelo, n_features_to_select=n, direction=dir)
    sfs.fit(X, y)
    return sfs.get_support()
