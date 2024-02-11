# Usa la imagen oficial de Python
FROM python:3.7

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el código actual al contenedor en /app
COPY . /app

# Instala las dependencias necesarias
RUN pip install --no-cache-dir joblib pandas fastapi uvicorn numpy pydantic scikit-learn

# Expone el puerto 8000 para que sea accesible desde el exterior
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Nota:
# docker build --no-cache --progress=plain -t profiles-classification .
# docker run --name container-jobs-profiles -d -p 8000:8000 profiles-classification
# docker run -it --name container-clsfr -p 8000:8000 classifier-prof bash