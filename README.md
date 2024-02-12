# ClasificadorPerfilesDeTrabajo

Este proyecto consiste en un clasificador de perfiles de trabajo construido con FastAPI, que utiliza las siguientes bibliotecas: joblib, pandas, fastapi, uvicorn, numpy, pydantic, scikit-learn y starlette.
Crear entorno virtual e instalar dependencias
## Requerimientos
- Python3(v3.11.2)
- Node JS(v12.14.0)
- Angular(V12.2.16)

## Run Server Back-End

1. Crea un entorno virtual:
```
   git clone https://github.com/JoseLeviRivera/ClasificadorPerfilesDeTrabajo.git
```
2. Crea un entorno virtual:
```
 python -m venv venv
```
3.  Activa el entorno virtual. 
En Windows:
```
venv\Scripts\activate
```
En Mac o Linux:
```
source venv/bin/activate
```
4. Instala las dependencias:
```
pip install joblib pandas fastapi uvicorn numpy pydantic scikit-learn
```
o tambien podría ser:
```
pip install -r requirements.txt
```
5. Puedes correr:
```
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
o también 
```
python3 app.py
```
## Run Server Font-End
1. Instalar dependencias
```
 npm install
```
3. Correr servidor web 
```
 npm start
```
## Run con Docker y Docker Compose
Dockerfile para el Backend
```
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

```

Dockerfile para el Frontend

```
# Usa la imagen oficial de Node.js slim (Alpine) como base
FROM node:12.14.0-alpine

# Establece el directorio de trabajo en la carpeta de la aplicación Angular
WORKDIR /usr/src/app

# Copia los archivos necesarios para instalar las dependencias
COPY package*.json ./

# Instala las dependencias
RUN npm install

# Copia el resto de los archivos de la aplicación
COPY . .

# Exponer el puerto 4200 (puerto por defecto de desarrollo de Angular)
EXPOSE 4200

# Inicia el servidor HTTP incorporado en Node.js para servir la aplicación
CMD ["npm", "start"]

```
Correr con dockerCompose
```
version: '3'

services:
  backend:
    image: joseleviriv/app-profiles-classification:latest
    ports:
      - "8000:8000"

  frontend:
    image: joseleviriv/web-profiles-classification-app:0.0.1
    network_mode: "host"

```
Para desplegar ambos servicios con Docker Compose, ejecuta el siguiente comando en el directorio donde se encuentra el archivo docker-compose.yml:
```
  docker-compose up -d

```
Para detener los servicios con Docker Compose:
```
  docker-compose donw

```
## Información
![Clasificador Perfiles de trabajo](https://clasificadorperfilestrabajo.web.app/)
![Imagen de clasificador Backend ](https://hub.docker.com/repository/docker/joseleviriv/app-profiles-classification/general)
![Imagen de clasificador Frontend](https://hub.docker.com/repository/docker/joseleviriv/web-profiles-classification-app/general)


