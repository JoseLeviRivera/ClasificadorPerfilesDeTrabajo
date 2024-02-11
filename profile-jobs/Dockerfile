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

#Nota:
# docker build --no-cache --progress=plain -t web-profiles-classification . 
# docker run --network="host" --name web-container-jobs-profiles -d -p 4200:4200 web-profiles-classification
