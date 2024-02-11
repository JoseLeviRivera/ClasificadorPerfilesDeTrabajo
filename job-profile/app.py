import uvicorn
import subprocess
import warnings
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from models.Profile import Profile

subprocess.run(['python', "train_model.py"])

# Fast Api
app = FastAPI()


# Ignorar advertencias relacionadas con los nombres de las características en Scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")



# Configurar el middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

profile_model = joblib.load(open('regression/model.pkl', 'rb'))

@app.get("/")
async def get_info():
    return {"response": "Hello World! "}

@app.get("/info")
async def get_info():
    return {"app": "job-profile", "version": "1.0", "status": "running"}

@app.get("/reports")
async def get_report():
    try:
        # Intenta abrir y leer el contenido del archivo JSON
        with open( "reports/model_results.json", "r") as file:
            report_content = file.read()
    except FileNotFoundError:
        # Manejar el caso en que el archivo no se encuentre
        raise HTTPException(status_code=404, detail="Report not found")
    # Devuelve el contenido del archivo JSON como respuesta
    return JSONResponse(content=report_content, media_type="application/json")


@app.get("/cross_validations")
async def get_cross_validations():
    try:
        # Intenta abrir y leer el contenido del archivo JSON
        with open( "reports/cross_validations.json", "r") as file:
            report_content = file.read()
    except FileNotFoundError:
        # Manejar el caso en que el archivo no se encuentre
        raise HTTPException(status_code=404, detail="Report not found")
    # Devuelve el contenido del archivo JSON como respuesta
    return JSONResponse(content=report_content, media_type="application/json")

@app.post("/classify_profile")
def classify_profile(profile_input: Profile):
    # Convert input data to a NumPy array
    input_data = np.array([
        [profile_input.DSA, profile_input.DBMS, profile_input.OS, profile_input.CN,
         profile_input.Mathmetics, profile_input.Aptitude, profile_input.Comm,
         profile_input.Problem_Solving, profile_input.Creative, profile_input.Hackathons,
         profile_input.Skill_1, profile_input.Skill_2]
    ])

    # Hacer la predicción
    prediction = profile_model.predict(input_data)

    # Convertir el resultado a una lista antes de devolverlo
    prediction_list = prediction.tolist()

    # Devolver la respuesta
    return {"predicted_profile": prediction_list}

if __name__ == '__main__':
    try:
        # Iniciar el servidor FastAPI
        server_port = 8000
        uvicorn.run("app:app", host="localhost", port=server_port, reload=True)
    except KeyboardInterrupt:
        # Manejar Ctrl+C para detener el servidor
        pass
