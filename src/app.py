from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI() 

templates = Jinja2Templates(directory="templates")

model = joblib.load("models/final_model.pkl")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def handle_form(request: Request,
                Age: int = Form(...),
                Sex: str = Form(...),
                ChestPainType: str = Form(...),
                RestingBP: int = Form(...),
                Cholesterol: int = Form(...),
                FastingBS: int = Form(...),
                RestingECG: str = Form(...),
                MaxHR: int = Form(...),
                ExerciseAngina: str = Form(...),
                Oldpeak: float = Form(...),
                ST_Slope: str = Form(...)):
    
    data = {
        "Age": Age,
        "Sex": Sex,
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }
    try:
        input_df = pd.DataFrame([data])
        proba = model.predict_proba(input_df)[:, 1][0]
        result = f"Вероятность сердечного заболевания: {proba:.2%}"
    except Exception as e:
        result = f"Ошибка: {e}"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
