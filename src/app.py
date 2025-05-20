from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils import load_model, InputData

app = FastAPI()
templates = Jinja2Templates(directory="templates")
model = load_model()

@app.get('/', response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse('form.html', {'request': request})

@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request,
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

    data = InputData(
        Age=Age, Sex=Sex, ChestPainType=ChestPainType, RestingBP=RestingBP,
        Cholesterol=Cholesterol, FastingBS=FastingBS, RestingECG=RestingECG,
        MaxHR=MaxHR, ExerciseAngina=ExerciseAngina, Oldpeak=Oldpeak, ST_Slope=ST_Slope
    )
    df = data.to_dataframe()
    pred = model.predict(df)[0]
    return templates.TemplateResponse('form.html', {'request': request, 'result': f"Prediction: {pred}"})