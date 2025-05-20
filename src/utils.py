import joblib
from pydantic import BaseModel
import pandas as pd

MODEL_PATH = "models/model.pkl"

def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)

def load_model(path=MODEL_PATH):
    return joblib.load(path)

class InputData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

    def to_dataframe(self):
        return pd.DataFrame([self.dict()])