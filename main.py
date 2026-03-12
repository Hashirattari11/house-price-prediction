from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from predictor import predict_price

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": ""})

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    bedrooms: float = Form(...),
    bathrooms: float = Form(...),
    floors: float = Form(...),
    sqft_living: float = Form(...)
):
    result = predict_price(bedrooms, bathrooms, floors, sqft_living)
    return templates.TemplateResponse("index.html", {"request": request, "prediction_text": result})
