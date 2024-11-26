from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
import torch

from model import model, inference

app = FastAPI()

IMAGES_DIR = os.path.join(os.getcwd(), 'imgs')

# Determine the environment via an environment variable
environment = os.getenv('ENV_TYPE', 'local')

if environment == 'docker':
    CHKPT_DIR = os.path.join('best_model_checkpoint.pth')
    DEVICE = 'cpu'
else:
    CHKPT_DIR = os.path.join('..', 'best_model_checkpoint.pth')
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model.load_state_dict(torch.load(CHKPT_DIR, weights_only=True, map_location=torch.device(DEVICE)))

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    dest = os.path.join(IMAGES_DIR,file.filename)

    try:
        contents = file.file.read()
        with open(dest, 'wb') as f:
            f.write(contents)

    except Exception as e:
        return {"message": f"There was an error uploading the file:\n{e}"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename} and saved to {dest}."}


@app.get("/predict/")
def predict(filename: str):
    prediction = inference(model, os.path.join(IMAGES_DIR, filename))
    return {"label": prediction}


if __name__ == "__main__":
    uvicorn.run("api:app", port=8000, host='0.0.0.0')