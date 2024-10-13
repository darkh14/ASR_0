from typing import Annotated
import os

from processing import Task
from model import GigaAMRNNTModel, GigaAMCoder

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

import uvicorn

from settings import RESULT_FOLDER

app = FastAPI()

@app.post("/transcribe_and_get")
async def transcribe_file_sync(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload file sent"}

    coder = GigaAMCoder()
    model = GigaAMRNNTModel(coder)
    task = Task(model)

    task.process(file)
    basename, _ = os.path.splitext(file.filename)
    
    path = os.path.join(RESULT_FOLDER, task.result_file_name)
    result = FileResponse(path=path, filename='{}.txt'.format(basename), media_type='multipart/form-data')

    return result

@app.post("/start_transcribing")
async def start_transcribing(file: UploadFile | None = None) -> str:
    
    if not file:
        return {"message": "No upload file sent"}

    coder = GigaAMCoder()
    model = GigaAMRNNTModel(coder)
    task = Task(model)

    task_id = task.process(file, background=True)

    return task_id

@app.get("/get_status/")
async def get_status(task_id: str) -> dict:

    result = Task.get_status(task_id)

    return result

@app.post("/get_file/")
async def get_file(task_id: str):
    
    status = Task.get_status(task_id)['status']

    assert status == 'READY', 'Transcribing by task "{}" is not ready'.format(task_id)

    file_name = Task.get_result_file_name(task_id)
    
    path = os.path.join(RESULT_FOLDER, file_name)
    result = FileResponse(path=path, filename='1.txt', media_type='multipart/form-data')

    return result


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8070, log_level="info")