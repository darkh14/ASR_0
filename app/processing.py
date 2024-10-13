import uuid
import os
import json
from multiprocessing import Process

from settings import SOURCE_FOLDER, RESULT_FOLDER
from model import GigaAMCoder, GigaAMRNNTModel


class TaskStatusController:
    
    def __init__(self, task_id):
        self.task_id = task_id

        self._status = ''
        self._progress = 0
        self._error = ''

    def set_task_data(self, status=None, progress=None, error=None):

        filename = 'task_{}.json'.format(self.task_id)
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as fp:
                task_data = json.load(fp)
        else:
            task_data = self._new_task_data()

        if status is not None:
            self._status = status
            task_data['status'] = status

        if progress is not None:
            self._progress = progress
            task_data['progress'] = progress

        if progress is not None:
            self._progress = progress
            task_data['progress'] = progress       

        with open(filename, 'w', encoding='utf-8') as fp:
            json.dump(task_data, fp)

    def set_status(self, status):
        self.set_task_data(status=status)

    def set_progress(self, progress):
        self.set_task_data(progress=progress)

    def set_error(self, error):
        self.set_task_data(error=error)   

    @staticmethod
    def _new_task_data():
        return {'status': '', 'progress': 0, 'error': ''}
    
    def read_task_data(self):
        
        filename = 'task_{}.json'.format(self.task_id)
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as fp:
                task_data = json.load(fp)

            self._status = task_data['status']
            self._progress = task_data['progress']
            self._error = task_data['error']
        else:
            self._status = ''
            self.progress = 0
            self.error = ''

    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, status):
        self.set_task_data(status=status)

    @property
    def progress(self):
        return self._progress
    
    @progress.setter
    def progress(self, progress):
        self.set_task_data(progress=progress)              

    @property
    def error(self):
        return self._error
    
    @error.setter
    def error(self, error):
        self.set_task_data(error=error)        
    

class Task:

    def __init__(self, model, task_id='', backgound_part=False):

        self.model = model
        self.task_id = task_id or str(uuid.uuid4())
        self.source_file_ext = ''
        self.result_file_ext = ''

        self.source_file_name = ''
        self.result_file_name = ''

        self.result_data = None
        self.backgound_part = backgound_part

        self.status_controller = TaskStatusController(self.task_id)     

    def process(self, fp, background=False):

        assert not self.backgound_part, 'Call process in background part is forbidden' 

        self.form_sorce_file_name_ext(fp.filename)
        self._write_source_file(fp)

        if background:
            self.process_model_in_background()
        else:
            self._process_model()

            self._write_result_file()

            self.delete_source_file()

        return self.task_id
    
    def _write_source_file(self, fp):
        
        path = os.path.join(SOURCE_FOLDER, self.source_file_name)

        with open(path, 'wb') as wfp:
            wfp.write(fp.file.read())

    def form_sorce_file_name_ext(self, external_filename):

        self.source_file_ext = external_filename.split('.')[-1]
        self.source_file_name = '{}_source.{}'.format(self.task_id, self.source_file_ext)

    def process_model_in_background(self):
        self.status_controller.set_task_data(status='TASK_STARTED', progress=0, error='') 
        prc = Process(target=process_task_background_part, args=(self.task_id, self.source_file_name))

        prc.start()

    def process_model_background_part(self, external_filename):
        assert self.backgound_part, 'Call "process_model_background_part: allowed only in background part'
        self.status_controller.set_task_data(status='TRANSCRIBING_STARTED', progress=0)

        self.form_sorce_file_name_ext(external_filename)
        self._process_model(background_part=True)
        self._write_result_file()
        self.status_controller.set_task_data(status='READY', progress=100)

    def _write_result_file(self):
        
        if not self.result_file_ext:
            self.result_file_ext = 'txt'

        self.result_file_name = '{}_result.{}'.format(self.task_id, self.result_file_ext)

        self.model.write_result_file(self.result_data, self.result_file_name)

    def delete_source_file(self):
        path = os.path.join(SOURCE_FOLDER, self.source_file_name)
        os.remove(path)
    
    def delete_result_file(self):
        path = os.path.join(RESULT_FOLDER, self.result_file_name)         
        os.remove(path)

    def _process_model(self, background_part=False):
        self.result_data = self.model.process(self.source_file_name, progress_writer = self.status_controller)

    @classmethod
    def get_status(cls, task_id):
        status_controller = TaskStatusController(task_id)
        status_controller.read_task_data()
        return {'status': status_controller.status,
                'progress': status_controller.progress,
                'error': status_controller.error}
    
    @classmethod
    def get_result_file_name(cls, task_id):
        
        file_list = list(os.listdir(RESULT_FOLDER))

        result = ''

        for file_name in file_list:
            file_path = os.path.join(RESULT_FOLDER, file_name)
            print(task_id, file_name, file_path, os.path.isfile(file_path), file_name.split('_')[0] == task_id)
            if os.path.isfile(file_path) and file_name.split('_')[0] == task_id:
                result = file_name
                break

        return result


def process_task_background_part(task_id, external_filename):

    coder = GigaAMCoder()
    model = GigaAMRNNTModel(coder)
    task = Task(model, task_id=task_id, backgound_part=True)

    task.process_model_background_part(external_filename)