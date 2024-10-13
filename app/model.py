# pip install pydub
# pip install moviepy
# !apt-get install sox libsndfile1 ffmpeg
# !pip install matplotlib>=3.3.2

# pip install --upgrade pip
# pip install --upgrade setuptools

# pip install wheel ninja
     # pip install xformers --no-dependencies
# pip install xformers==0.0.20

# !python -m pip install git+https://github.com/NVIDIA/NeMo.git@1fa961ba03ab5f8c91b278640e29807079373372#egg=nemo_toolkit[asr] ## not all
# !python -m pip install pyannote.audio==3.2.0

# pip uninstall huggingface_hub
# pip install huggingface-hub==0.22

__all__ = ['GigaAMRNNTModel', 'GigaAMCoder']

from settings import SOURCE_FOLDER, RESULT_FOLDER, AUDIO_FOLDER, DECODED_AUDIO_FOLDER

import os
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import os
import locale

from io import BytesIO
from typing import List, Tuple

import numpy as np
from pyannote.audio import Pipeline

from dotenv import load_dotenv, get_key, set_key

import torch
import torchaudio
from nemo.collections.asr.models import EncDecRNNTBPEModel
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessor as NeMoAudioToMelSpectrogramPreprocessor,
)
from nemo.collections.asr.parts.preprocessing.features import (
    FilterbankFeaturesTA as NeMoFilterbankFeaturesTA,
)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

HF_TOKEN = get_key(dotenv_path, 'HF_TOKEN')

locale.getpreferredencoding = lambda: "UTF-8"

device = "cuda" if torch.cuda.is_available() else "cpu"


class FilterbankFeaturesTA(NeMoFilterbankFeaturesTA):
    def __init__(self, mel_scale: str = "htk", wkwargs=None, **kwargs):
        if "window_size" in kwargs:
            del kwargs["window_size"]
        if "window_stride" in kwargs:
            del kwargs["window_stride"]

        super().__init__(**kwargs)

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=kwargs["nfilt"],
                window_fn=self.torch_windows[kwargs["window"]],
                mel_scale=mel_scale,
                norm=kwargs["mel_norm"],
                n_fft=kwargs["n_fft"],
                f_max=kwargs.get("highfreq", None),
                f_min=kwargs.get("lowfreq", 0),
                wkwargs=wkwargs,
            )
        )


class AudioToMelSpectrogramPreprocessor(NeMoAudioToMelSpectrogramPreprocessor):
    def __init__(self, mel_scale: str = "htk", **kwargs):
        super().__init__(**kwargs)
        kwargs["nfilt"] = kwargs["features"]
        del kwargs["features"]
        self.featurizer = (
            FilterbankFeaturesTA(  # Deprecated arguments; kept for config compatibility
                mel_scale=mel_scale,
                **kwargs,
            )
        )


class GigaAMCoder:

    def __init__(self):
        self.source_file_name = ''
        self._audio_file_name = ''
        self._audio_mono_file_name = ''

        self.result_file_name = ''

    def decode_source_file(self, source_file_name):

        self.source_file_name = source_file_name

        self._audio_file_name = self._get_audio_file_name(self.source_file_name)
        self._video_to_audio(self.source_file_name, self._audio_file_name)
        self._audio_mono_file_name = self._get_mono_file_name(self._audio_file_name)
        self._stereo_to_mono16(self._audio_file_name, self._audio_mono_file_name)
        path = os.path.join(AUDIO_FOLDER, self._audio_file_name)
        os.remove(path)

        self.result_file_name = self._audio_mono_file_name

    def _get_audio_file_name(self, video_file_name):
        basename, ext = os.path.splitext(video_file_name)
        return '{}_.mp3'.format(basename)

    def _get_mono_file_name(self, audio_file_name):
        basename, ext = os.path.splitext(audio_file_name)
        return '{}_mono16.wav'.format(basename)

    def _stereo_to_mono16(self, audio_file_stereo, audio_file_mono):

        audio_path_stereo = os.path.join(AUDIO_FOLDER, audio_file_stereo)
        audio_path_mono = os.path.join(DECODED_AUDIO_FOLDER, audio_file_mono)
        sound = AudioSegment.from_mp3(audio_path_stereo)
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        print(sound.frame_rate)
        sound.export(audio_path_mono, format='wav', )

    def _video_to_audio(self, video_file_name, audio_file_name):
        """Converts video to audio using MoviePy library
        that uses `ffmpeg` under the hood"""

        video_path = os.path.join(SOURCE_FOLDER, video_file_name)
        audio_path = os.path.join(AUDIO_FOLDER, audio_file_name)

        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)


class GigaAMRNNTModel:

    def __init__(self, coder: GigaAMCoder):
        self.coder: GigaAMCoder = coder

        self.source_file_name = ''
        self.decoded_source_file_name = ''
        self.result_file_name = ''
        self._segments_batch = 10
        self._batch_size = 2

        self.progress_writer = None

        self._transcribing_model = self._get_transcribing_model()

    def process(self, source_file_name, progress_writer=None):

        if progress_writer:
            self.progress_writer = progress_writer

        self.source_file_name = source_file_name

        self.coder.decode_source_file(source_file_name)

        self.decoded_source_file_name = self.coder.result_file_name
        result_data = self._process_transcribing()

        self.delete_decoded_source_file()

        return result_data

    def _process_transcribing(self):
        
        pipeline = self._get_vad_pipline()
        path = os.path.join(DECODED_AUDIO_FOLDER, self.decoded_source_file_name)
        segments, boundaries = self._segment_audio(path, pipeline)

        transcriptions = []

        seg_len =self._segments_batch

        seg_qty = len(segments)/seg_len
        seg_qty = int(seg_qty) if seg_qty%1 ==0 else int(seg_qty) + 1

        progress = 0
        for ind in range(seg_qty):
            i = ind*seg_len
            ip1 = i+seg_len
            if ip1 >= len(segments):
                ip1 = len(segments)

            c_segments = segments[i:ip1]

            model = self._get_transcribing_model()

            c_transcriptions = model.transcribe(c_segments, batch_size=self._batch_size)
            progress = int((ind+1)*100/seg_qty) 

            print('-'*10 + ' Transcribing progress - {}%'.format(progress))

            if self.progress_writer:
                self.progress_writer.progress = progress
                            
            transcriptions.extend(c_transcriptions[0])

        return transcriptions, boundaries

    def _get_transcribing_model(self):
        model = EncDecRNNTBPEModel.from_config_file("../models_data/models--gigaam--rnnt/rnnt_model_config.yaml")
        ckpt = torch.load("../models_data/models--gigaam--rnnt/rnnt_model_weights.ckpt", map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        model.eval()
        model = model.to(device)

        return model

    def _audiosegment_to_numpy(self, audiosegment: AudioSegment) -> np.ndarray:
        """Convert AudioSegment to numpy array."""
        samples = np.array(audiosegment.get_array_of_samples())
        if audiosegment.channels == 2:
            samples = samples.reshape((-1, 2))

        samples = samples.astype(np.float32, order="C") / 32768.0
        return samples

    def _segment_audio(self, audio_path: str, 
                      pipeline: Pipeline, 
                      max_duration: float = 22.0, 
                      min_duration: float = 15.0, 
                      new_chunk_threshold: float = 0.2,) -> Tuple[List[np.ndarray], List[List[float]]]:
        # Prepare audio for pyannote vad pipeline
        audio = AudioSegment.from_wav(audio_path)
        audio_bytes = BytesIO()
        audio.export(audio_bytes, format="wav")
        audio_bytes.seek(0)

        # Process audio with pipeline to obtain segments with speech activity
        sad_segments = pipeline({"uri": "filename", "audio": audio_bytes})

        segments = []
        curr_duration = 0
        curr_start = 0
        curr_end = 0
        boundaries = []

        # Concat segments from pipeline into chunks for asr according to max/min duration
        for segment in sad_segments.get_timeline().support():
            start = max(0, segment.start)
            end = min(len(audio) / 1000, segment.end)
            if (
                curr_duration > min_duration and start - curr_end > new_chunk_threshold
            ) or (curr_duration + (end - curr_end) > max_duration):
                audio_segment = self._audiosegment_to_numpy(
                    audio[curr_start * 1000 : curr_end * 1000]
                )
                segments.append(audio_segment)
                boundaries.append([curr_start, curr_end])
                curr_start = start

            curr_end = end
            curr_duration = curr_end - curr_start

        if curr_duration != 0:
            audio_segment = self._audiosegment_to_numpy(
                audio[curr_start * 1000 : curr_end * 1000]
            )
            segments.append(audio_segment)
            boundaries.append([curr_start, curr_end])

        return segments, boundaries

    def _get_vad_pipline(self):
        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection", cache_dir='../models_data', use_auth_token=HF_TOKEN)
        pipeline = pipeline.to(torch.device(device))

        return pipeline

    def write_result_file(self, result_data, result_file_name):

        path = os.path.join(RESULT_FOLDER, result_file_name)

        transcriptions, boundaries = result_data
        with open(path, 'w', encoding='utf-8') as fp:
            for transcription, boundary in zip(transcriptions, boundaries):
                boundary_0 = self._format_time(boundary[0])
                boundary_1 = self._format_time(boundary[1])
                f_str = f"[{boundary_0} - {boundary_1}]: {transcription}\n"

                fp.writelines([f_str])

    def delete_result_file(self):
        path = os.path.join(RESULT_FOLDER, self.result_file_name)
        os.remove(path)
        
    def delete_decoded_source_file(self):
        path = os.path.join(DECODED_AUDIO_FOLDER, self.decoded_source_file_name)
        os.remove(path)

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        full_seconds = int(seconds)
        milliseconds = int((seconds - full_seconds) * 100)

        if hours > 0:
            return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
        else:
            return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"
