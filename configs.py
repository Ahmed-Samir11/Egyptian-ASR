import os
from datetime import datetime

from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join("/content/drive/MyDrive/AIC/202406210246", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.frame_length = 256 
        self.frame_step = 160
        self.fft_length = 512

        self.vocab = "ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْ؟' "
        self.input_shape = [1765, 257]
        self.max_text_length = 257
        self.max_spectrogram_length = 1765

        self.batch_size = 32
        self.learning_rate = 0.0005
        self.train_epochs = 1000
        self.train_workers = 20
