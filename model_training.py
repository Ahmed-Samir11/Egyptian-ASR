import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
import os
import tarfile
import pandas as pd
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from mltu.preprocessors import WavReader
from mltu.tensorflow.dataProvider import DataProvider
from mltu.transformers import LabelIndexer, LabelPadding, SpectrogramPadding
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CERMetric, WERMetric
from model import build_model
from configs import ModelConfigs

dataset_path = "/content/drive/MyDrive/AIC"
metadata_path = dataset_path + "/train_batch.csv"
wavs_path = dataset_path + "/train_batch/"
metadata_df = pd.read_csv(metadata_path, quoting=2)
metadata_df = metadata_df.dropna(subset=['transcript'])
print(metadata_df.info())
dataset = [[f"/content/drive/MyDrive/AIC/train_batch/{file}.wav", label] for file, label in metadata_df.values.tolist()]

configs = ModelConfigs()
# If the training is done through multiple steps, that is chunking data and retraining the model, use this loop once with full dataframe and then comment it
max_text_length, max_spectrogram_length = 0, 0
for file_path, label in tqdm(dataset):
    spectrogram = WavReader.get_spectrogram(file_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
    valid_label = [c for c in label if c in configs.vocab]
    max_text_length = max(max_text_length, len(valid_label))
    max_spectrogram_length = max(max_spectrogram_length, spectrogram.shape[0])
    configs.input_shape = [max_spectrogram_length, spectrogram.shape[1]]

configs.max_spectrogram_length = max_spectrogram_length
configs.max_text_length = max_text_length
configs.save()

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
        ],
    transformers=[
        SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# Manually split the dataset instead of using deepcopy
split_index = int(len(dataset) * 0.9)
train_dataset = dataset[:split_index]
val_dataset = dataset[split_index:]

train_data_provider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
        ],
    transformers=[
        SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

val_data_provider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        WavReader(frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length),
        ],
    transformers=[
        SpectrogramPadding(max_spectrogram_length=configs.max_spectrogram_length, padding_value=0),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
        ],
)

# Build the model
optimizer = tf.keras.optimizers.Adam(learning_rate=configs.learning_rate)
model = build_model(
    input_dim = configs.input_shape,
    output_dim = len(configs.vocab),
    dropout=0.5
)
model.compile(
    optimizer=optimizer, 
    loss=CTCloss(), 
    metrics=[
        CERMetric(vocabulary=configs.vocab),
        WERMetric(vocabulary=configs.vocab)
    ],
    run_eagerly=False
)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode="min")
trainLogger = TrainLogger(configs.model_path)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.8, min_delta=1e-10, patience=5, verbose=1, mode="auto")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Restore from checkpoint if available
start_epoch = 0
if checkpoint_manager.latest_checkpoint:
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    start_epoch = int(checkpoint_manager.latest_checkpoint.split('-')[-1].split('.')[0])
    print(f"Resuming training from checkpoint: {checkpoint_manager.latest_checkpoint}")

model.summary(line_length=110)

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
    workers=configs.train_workers
)

# Save model, and training and validation datasets as csv files
model.save('fully_trained_model')
train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))
