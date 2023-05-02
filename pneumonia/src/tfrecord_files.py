import os
import tensorflow as tf
import numpy as np


class TFRecordReader:
    # Set up paths and filenames
    RAW_DATA_DIR = "../data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    FILE_PATTERN = "*.jpeg"
    NUM_SHARDS = 10

    FEATURE_SPEC = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
    }

    def __init__(self):
        # Get a list of all raw data files in the data/raw directory
        self.raw_data_files = tf.io.gfile.glob(os.path.join(self.RAW_DATA_DIR, self.FILE_PATTERN))

    # Define a function to preprocess the raw data
    def preprocess_data(self, raw_data):
        # Perform any necessary data preprocessing steps here
        preprocessed_data = raw_data * 2  # Example preprocessing step
        return preprocessed_data

    # Define a function to write a set of preprocessed data to a single TFRecord file
    def write_tfrecord_file(self, preprocessed_data, filename):
        # Open the TFRecord file for writing
        with tf.io.TFRecordWriter(filename) as writer:
            # Loop over the preprocessed data and serialize each record as a TFExample
            for i, record in enumerate(preprocessed_data):
                example = tf.train.Example(features=tf.train.Features(feature={
                    "image": tf.train.Feature(float_list=tf.train.FloatList(value=[record[0]])),
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[record[1]])),
                    "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[record[2]])),
                }))
                # Write the serialized record to the file
                writer.write(example.SerializeToString())
            print(f"{len(preprocessed_data)} records written to {filename}")

    def read_data(self):
        for raw_data_file in self.raw_data_files:
            # Read the raw data from the file
            raw_data = np.loadtxt(raw_data_file)
            # Preprocess the raw data
            preprocessed_data = self.preprocess_data(raw_data)
            # Split the preprocessed data into NUM_SHARDS chunks and write each chunk to a separate TFRecord file
            shard_size = len(preprocessed_data) // self.NUM_SHARDS
            for shard_idx in range(self.NUM_SHARDS):
                start_idx = shard_idx * shard_size
                end_idx = (shard_idx + 1) * shard_size
                shard_data = preprocessed_data[start_idx:end_idx]
                filename = os.path.join(self.PROCESSED_DATA_DIR,
                                        f"{os.path.basename(raw_data_file)}.shard{shard_idx+1}.tfrecord")
                # Write the shard of preprocessed data to a TFRecord file
                self.write_tfrecord_file(shard_data, filename)

    def zip_folder(folder_path, zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    zipf.write(os.path.join(root, file))
        print(f"Folder {folder_path} compressed into {zip_path}")