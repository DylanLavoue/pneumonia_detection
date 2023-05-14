import tensorflow as tf
import numpy as np
import os


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

class TFRecordReader:
    def __init__(self, file_pattern, feature_spec):
        """
        Initializes the TFRecordReader class.

        Args:
            file_pattern (str): File pattern specifying the location of the TFRecord files.
            feature_spec (dict): A dictionary that defines the expected features in the TFRecord files,
                                 where the keys are the feature names and the values are the corresponding
                                 tf.io.FixedLenFeature or tf.io.VarLenFeature specifications.
        """
        self.file_pattern = file_pattern
        self.feature_spec = feature_spec

    # Define a function to preprocess the raw data
    def preprocess_data(raw_data):
        # Perform any necessary data preprocessing steps here
        preprocessed_data = raw_data * 2  # Example preprocessing step
        return preprocessed_data

    # Define a function to write a set of preprocessed data to a single TFRecord file
    def write_tfrecord_file(preprocessed_data, filename):
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


    # Get a list of all raw data files in the data/raw directory
    raw_data_files = tf.io.gfile.glob(os.path.join(RAW_DATA_DIR, FILE_PATTERN))
    # Loop over the raw data files and preprocess each file
    for raw_data_file in raw_data_files:
        # Read the raw data from the file
        raw_data = np.loadtxt(raw_data_file)
        # Preprocess the raw data
        preprocessed_data = preprocess_data(raw_data)
        # Split the preprocessed data into NUM_SHARDS chunks and write each chunk to a separate TFRecord file
        shard_size = len(preprocessed_data) // NUM_SHARDS
        for shard_idx in range(NUM_SHARDS):
            start_idx = shard_idx * shard_size
            end_idx = (shard_idx + 1) * shard_size
            shard_data = preprocessed_data[start_idx:end_idx]
            filename = os.path.join(PROCESSED_DATA_DIR, f"{os.path.basename(raw_data_file)}.shard{shard_idx+1}.tfrecord")
            # Write the shard of preprocessed data to a TFRecord file
            write_tfrecord_file(shard_data, filename)