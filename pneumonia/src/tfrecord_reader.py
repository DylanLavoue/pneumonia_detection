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

    # def _parse_fn(self, serialized_example):
    #     """
    #     Parses a serialized TFRecord example.

    #     Args:
    #         serialized_example (tf.Tensor): A serialized example from the TFRecord file.

    #     Returns:
    #         Parsed example with features as a dictionary of tensors.
    #     """
    #     parsed_example = tf.io.parse_single_example(serialized_example, self.feature_spec)
    #     # Add any additional parsing logic here, e.g., casting data types, reshaping tensors, etc.
    #     return parsed_example

    # def create_dataset(self, batch_size=1, num_epochs=None, shuffle_buffer_size=None,
    #                    prefetch_buffer_size=None):
    #     """
    #     Creates a TensorFlow Dataset from the TFRecord files.

    #     Args:
    #         batch_size (int): Number of examples to combine into a single batch.
    #         num_epochs (int): Number of times to iterate over the dataset. If set to None,
    #                           the dataset will be repeated indefinitely.
    #         shuffle_buffer_size (int): Number of elements from the dataset to buffer in a random order
    #                                   before drawing from it.
    #         prefetch_buffer_size (int): Number of elements from the dataset to prefetch in order to
    #                                    improve performance.

    #     Returns:
    #         A TensorFlow Dataset containing the parsed and preprocessed examples.
    #     """
    #     # Create a list of file patterns if multiple file patterns are provided
    #     if isinstance(self.file_pattern, list):
    #         files = tf.data.Dataset.list_files(self.file_pattern)
    #     else:
    #         files = self.file_pattern

    #     # Create a TFRecordDataset from the list of files
    #     dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    #     # Shuffle the dataset if shuffle_buffer_size is provided
    #     if shuffle_buffer_size:
    #         dataset = dataset.shuffle(shuffle_buffer_size)

    #     # Parse the serialized examples
    #     dataset = dataset.map(self._parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    #     # Batch the dataset
    #     dataset = dataset.batch(batch_size)

    #     # Repeat the dataset for multiple epochs if num_epochs is provided
    #     if num_epochs:
    #         dataset = dataset.repeat(num_epochs)

    #     # Prefetch the dataset for better performance
    #     if prefetch_buffer_size:
    #         dataset = dataset.prefetch(prefetch_buffer_size)

    #     return dataset


# def write_records_to_tfrecord(example_path):
#     # Open the TFRecord file for writing
#     with tf.io.TFRecordWriter(example_path) as file_writer:
#         # Generate random x, y values and write records to the file
#         for _ in range(4):
#             x, y = np.random.random(), np.random.random()

#             # Serialize the record as a TFExample
#             record_bytes = tf.train.Example(features=tf.train.Features(feature={
#                 "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
#                 "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
#             })).SerializeToString()

#             # Write the serialized record to the file
#             file_writer.write(record_bytes)

# # Example usage:
# example_path = "example.tfrecord"
# write_records_to_tfrecord(example_path)



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