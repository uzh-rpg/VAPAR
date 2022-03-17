import os
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Required for catkin import strategy
try:
    from .nets import create_network
    from .body_dataset import create_dataset
except:
    from nets import create_network
    from body_dataset import create_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BodyrateLearner(object):

    def __init__(self, settings, expect_partial=False):
        self.config = settings
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.set_visible_devices(physical_devices[self.config.gpu:(self.config.gpu + 1)], "GPU")
            tf.config.experimental.set_memory_growth(physical_devices[self.config.gpu], True)

        self.min_val_loss = tf.Variable(np.inf, name='min_val_loss', trainable=False)

        self.network = create_network(self.config)
        self.loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, clipvalue=.2)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='validation_loss')

        self.global_epoch = tf.Variable(0)

        self.ckpt = tf.train.Checkpoint(step=self.global_epoch,
                                        optimizer=self.optimizer,
                                        net=self.network)

        if self.config.resume_training:
            if expect_partial:
                model_loaded = self.ckpt.restore(self.config.resume_ckpt_file).expect_partial()
            else:
                model_loaded = self.ckpt.restore(self.config.resume_ckpt_file)

            if model_loaded:
                print("------------------------------------------")
                print("[BodyrateLearner] Restored from {}".format(self.config.resume_ckpt_file))
                print("------------------------------------------")
                return

        print("------------------------------------------")
        print("[BodyrateLearner] Initializing from scratch.")
        print("------------------------------------------")

    @tf.function
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.network(inputs)
            loss = self.loss(labels, predictions)
        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        self.train_loss.update_state(loss)
        return gradients

    @tf.function
    def val_step(self, inputs, labels):
        predictions = self.network(inputs)
        loss = self.loss(labels, predictions)
        self.val_loss.update_state(loss)

    def adapt_input_data(self, features):
        if (self.config.use_fts_tracks or self.config.use_images) and self.config.attention_fts_type != "none":
            inputs = {
                "attention_fts": features[2],
                "fts" if self.config.use_fts_tracks else "image": features[1],
                "state": features[0],
            }
        elif self.config.use_fts_tracks or self.config.use_images:
            inputs = {
                "fts" if self.config.use_fts_tracks else "image": features[1],
                "state": features[0],
            }
        elif self.config.attention_fts_type != "none":
            inputs = {
                "attention_fts": features[1],
                "state": features[0],
            }
        else:
            inputs = {
                "state": features[0],
            }
        if self.config.attention_branching or self.config.gate_direction_branching:
            inputs["attention_label" if self.config.attention_branching else "gate_direction_label"] = features[-1]
        if self.config.no_ref and not self.config.use_imu:
            del inputs["state"]
        if len(inputs) == 0:
            raise KeyError("No inputs for network specified.")
        return inputs

    def write_train_summaries(self, features, gradients):
        with self.summary_writer.as_default():
            tf.summary.scalar('Train Loss', self.train_loss.result(),
                              step=self.optimizer.iterations)
            for g, v in zip(gradients, self.network.trainable_variables):
                tf.summary.histogram(v.name, g, step=self.optimizer.iterations)

    def train(self):
        print("[BodyrateLearner] Training Network")
        if not hasattr(self, 'train_log_dir'):
            # This should be done only once
            self.train_log_dir = os.path.join(self.config.log_dir, 'train')
            self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.train_log_dir, max_to_keep=10)
        else:
            # we are in DAgger mode, so let us reset the best loss
            self.min_val_loss = np.inf
            self.train_loss.reset_states()
            self.val_loss.reset_states()

        dataset_train = create_dataset(self.config.train_dir,
                                       self.config, training=True)
        dataset_val = create_dataset(self.config.val_dir,
                                     self.config, training=False)

        for epoch in range(self.config.max_training_epochs):
            epoch_start = time.time()

            # train
            for k, (features, label) in enumerate(tqdm(dataset_train.batched_dataset, disable=True)):
                features = self.adapt_input_data(features)
                gradients = self.train_step(features, label)
                if tf.equal(k % self.config.summary_freq, 0):
                    self.write_train_summaries(features, gradients)
                    self.train_loss.reset_states()
            # eval
            for features, label in tqdm(dataset_val.batched_dataset, disable=True):
                features = self.adapt_input_data(features)
                self.val_step(features, label)
            validation_loss = self.val_loss.result()
            with self.summary_writer.as_default():
                tf.summary.scalar("Validation Loss", validation_loss, step=tf.cast(self.global_epoch, dtype=tf.int64))
            self.val_loss.reset_states()

            self.global_epoch = self.global_epoch + 1
            self.ckpt.step.assign_add(1)

            print("[BodyrateLearner] Epoch: {}, validation Loss: {:.4f} (after {:.2f}s)"
                  .format(self.global_epoch, validation_loss, time.time() - epoch_start))

            if validation_loss < self.min_val_loss or ((epoch + 1) % self.config.save_every_n_epochs) == 0:
                if validation_loss < self.min_val_loss:
                    self.min_val_loss = validation_loss
                save_path = self.ckpt_manager.save()
                print("[BodyrateLearner] Saved checkpoint for epoch {}: {}".format(int(self.ckpt.step), save_path))

        # Reset the metrics for the next epoch
        print("------------------------------------------------")
        print("[BodyrateLearner] Training finished successfully")
        print("------------------------------------------------")

    def test(self):
        print("[BodyrateLearner] Testing Network")
        self.train_log_dir = os.path.join(self.config.log_dir, 'test')
        dataset_val = create_dataset(self.config.test_dir,
                                     self.config, training=False)

        for features, label in tqdm(dataset_val.batched_dataset):
            features = self.adapt_input_data(features)
            self.val_step(features, label)
        validation_loss = self.val_loss.result()
        self.val_loss.reset_states()

        print("[BodyrateLearner] Testing Loss: {:.4f}".format(validation_loss))

    @tf.function
    def inference(self, inputs):
        predictions = self.network(inputs)
        return predictions
