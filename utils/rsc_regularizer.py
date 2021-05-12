# Copyright 2021 PIC4SeR. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow_probability as tfp
import tensorflow as tf
import os
import numpy as np
from glob import glob
from scipy.stats import logistic


class RSCModelWrapper():
    """
    A wrapper class for RSC regularizer. It takes a backbone and a classification head and it applies RSC during the training phase.

    ...
    
    Attributes
    ----------
    backbone: obj
        model object containing the backbone network
    classification_head: obj
        model object containing the classification head
    percentile: int
        gradient percentaile. All greater gradients are discarded
    batch_percentage: int
        percentage of batch samples to apply RSC regularization
    trainable_backbone: bool
        if True it freezes backbone parameters
    
    Methods
    -------

    predict(x, batch_size=32):
        Predict a sample
    evaluate(x, y_true, batch_size=32):
        Use the model to compute metrics. Model must be compiled
    summary():
        Print summary of the model
    save_weigths(bin_dir_custom=None):
        Save model weights
    compile(loss, metric, optimizer, pre_process_fc=None, name_model='model',
        checkpoint_dir='checkpoint', log_dir='logs', bin_dir='bin', do_not_restore=False, save_old=False, max_bck=3)
        Compile model with given loss, metrics and optimizer. Moreover it takes care of restoring a prerviously trained model.
    restore():
        Restore a previously trained model
    fit(X=None, y=None, batch_size=None, buffer_size=1000, epochs=100, evaluate_every="epoch",
            validation_data=None, initial_epoch=0, save_best_only=True, track="accuracy")
        Fit a model with the given data
    
    """
    def __init__(self, backbone, classification_head, percentile=66, batch_percentage=33, trainable_backbone=False, **kwargs):
        """    
        Parameters
        ----------
        backbone: obj
            model object containing the backbone network
        classification_head: obj
            model object containing the classification head
        percentile: int
            gradient percentaile. All greater gradients are discarded
        batch_percentage: int
            percentage of batch samples to apply RSC regularization
        trainable_backbone: bool
            if True it freezes backbone parameters
        """
        self.z = None
        self.percentile = percentile
        self.backbone = backbone
        self.batch_percentage = batch_percentage
        self.classification_head = classification_head
        self.model = self._build_model()
        self.compiled = False

        
        if not trainable_backbone:
            for layer in self.backbone.layers:
                layer.trainable = False
        
    def _build_model(self):
        """
        Build a model for summary and weights update
        """
        inputs = tf.keras.Input(shape=self.backbone.input.shape[1:])
        
        x = self.backbone(inputs)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        outputs = self.classification_head(x)
        
        return tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    def _call(self, inputs, mask=None):
        """
        Private method to run the graph with or without a gradient mask
        """
        self.z = self.backbone(inputs)
        
        if mask != None:
            x = self.z * mask # Ex. spatial_wise B x 7 x 7 x 512  <> B x 7 x 7 x 1
        else: 
            x = self.z
        
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        return self.classification_head(x)
    
    def predict(self, x, batch_size=32):
        """
        Use the model the obtain a prediction with the given sample(s)
                
        Parameters
        ----------
        x: tensor
            data to be predicted
        batch_size: int
            batch dimension during inference   
        """
        if x.shape[0] != 1:
            n_iter = x.shape[0] // batch_size
            n_remain = x.shape[0] % n_iter
        else:
            n_iter = 1
        y_pred = []
        for i in range(n_iter):
            y_pred.extend(tf.nn.softmax(self.model(x[batch_size*i:batch_size*(i+1)]), axis=-1))
            
        if x.shape[0] != 1:
            y_pred.extend(tf.nn.softmax((self.model(x[-n_remain:])), axis=-1))                          
        return y_pred
    
    def evaluate(self, x, y_true, batch_size=32):
        """
        Use the model to compute score of the given metrics
                
        Parameters
        ----------
        x: tensor
            data to be predicted
        y_true: tensor
            data true labels
        batch_size: int
            batch dimension during inference   
        """
        if self.compiled:
            loss_mean = tf.keras.metrics.Mean(name='loss')
            self.metric.reset_states()
            n_iter = x.shape[0] // batch_size
            n_remain = x.shape[0] % n_iter
            stateful_metrics=['Loss', 'Accuracy']
            pb_i = tf.keras.utils.Progbar(n_iter, stateful_metrics=stateful_metrics)
            values = [('Loss', self.train_loss.result()), ('Accuracy', self.train_accuracy.result())]
            for i in range(n_iter):
                y_pred = tf.nn.softmax(self.model(x[batch_size*i:batch_size*(i+1)]), axis=-1)
                loss_mean(self.loss(y_true[batch_size*i:batch_size*(i+1)][...,None], y_pred))
                self.metric(y_true[batch_size*i:batch_size*(i+1)][...,None], y_pred)
                values.append(('Loss', loss_mean.result()))
                values.append(('Accuracy', self.metric.result()))
                pb_i.add(1, values=values)
                
            y_pred = tf.nn.softmax(self.model(x[-n_remain:]), axis=-1)
            loss_mean(self.loss(y_true[-n_remain:][...,None], y_pred))
            self.metric(y_true[-n_remain:][...,None], y_pred)
            values.append(('Loss', loss_mean.result()))
            values.append(('Accuracy', self.metric.result()))                                     
            return [loss_mean.result().numpy(), self.metric.result().numpy()]
        else:
            print("[INFO] Model not yet compiled!")
    
    def summary(self):
        """
        Print the summary of the model
        """
        return self.model.summary()
    
    def save_weigths(self, bin_dir_custom=None):
        """
        Save weights of the model in a given folder
        
        Parameters
        ----------
        bin_dir_custom: str
            bin path
        """
        if self.compiled:
            self.restore()
            if bin_dir_custom:
                self.model.save_weights(bin_dir_custom)
            else:
                self.model.save_weights(self.bin_dir)

    def compile(self, loss, metric, optimizer, pre_process_fc=None, name_model='model', 
        checkpoint_dir='checkpoint', log_dir='logs', bin_dir='bin', do_not_restore=False, save_old=False, max_bck=3):
        """
        Compile the model with given configurations, loss, metrics and optimizer. Moreover it takes as input a custom pre-processing function
        to chain to tf.data object some custom pre-processing functions.
                        
        Parameters
        ----------
        loss: obj
            loss object
        metric: obj
            metric object
        optimizer: obj
            optimizer object
        pre_process_fc: func
            function with tf.data pre-processing custom pipeline
        name_model: str
            name of the model
        checkpoint_dir: str
            checkpoint path
        log_dir: str
            logs path
        bin_dir: str
            bin path
        do_not_restore: bool
            restore old model
        save_old: bool
            save old folder if name not changed
        max_bck: int
            max old folder with same name to keep
        """
        self.compiled = True
        self.name_model = name_model
        self.loss = loss
        self.metric = metric
        self.log_dir = os.path.join(log_dir, self.name_model)
        self.ckp_dir = os.path.join(checkpoint_dir, self.name_model)
        self.bin_dir = os.path.join(bin_dir, self.name_model)
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
        
        
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(0), loss=tf.Variable(np.inf),
                                              accuracy=tf.Variable(0.), optimizer=optimizer, model=self.model)
        
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint, max_to_keep=1,directory=self.ckp_dir)
        
        
        if pre_process_fc != None:
            self.batch_dataset = pre_process_fc
        else:
            self.batch_dataset = self._batch_dataset_default
        
        if do_not_restore:
            if save_old:
                self._backup_checkpoint(max_bck)
        else:
            self.restore()
    
    def _backup_checkpoint(self,max_bck):
        """
        Private method to save previously trained models with the same name into the configuration file.
        """
        if os.path.isdir(self.ckp_dir):
            bck_dir = self.ckp_dir + "_old"
            bck_list = [int(s.split('_')[-1][3:]) for s in glob(f"{bck_dir}*")]
            if len(bck_list):
                i = min(max(bck_list)+1,max_bck)
            else:
                i = 0
            bck_dir += f'{i}'
            os.rename(self.ckp_dir,bck_dir)
            print(f'Checkpoint backup saved for model {self.name_model}: {bck_dir}')
        if os.path.isdir(self.log_dir):
            bck_dir = self.log_dir + "_old"
            bck_list = [int(s.split('_')[-1][3:]) for s in glob(f"{bck_dir}*")]
            if len(bck_list):
                i = min(max(bck_list)+1,max_bck)
            else:
                i = 0
            bck_dir += f'{i}'
            os.rename(self.log_dir,bck_dir)
            print(f'Tensorboard graph backup saved for model {self.name_model}: {bck_dir}')
            
    def restore(self):
        """
        Restore a privously trained model
        """
        if self.checkpoint_manager.latest_checkpoint and self.compiled:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            print(f'Model restored from checkpoint at step {self.checkpoint.step.numpy()}.')


    def _batch_dataset_default(self, X, y, batch_size, buffer_size):
        """
        Default tf.data creation
        """
        ds = tf.data.Dataset.from_tensor_slices((X,y))
        ds = ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
        
    def fit(self, X=None, y=None, batch_size=None, buffer_size=1000, epochs=100, evaluate_every="epoch",
            validation_data=None, initial_epoch=0, save_best_only=True, track="accuracy"):
        """
        Train a model with the given data
        
        Parameters
        ----------
        X: tensor
            training data
        y: tensor
            training data true labels
        batch_size: int
            batch size dimension
        buffer_size: int
            shuffle tf.data buffer size
        epochs: int
            number of epoches
        evaluate_every: str
            evaluate every 'epoch' or 'step'
        validation_data: tupla
            tupla with X_val and y_val to evaluate with
        initial_epoch: int
            initial epoch to train with
        save_best_only:
            delete old saved weights during training
        track: str
            trach 'accuracy' or 'loss'
        """
        
        self.n_rsc_samples = int(batch_size * self.batch_percentage * 0.01)

        track = track.lower()
        if track not in ["accuracy","loss"]:
            raise ValueError(f"Cannot track {track}.")

        ds_len = X.shape[0]
        train_ds = self.batch_dataset(X, y, batch_size, buffer_size)
        writer_train = tf.summary.create_file_writer(os.path.join(self.log_dir, f'train'))
        

        stateful_metrics=['Loss', 'Accuracy']
        loss_to_track = tf.constant(np.inf)
        accuracy_to_track = tf.constant(0.)
    
        if validation_data is not None:
            val_ds = self.batch_dataset(validation_data[0], validation_data[1],batch_size,buffer_size)
            writer_val = tf.summary.create_file_writer(os.path.join(self.log_dir, f'val'))
            stateful_metrics.append('Val Loss')
            stateful_metrics.append('Val Accuracy')
        
        total_step = tf.cast(self.checkpoint.step,tf.int64)
        steps_per_epoch = np.ceil(ds_len/batch_size).astype("int")
        
        if validation_data is not None:
            if evaluate_every == "step":
                evaluate_every = 1
            elif evaluate_every == "epoch":
                evaluate_every = steps_per_epoch
            else:
                if not isinstance(evaluate_every,int):
                    raise ValueError(f'Wrong "evaluate_every": {evaluate_every}. Acceptable values are "step", "epoch" or int.')
                else:
                    evaluate_every = min(evaluate_every,steps_per_epoch)
        print(f"Validating validation dataset every {evaluate_every} steps.")
        
        for epoch in range(epochs - initial_epoch):
            print("\nEpoch {}/{}".format(epoch + 1 + initial_epoch, epochs))
            pb_i = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=stateful_metrics)
            
            for step, (x_train_step, y_train_step) in enumerate(train_ds):
                if step == 0: #a new epoch is starting -> reset metrics
                    self.train_loss.reset_states()
                    self.train_accuracy.reset_states()

                total_step += 1

                self._train_step(x_train_step, y_train_step)
                
                self.checkpoint.step.assign_add(1)
                
                with writer_train.as_default():
                    tf.summary.scalar('Accuracy', self.train_accuracy.result(), step=total_step)
                    tf.summary.scalar('Loss', self.train_loss.result(), step=total_step)
                writer_train.flush()
                
                values = [('Loss', self.train_loss.result()), ('Accuracy', self.train_accuracy.result())]
                    
                if validation_data is not None:
                    if step != 0 and ((step + 1) % evaluate_every) == 0:
                        self.val_loss.reset_states()
                        self.val_accuracy.reset_states()

                        for x_val_step, y_val_step in val_ds:
                            self._test_step(x_val_step, y_val_step)
                            
                        with writer_val.as_default():
                            tf.summary.scalar('Accuracy', self.val_accuracy.result(), step=total_step)
                            tf.summary.scalar('Loss', self.val_loss.result(), step=total_step)
                        writer_val.flush()

                        values.append(('Val Loss', self.val_loss.result()))
                        values.append(('Val Accuracy', self.val_accuracy.result()))
                        
                        loss_to_track = self.val_loss.result()
                        accuracy_to_track = self.val_accuracy.result()           
                else: # if validation is not available, track training
                    loss_to_track = self.train_loss.result()
                    accuracy_to_track = self.train_accuracy.result()
                
                pb_i.add(1, values=values) #update bar
                
                if save_best_only:
                    if (track=="loss" and loss_to_track >= self.checkpoint.loss) or \
                       (track=="accuracy" and accuracy_to_track <= self.checkpoint.accuracy): # no improvement, skip saving checkpoint
                        continue
                
                self.checkpoint.loss = loss_to_track
                self.checkpoint.accuracy = accuracy_to_track
                self.checkpoint_manager.save()

    @tf.function            
    def _train_step(self, x, y_true):
        """
        Private function to execute a training step with RSC regularizer.
        """
        x_rsc = x[:self.n_rsc_samples]
        y_true_rsc = y_true[:self.n_rsc_samples]
        
        x_no_rsc = x[self.n_rsc_samples:]
        y_true_no_rsc = y_true[self.n_rsc_samples:]
        
        #-------------------------------------
        #--------------NO RSC-----------------
        #-------------------------------------
        
        with tf.GradientTape() as tape: 
            y_pred = self._call(x_no_rsc)
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            loss = self.loss(y_true_no_rsc, y_pred)
            
        gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables) # compute gradients respect to model variables
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
        
        self.metric.reset_states()
        metric = self.metric(y_true_no_rsc, y_pred)
        self.train_loss(loss)
        self.train_accuracy(metric)
  
    
        #-------------------------------------
        #--------------RSC--------------------
        #-------------------------------------
        
        y_true_one_hot = tf.one_hot(tf.cast(y_true_rsc, dtype=tf.int32), tf.cast(self.classification_head.layers[-1].units, dtype=tf.int32), dtype=tf.float32)        

        with tf.GradientTape() as tape: # compute first loss
            y_pred = self._call(x_rsc)
            gz_func = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1) # B x 1
            
            choose_sw_cw = np.random.randint(0, 9) # if spatial or channel wise
   
            #-----------SPATIAL-WISE----------
        
        if choose_sw_cw <= 4:    
            gradients = tape.gradient(gz_func, self.z) # compute gradients respect to Z  # Ex. B x 7 x 7 x 512
            gradients_sw = tf.reduce_mean(gradients, axis=[-1]) #  Ex. B x 7 x 7
            
            mask = tf.cast(gradients_sw < tfp.stats.percentile(x=gradients_sw, q=self.percentile, axis=[1,2], keepdims=True), dtype=tf.float32)[...,None] # Ex. B x 7 x 7 x 1

    
             #-----------CHANNEL-WISE----------
        
        else:     
            gradients = tape.gradient(gz_func, self.z) # compute gradients respect to Z # Ex. B x 7 x 7 x 512
            gradients_cw = tf.reduce_mean(gradients, axis=[1,2]) # Ex. B x 512          
            mask = tf.cast(gradients_cw < tfp.stats.percentile(x=gradients_cw, q=self.percentile, axis=[1], keepdims=True), dtype=tf.float32) # Ex. B x 512
            mask = tf.reshape(mask, (mask.shape[0],1,1,-1)) # B x 1 x 1 x 512
        
        with tf.GradientTape() as tape: 
            y_pred = self._call(x_rsc, mask=mask)
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            loss = self.loss(y_true_rsc, y_pred)
            
        gradients = tape.gradient(loss, self.checkpoint.model.trainable_variables) # compute gradients respect to model variables
        self.checkpoint.optimizer.apply_gradients(zip(gradients, self.checkpoint.model.trainable_variables))
    
    @tf.function
    def _test_step(self, x, y_true):
        """
        Private function to execute a testing step with the model during the training phase.
        """
        y_pred = self._call(x)

        loss = self.loss(y_true, y_pred)
        self.metric.reset_states()
        metric = self.metric(y_true, y_pred)

        self.val_loss(loss)
        self.val_accuracy(metric)

