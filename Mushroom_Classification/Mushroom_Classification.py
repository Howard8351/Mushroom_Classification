import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.data import Dataset
import numpy as np
import pathlib
import random
import math

#匯入類別資料
Mushroom_data_path = "mushrooms.csv"
Mushroom_data  = pathlib.Path(Mushroom_data_path).read_text()
#以分隔符號將資料分割並移除header
Mushroom_data  = Mushroom_data.split("\n")[1:]
#定義每個特徵的資料類別
col_data_type = [str()] * 23
#根據給予的資料類別建立一個包含數個tensor的list
#每個tensor代表一個特徵
Mushroom_data  = tf.io.decode_csv(Mushroom_data, record_defaults = col_data_type)
#Mushroom_data中都是用chart表示特徵類別
#要轉成數值才能分析
def string_feature_to_numeric_label(Mushroom_data, index):
    Mushroom_data[index] = tf.unique(Mushroom_data[index]).idx
    index +=1
    return (Mushroom_data, index)
condition          = lambda Mushroom_data, index : index < len(Mushroom_data)
while_index        = 0
tf.while_loop(condition, string_feature_to_numeric_label, (Mushroom_data, while_index), parallel_iterations = len(Mushroom_data))
#確認有幾種類別
number_of_class = tf.unique(Mushroom_data[0]).y.shape[0]

Mushroom_data = tf.transpose(tf.convert_to_tensor(Mushroom_data))

class Mushroom_data_classificatioin():
    def __init__(self, Mushroom_data, learning_rate, batch_size, epochs, number_of_class):
        self.Mushroom_data         = Mushroom_data
        self.batch_size            = batch_size
        self.learning_rate         = learning_rate
        self.epochs                = epochs
        self.number_of_class       = number_of_class
        self.train_dataset         = None
        self.train_data_szie       = None
        self.test_dataset          = None
        self.test_data_size        = None
        self.train_step_each_epoch = None
        self.test_step_each_epoch  = None
        self.model                 = None
        self.train_data_label      = None
        self.test_data_label       = None

    def creat_dataset(self, training = True):
        #將資料分成訓練和測試資料集
        number_of_data        = self.Mushroom_data.shape[0]
        random.seed(10)
        random_sample_index   = random.sample(range(number_of_data), number_of_data)
        train_data_proportion = 0.8        
        self.train_data_size  = math.ceil(number_of_data * train_data_proportion)
        self.test_data_size   = number_of_data - self.train_data_size
        train_data_set        = tf.convert_to_tensor(self.Mushroom_data.numpy()[0:self.train_data_size, :])
        test_data_set         = tf.convert_to_tensor(self.Mushroom_data.numpy()[self.train_data_size:number_of_data, :])
        if not training:
            self.train_data_label = tf.convert_to_tensor(self.Mushroom_data.numpy()[0:self.train_data_size, 0])
            self.test_data_label  = tf.convert_to_tensor(self.Mushroom_data.numpy()[self.train_data_size:number_of_data, 0])
        
        #建立訓練資料集和測試資料集
        if training:
            self.train_dataset, self.train_step_each_epoch = self.creat_list_dataset(train_data_set, self.train_data_size, training, shuffle = True)
            self.test_dataset, self.test_step_each_epoch   = self.creat_list_dataset(test_data_set, self.test_data_size, training, shuffle = False)
        else:
            self.train_dataset, self.train_step_each_epoch = self.creat_list_dataset(train_data_set, self.train_data_size, training, shuffle = False)
            self.test_dataset, self.test_step_each_epoch   = self.creat_list_dataset(test_data_set, self.test_data_size, training, shuffle = False)

    def creat_list_dataset(self, data_set, data_size, training = True, shuffle = True):
        dataset = Dataset.from_tensor_slices(data_set)
        
        if training:
            dataset = dataset.map(self.data_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)
            if shuffle:
                dataset = dataset.shuffle(data_size).repeat()
            else:
                dataset = dataset.repeat()
        else:
            dataset = dataset.map(self.model_test_data_process, num_parallel_calls = tf.data.experimental.AUTOTUNE)

        dataset         = dataset.batch(self.batch_size)
        step_each_epoch = math.ceil(data_size / self.batch_size)

        return dataset, step_each_epoch

    def data_process(self, data):
        label, data_feature = tf.split(data, [1, data.shape[0] -1], 0)
        label               = tf.one_hot(tf.squeeze(label), self.number_of_class)
        return (data_feature, label)

    def model_test_data_process(self, data):
        label, data_feature = tf.split(data, [1, data.shape[0] -1], 0)
        return data_feature

    def creat_model(self):
        input_layer   = Input(shape = (self.Mushroom_data.shape[1] - 1,))
        dense_layer_1 = self.creat_dense_layer(input_layer, 32)
        dense_layer_2 = self.creat_dense_layer(dense_layer_1, 64)
        dense_layer_3 = self.creat_dense_layer(dense_layer_2, 16)
        output_layer  = layers.Dense(self.number_of_class, activation = 'softmax')(dense_layer_3)

        self.model = Model(input_layer, output_layer)
        self.model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate),
                           loss = tf.keras.losses.CategoricalCrossentropy(), metrics = "accuracy")

    def creat_dense_layer(self, input_layer, nodes):
        dense_layer = layers.Dense(nodes)(input_layer)
        bn_layer    = layers.BatchNormalization()(dense_layer)
        act_layer   = layers.PReLU()(bn_layer)

        return act_layer
        
    def model_training(self):
        self.creat_dataset(training = True)
        self.creat_model()
        self.model.summary()
        

        #建立callback
        callback_path              = "model_callback_output/model_weights"
        tensorboard_path           = "tensorboard_output" 
        model_check_point_callback = tf.keras.callbacks.ModelCheckpoint(filepath = callback_path, monitor = "val_loss", save_best_only = True,
                                                                        save_weights_only = True)
        tensorboard_callback       = tf.keras.callbacks.TensorBoard(log_dir = tensorboard_path, histogram_freq = 1)

        history = self.model.fit(x = self.train_dataset, steps_per_epoch = self.train_step_each_epoch, epochs = self.epochs,
                                  validation_data = self.test_dataset, validation_steps = self.test_step_each_epoch, 
                                  callbacks = [model_check_point_callback, tensorboard_callback])

    def mode_test(self):
        self.creat_dataset(training = False)
        self.creat_model()
        callback_path = "model_callback_output/model_weights"
        self.model.load_weights(callback_path)

        train_data_label  = tf.one_hot(self.train_data_label, self.number_of_class)
        train_data_number = tf.unique_with_counts(self.train_data_label).count.numpy()
        class_accuracy    = [0.] * self.number_of_class

        model_predict = self.model.predict(x = self.train_dataset, steps = self.train_step_each_epoch)
        accuracy      = tf.keras.metrics.categorical_accuracy(train_data_label, model_predict)
        for index in range(self.train_data_label.shape[0]):
            class_accuracy[self.train_data_label[index].numpy()] += accuracy[index].numpy()
        class_accuracy = class_accuracy / train_data_number
        accuracy       = tf.math.reduce_mean(accuracy)
        print(f"Model train Accuracy:{accuracy}")
        for index in range(self.number_of_class):
            accuracy = class_accuracy[index]
            print(f"Class {index} train Accuracy: {accuracy}")


        test_data_label  = tf.one_hot(self.test_data_label, self.number_of_class)
        test_data_number = tf.unique_with_counts(self.test_data_label).count.numpy()
        class_accuracy    = [0.] * self.number_of_class

        model_predict = self.model.predict(x = self.test_dataset, steps = self.test_step_each_epoch)
        accuracy      = tf.keras.metrics.categorical_accuracy(test_data_label, model_predict)
        for index in range(self.test_data_label.shape[0]):
            class_accuracy[self.test_data_label[index].numpy()] += accuracy[index].numpy()
        class_accuracy = class_accuracy / test_data_number
        accuracy       = tf.math.reduce_mean(accuracy)
        print(f"Model test Accuracy:{accuracy}")
        for index in range(self.number_of_class):
            accuracy = class_accuracy[index]
            print(f"Class {index} test Accuracy: {accuracy}")
        

learning_rate = 0.0001
batch_size    = 16
epochs        = 20

Mushroom_classification_model = Mushroom_data_classificatioin(Mushroom_data, learning_rate, batch_size, epochs, number_of_class)

Mushroom_classification_model.model_training()

Mushroom_classification_model.mode_test()