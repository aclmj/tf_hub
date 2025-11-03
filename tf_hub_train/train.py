import os
# 强制只用 CPU（最稳妥）
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_DISABLE_XLA"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from keras.src.ops import shape
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("XLA enabled:", tf.config.optimizer.get_jit())  # 确认应为 False
print("GPU available:", tf.config.list_physical_devices('GPU'))
# print("Version: ", tf.__version__)
# print("Eager mode: ", tf.executing_eagerly())
# print("Hub version: ", hub.__version__)
# print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

train_data,validation_data,test_data = tfds.load(
    name='imdb_reviews',
    split=['train[:60%]', 'train[60%:]','test'],
    as_supervised=True,
    # try_gcs=False,
)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# print(train_examples_batch)
# print(train_labels_batch)

#创建一个使用 Tensorflow Hub 模型嵌入（embed）语句的Keras层
embedding = '/mnt/d/Users/31786/Downloads/model'
# hub_layer = hub.KerasLayer(embedding,
#                            input_shape=[],
#                            dtype=tf.string,
#                            trainable=True)
# print(hub_layer(train_examples_batch[:3]))
# model = keras.Sequential()
# model.add(hub_layer)
# model.add(keras.layers.Dense(16,activation='relu'))
# model.add(keras.layers.Dense(1))
# print(model.summary())
# 函数式API
# inputs =keras.Input(shape=(), dtype=tf.string)
#
# #强制hub层在CPU上
# with tf.device('/cpu:0'):
#     x = keras.layers.Lambda(lambda x:hub_layer(x), output_shape=(512,))(inputs) #用lambda函数包装hub层，把输入传给Hub层
# x = keras.layers.Dense(16,activation='relu')(x)
# outputs = keras.layers.Dense(1)(x)
#
# model = keras.Model(inputs=inputs, outputs=outputs)
# print(model.summary())
#hub层必须在CPU上构建
# with tf.device('/CPU:0'):
#     hub_layer = hub.KerasLayer(embedding,
#                                input_shape=[],
#                                dtype=tf.string,
#                                trainable=True)
# #函数式API
# inputs = keras.Input(shape=(),dtype=tf.string)
#
# # #hub层在CPU上运行
# with tf.device('/cpu:0'):
# #     x = hub_layer(inputs)
#     x = keras.layers.Lambda(lambda x:hub_layer(x), output_shape=(512,))(inputs) #用lambda函数包装hub层，把输入传给Hub层
# with tf.device('/GPU:0'):
#     x = keras.layers.Dense(16,activation='relu')(x)
#     outputs = keras.layers.Dense(1)(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# model.summary()

# with tf.device('/cpu:0'):
#     hub_layer = hub.KerasLayer(embedding,
#                                input_shape=[],
#                                dtype=tf.string,
#                                trainable=True)
#
# inputs = keras.Input(shape=(), dtype=tf.string,)
# with tf.device('/cpu:0'):
#     x = keras.layers.Lambda(lambda x: hub_layer(x), output_shape=(50,))(inputs)
# with tf.device('/gpu:0'):
#     x = keras.layers.Dense(16,activation='relu')(x)
#     outputs = keras.layers.Dense(1)(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# model.summary()
with tf.device('/CPU:0'):
    hub_layer = hub.KerasLayer(embedding,
                               input_shape=[],
                               dtype=tf.string,
                               trainable=True)
    inputs = keras.Input(shape=(), dtype=tf.string)
    x = keras.layers.Lambda(lambda x: hub_layer(x), output_shape=(512,))(inputs)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)


    #优化器和损失函数
    model.compile(optimizer= 'adam',
                  loss= keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    #训练模型
    history = model.fit(train_data.shuffle(10000).batch(512).prefetch(tf.data.AUTOTUNE),
                        epochs=10,
                        validation_data=validation_data.batch(512).prefetch(tf.data.AUTOTUNE),
                        verbose=1)
results = model.evaluate(test_data.batch(512).prefetch(tf.data.AUTOTUNE), verbose=2)
for name, value in zip(model.metrics_names, results):
    print('%s: %s' % (name, value))

model.save('hub_imdb.keras')