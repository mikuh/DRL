import tensorflow as tf

class CNNEmbeddingNet(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.layer2 = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.layer3 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        self.layer4 = tf.keras.layers.Flatten()
        self.layer5 = tf.keras.layers.Dense(256, activation='relu')

    def call(self, inputs):
        inputs /= 255
        output = self.layer1(inputs)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output


class DenseEmbeddingNet(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(256, activation='relu')
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')

    def call(self, inputs):
        output = self.layer1(inputs)
        output = self.layer2(output)
        return output


class PolicyNet(tf.keras.layers.Layer):
    def __init__(self, env):
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(env.action_space.n, activation='softmax')
        super().__init__()

    def call(self, inputs):
        inputs = self.layer1(inputs)
        return self.logits(inputs)


class ValueNet(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, inputs):
        inputs = self.layer1(inputs)
        return self.value(inputs)


class QNet(tf.keras.layers.Layer):
    def __init__(self, env):
        # self.layer1 = tf.keras.layers.Dense(64, activation='relu')
        self.logits = tf.keras.layers.Dense(env.action_space.n)
        super().__init__()

    def call(self, inputs):
        # inputs = self.layer1(inputs)
        return self.logits(inputs)