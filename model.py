import tensorflow as tf


class NCFModel(tf.keras.Model):
    def __init__(self, num_users=None, num_items=None, embedding_size=16, **kwargs):
        super(NCFModel, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        if num_users and num_items:
            self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)
            self.item_embedding = tf.keras.layers.Embedding(num_items, embedding_size)
            self.dense1 = tf.keras.layers.Dense(64, activation='relu')
            self.dense2 = tf.keras.layers.Dense(32, activation='relu')
            self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        user_id, item_id = inputs
        user_vector = self.user_embedding(user_id)
        item_vector = self.item_embedding(item_id)
        x = tf.concat([user_vector, item_vector], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)
