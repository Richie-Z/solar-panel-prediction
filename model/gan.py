import tensorflow as tf
from tensorflow.keras import layers, models


def build_generator(latent_dim, num_features):
    noise_input = tf.keras.Input(shape=(latent_dim,))
    weather_input = tf.keras.Input(shape=(num_features,))
    weather_normalized = tf.keras.layers.BatchNormalization()(weather_input)
    x = tf.keras.layers.Concatenate()([noise_input, weather_normalized])
    x = tf.keras.layers.Dense(2048)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    def residual_block(x, units, dropout_rate=0.3):
        skip = x
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if skip.shape[-1] == units:
            x = tf.keras.layers.Add()([x, skip])
        return x

    x = residual_block(x, 1024, 0.3)
    x = residual_block(x, 512, 0.3)
    x = residual_block(x, 256, 0.2)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    def custom_activation(x):

        return tf.keras.activations.softplus(x) + tf.keras.backend.epsilon()

    output = tf.keras.layers.Dense(1, activation=custom_activation)(x)

    return tf.keras.Model(
        inputs=[noise_input, weather_input], outputs=output, name="Generator"
    )


def build_discriminator(num_features):
    pv_input = tf.keras.Input(shape=(1,))
    weather_input = tf.keras.Input(shape=(num_features,))

    pv_normalized = tf.keras.layers.BatchNormalization()(pv_input)
    weather_normalized = tf.keras.layers.BatchNormalization()(weather_input)

    x = tf.keras.layers.Concatenate()([pv_normalized, weather_normalized])

    def dense_block(x, units, dropout_rate=0.2):
        x = tf.keras.layers.Dense(
            units, kernel_constraint=tf.keras.constraints.MaxNorm(1.0)
        )(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        return x

    x = dense_block(x, 512, 0.3)
    x = dense_block(x, 256, 0.3)
    x = dense_block(x, 128, 0.2)
    x = dense_block(x, 64, 0.2)

    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(
        inputs=[pv_input, weather_input], outputs=output, name="Discriminator"
    )


class SolarGAN(tf.keras.Model):
    def __init__(self, latent_dim, num_features):
        super(SolarGAN, self).__init__()
        self.latent_dim = latent_dim
        self.num_features = num_features
        self.generator = build_generator(latent_dim, num_features)
        self.discriminator = build_discriminator(num_features)
        self.gp_weight = 10.0

        self.l1_weight = tf.Variable(0.1, trainable=False)

    def compile(self, g_optimizer, d_optimizer):
        super(SolarGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @tf.function
    def train_step(self, data):
        real_pv, weather_features = data
        batch_size = tf.shape(real_pv)[0]

        d_steps = tf.minimum(
            5,
            tf.maximum(
                1,
                tf.cast(
                    (
                        tf.abs(
                            self.generator.losses[-1] / self.discriminator.losses[-1]
                        )
                        if len(self.generator.losses) > 0
                        and len(self.discriminator.losses) > 0
                        else 3
                    ),
                    tf.int32,
                ),
            ),
        )

        for _ in range(d_steps):
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_pv = self.generator([noise, weather_features], training=True)

                real_pred = self.discriminator(
                    [real_pv, weather_features], training=True
                )
                fake_pred = self.discriminator(
                    [fake_pv, weather_features], training=True
                )

                alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
                interpolated = alpha * real_pv + (1 - alpha) * fake_pv

                with tf.GradientTape() as gp_tape:
                    gp_tape.watch(interpolated)
                    interp_pred = self.discriminator(
                        [interpolated, weather_features], training=True
                    )

                grads = gp_tape.gradient(interp_pred, interpolated)
                grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
                gradient_penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)

                d_loss = (
                    tf.reduce_mean(fake_pred)
                    - tf.reduce_mean(real_pred)
                    + self.gp_weight * gradient_penalty
                )

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)

            d_gradients = [tf.clip_by_norm(g, 1.0) for g in d_gradients]
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

        g_steps = tf.minimum(
            3,
            tf.maximum(
                1,
                tf.cast(
                    (
                        tf.abs(
                            self.discriminator.losses[-1] / self.generator.losses[-1]
                        )
                        if len(self.generator.losses) > 0
                        and len(self.discriminator.losses) > 0
                        else 2
                    ),
                    tf.int32,
                ),
            ),
        )

        for _ in range(g_steps):
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_pv = self.generator([noise, weather_features], training=True)
                fake_pred = self.discriminator(
                    [fake_pv, weather_features], training=True
                )

                l1_loss = tf.reduce_mean(tf.abs(fake_pv - real_pv))
                self.l1_weight.assign(
                    tf.maximum(
                        0.01,
                        0.1
                        * tf.exp(-0.1 * tf.cast(self.optimizer.iterations, tf.float32)),
                    )
                )

                g_loss = -tf.reduce_mean(fake_pred) + self.l1_weight * l1_loss

            g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)

            g_gradients = [tf.clip_by_norm(g, 1.0) for g in g_gradients]
            self.g_optimizer.apply_gradients(
                zip(g_gradients, self.generator.trainable_variables)
            )

        return {"d_loss": d_loss, "g_loss": g_loss}
