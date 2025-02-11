import tensorflow as tf


def build_generator(latent_dim, num_features):
    noise_input = tf.keras.Input(shape=(latent_dim,))
    weather_input = tf.keras.Input(shape=(num_features,))

    weather_x = tf.keras.layers.BatchNormalization()(weather_input)
    weather_x = tf.keras.layers.Dense(32)(weather_x)
    weather_x = tf.keras.layers.LeakyReLU(alpha=0.2)(weather_x)
    weather_x = tf.keras.layers.BatchNormalization()(weather_x)

    noise_x = tf.keras.layers.Dense(64)(noise_input)
    noise_x = tf.keras.layers.LeakyReLU(alpha=0.2)(noise_x)
    noise_x = tf.keras.layers.BatchNormalization()(noise_x)

    x = tf.keras.layers.Concatenate()([noise_x, weather_x])

    def dense_block(x, units, dropout_rate=0.3):
        skip = x
        skip = tf.keras.layers.Dense(units)(skip) if skip.shape[-1] != units else skip

        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        return tf.keras.layers.Add()([x, skip])

    x = dense_block(x, 512, 0.3)
    x = dense_block(x, 256, 0.3)
    x = dense_block(x, 256, 0.3)
    x = dense_block(x, 128, 0.3)
    x = dense_block(x, 128, 0.3)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=[noise_input, weather_input], outputs=output)


def build_discriminator(num_features):
    pv_input = tf.keras.Input(shape=(1,))
    weather_input = tf.keras.Input(shape=(num_features,))

    pv_normalized = tf.keras.layers.BatchNormalization()(pv_input)
    weather_normalized = tf.keras.layers.BatchNormalization()(weather_input)

    x = tf.keras.layers.Concatenate()([pv_normalized, weather_normalized])

    def critic_block(x, units, dropout_rate=0.3):
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Dropout(dropout_rate)(x)

    x = critic_block(x, 128, 0.3)
    x = critic_block(x, 256, 0.3)
    x = critic_block(x, 512, 0.3)
    x = critic_block(x, 256, 0.3)

    output = tf.keras.layers.Dense(1)(x)

    return tf.keras.Model(
        inputs=[pv_input, weather_input], outputs=output, name="Discriminator"
    )


class SolarGAN(tf.keras.Model):
    def __init__(self, latent_dim, num_features):
        super(SolarGAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = build_generator(latent_dim, num_features)
        self.discriminator = build_discriminator(num_features)
        self.gp_weight = 10.0

    def compile(self, g_optimizer, d_optimizer):
        super(SolarGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @tf.function
    def train_step(self, data):
        real_pv, weather_features = data
        batch_size = tf.shape(real_pv)[0]

        d_steps = 3
        g_steps = 1

        d_loss_avg = 0
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
                interpolated = real_pv + alpha * (fake_pv - real_pv)

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
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )
            d_loss_avg += d_loss

        g_loss_avg = 0
        for _ in range(g_steps):
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_pv = self.generator([noise, weather_features], training=True)
                fake_pred = self.discriminator(
                    [fake_pv, weather_features], training=True
                )

                wasserstein_loss = -tf.reduce_mean(fake_pred)
                l1_loss = 0.2 * tf.reduce_mean(tf.abs(fake_pv - real_pv))
                l2_loss = 0.1 * tf.reduce_mean(tf.square(fake_pv - real_pv))
                smoothness_loss = 0.1 * tf.reduce_mean(
                    tf.abs(fake_pv[1:] - fake_pv[:-1])
                )

                g_loss = wasserstein_loss + l1_loss + l2_loss + smoothness_loss

            g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(
                zip(g_gradients, self.generator.trainable_variables)
            )
            g_loss_avg += g_loss

        return {"d_loss": d_loss_avg / d_steps, "g_loss": g_loss_avg / g_steps}
