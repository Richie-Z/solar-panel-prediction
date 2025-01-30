import tensorflow as tf
from tensorflow.keras import layers, models


def build_generator(latent_dim, num_features):
    noise_input = tf.keras.Input(shape=(latent_dim,))
    weather_input = tf.keras.Input(shape=(num_features,))

    # Concatenate noise and weather features
    x = tf.keras.layers.Concatenate()([noise_input, weather_input])

    # Initial dense layer with larger size
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Intermediate layers with skip connections
    def residual_block(x, units):
        skip = x
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # Add skip connection if dimensions match
        if skip.shape[-1] == units:
            x = tf.keras.layers.Add()([x, skip])
        return x

    x = residual_block(x, 512)
    x = residual_block(x, 256)

    # Pre-output layer
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Output layer with softplus activation (smoother than ReLU)
    output = tf.keras.layers.Dense(1, activation="softplus")(x)

    return tf.keras.Model(
        inputs=[noise_input, weather_input], outputs=output, name="Generator"
    )


def build_discriminator(num_features):
    pv_input = tf.keras.Input(shape=(1,))
    weather_input = tf.keras.Input(shape=(num_features,))

    # Concatenate PV yield and weather features
    x = tf.keras.layers.Concatenate()([pv_input, weather_input])

    # Dense layers with LeakyReLU
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # Output without activation for WGAN
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

    def compile(self, g_optimizer, d_optimizer):
        super(SolarGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    @tf.function
    def train_step(self, data):
        real_pv, weather_features = data
        batch_size = tf.shape(real_pv)[0]

        # Train discriminator
        for _ in range(3):  # Reduced number of D updates
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_pv = self.generator([noise, weather_features], training=True)

                real_pred = self.discriminator(
                    [real_pv, weather_features], training=True
                )
                fake_pred = self.discriminator(
                    [fake_pv, weather_features], training=True
                )

                # Gradient penalty
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

                # Discriminator loss with reduced weight on gradient penalty
                d_loss = (
                    tf.reduce_mean(fake_pred)
                    - tf.reduce_mean(real_pred)
                    + self.gp_weight * gradient_penalty
                )

            d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradients, self.discriminator.trainable_variables)
            )

        # Train generator twice
        for _ in range(2):
            noise = tf.random.normal([batch_size, self.latent_dim])

            with tf.GradientTape() as tape:
                fake_pv = self.generator([noise, weather_features], training=True)
                fake_pred = self.discriminator(
                    [fake_pv, weather_features], training=True
                )

                # Generator loss with added L1 loss for stability
                g_loss = -tf.reduce_mean(fake_pred) + 0.1 * tf.reduce_mean(
                    tf.abs(fake_pv - real_pv)
                )

            g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(
                zip(g_gradients, self.generator.trainable_variables)
            )

        return {"d_loss": d_loss, "g_loss": g_loss}
