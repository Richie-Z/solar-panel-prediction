import tensorflow as tf
from tensorflow.keras import layers, models


def build_generator(input_dim, output_dim):
    model = models.Sequential(name="Generator")

    # Initial dense layer
    model.add(layers.Dense(512, input_dim=input_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Add residual blocks
    def residual_block(x, units):
        skip = x
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, skip])
        return x

    # Middle layers with residual connections
    x = model.layers[-1].output
    x = residual_block(x, 512)
    x = residual_block(x, 512)

    # Output layer
    x = layers.Dense(output_dim, activation="linear")(x)

    return models.Model(model.input, x)


def build_discriminator(input_dim):
    model = models.Sequential(name="Discriminator")

    # Feature extraction layers
    model.add(layers.Dense(512, input_dim=input_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.LayerNormalization())  # Layer normalization instead of batch norm
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.LayerNormalization())
    model.add(layers.Dropout(0.3))

    # Additional layer for better feature extraction
    model.add(layers.Dense(128))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.LayerNormalization())
    model.add(layers.Dropout(0.3))

    # Output layer with gradient penalty
    model.add(layers.Dense(1, activation="linear"))  # Linear activation for WGAN

    return model


class SolarGAN(models.Model):
    def __init__(self, latent_dim, feature_dim, generator, discriminator):
        super(SolarGAN, self).__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weight = 10.0  # Gradient penalty weight

    def gradient_penalty(self, real_samples, fake_samples, weather_features):
        batch_size = tf.shape(real_samples)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(
                tf.concat([interpolated, weather_features], axis=1)
            )

        grads = gp_tape.gradient(pred, interpolated)[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, data):
        real_pv, weather_features = data
        batch_size = tf.shape(real_pv)[0]

        # Train discriminator
        for _ in range(5):  # Multiple discriminator updates per generator update
            noise = tf.random.normal([batch_size, self.latent_dim])
            generator_inputs = tf.concat([noise, weather_features], axis=1)

            with tf.GradientTape() as disc_tape:
                fake_pv = self.generator(generator_inputs, training=True)

                real_output = self.discriminator(
                    tf.concat([real_pv, weather_features], axis=1), training=True
                )
                fake_output = self.discriminator(
                    tf.concat([fake_pv, weather_features], axis=1), training=True
                )

                gp = self.gradient_penalty(real_pv, fake_pv, weather_features)
                disc_loss = (
                    tf.reduce_mean(fake_output)
                    - tf.reduce_mean(real_output)
                    + self.gp_weight * gp
                )

            disc_grads = disc_tape.gradient(
                disc_loss, self.discriminator.trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(disc_grads, self.discriminator.trainable_variables)
            )

        # Train generator
        noise = tf.random.normal([batch_size, self.latent_dim])
        generator_inputs = tf.concat([noise, weather_features], axis=1)

        with tf.GradientTape() as gen_tape:
            fake_pv = self.generator(generator_inputs, training=True)
            fake_output = self.discriminator(
                tf.concat([fake_pv, weather_features], axis=1), training=True
            )
            gen_loss = -tf.reduce_mean(fake_output)

        gen_grads = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gen_grads, self.generator.trainable_variables)
        )

        return {"d_loss": disc_loss, "g_loss": gen_loss}
