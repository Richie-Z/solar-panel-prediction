import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import (
    find_missing_date_ranges,
)

from gan import build_generator, build_discriminator, SolarGAN

from enums import DatasetColumns, WeatherDatasetColumns

from sklearn.preprocessing import MinMaxScaler
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


FILE_NAME = "dataset.csv"
WEATHER_DATASET = "dataset_weather.csv"


original_data = pd.read_csv(
    FILE_NAME,
    parse_dates=[DatasetColumns.STATISTICAL_PERIOD.value],
    index_col=DatasetColumns.STATISTICAL_PERIOD.value,
)

weather_data = pd.read_csv(
    WEATHER_DATASET,
    parse_dates=[WeatherDatasetColumns.DATETIME.value],
    index_col=WeatherDatasetColumns.DATETIME.value,
).asfreq("h")

weather_features = [
    WeatherDatasetColumns.TEMPERATURE_C.value,
    WeatherDatasetColumns.HUMIDITY_PERCENT.value,
]


gap_start, gap_end = find_missing_date_ranges(
    original_data, DatasetColumns.STATISTICAL_PERIOD.value
)
gap_dates = pd.date_range(start=gap_start, end=gap_end, freq="h")


pre_gap_data = original_data[original_data.index < gap_start].asfreq("h")
post_gap_data = original_data[original_data.index >= gap_end].asfreq("h")

pre_gap_train_size = int(len(pre_gap_data) * 0.8)
pre_gap_train = pre_gap_data.iloc[:pre_gap_train_size].copy()
pre_gap_test = pre_gap_data.iloc[pre_gap_train_size:]

pre_gap_train.loc[:, DatasetColumns.PV_YIELD.value] = pre_gap_train[
    DatasetColumns.PV_YIELD.value
].interpolate(method="linear")


pre_weather_data = weather_data[weather_data.index < gap_start].bfill()
pre_weather_data = pre_weather_data.reindex(pre_gap_data.index)
pre_weather_data_test = pre_weather_data.reindex(pre_gap_test.index)


gap_weather_data = weather_data.reindex(gap_dates).ffill()
post_weather_data = weather_data[weather_data.index >= gap_end].bfill()


pre_gap_train_combined = pre_gap_train.join(
    pre_weather_data[weather_features], how="inner"
)
pre_gap_test_combined = pre_gap_test.join(
    pre_weather_data_test[weather_features], how="inner"
)


LATENT_DIM = 50
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EPOCHS = 1000


def prepare_data(pre_gap_train_combined, pre_gap_test_combined, weather_features):

    combined_columns = [DatasetColumns.PV_YIELD.value] + weather_features
    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(pre_gap_train_combined[combined_columns])
    train_pv = train_scaled[:, 0:1].astype(np.float32)
    train_weather = train_scaled[:, 1:].astype(np.float32)

    test_scaled = scaler.transform(pre_gap_test_combined[combined_columns])
    test_pv = test_scaled[:, 0:1].astype(np.float32)
    test_weather = test_scaled[:, 1:].astype(np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(train_pv, tf.float32), tf.cast(train_weather, tf.float32))
    )
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_pv, tf.float32), tf.cast(test_weather, tf.float32))
    )
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset, scaler


def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.plot(history["d_loss"], label="Discriminator Loss")
    plt.plot(history["g_loss"], label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training History")
    plt.grid(True)
    plt.show()


def train_solar_gan(train_dataset, num_features):

    try:
        tf.config.experimental.set_memory_growth(
            tf.config.list_physical_devices("GPU")[0], True
        )
    except:
        pass

    solar_gan = SolarGAN(LATENT_DIM, num_features)

    initial_lr = 2e-4

    solar_gan.compile(
        g_optimizer=tf.keras.optimizers.Adam(initial_lr, beta_1=0.0, beta_2=0.9),
        d_optimizer=tf.keras.optimizers.Adam(initial_lr, beta_1=0.0, beta_2=0.9),
    )

    history = {"d_loss": [], "g_loss": []}
    best_loss = float("inf")
    patience = 50
    patience_counter = 0
    min_epochs = 300

    warmup_epochs = 10

    for epoch in range(EPOCHS):
        epoch_d_loss = []
        epoch_g_loss = []

        if epoch < warmup_epochs:
            current_lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            current_lr = initial_lr * (0.98 ** (epoch - warmup_epochs))

        solar_gan.g_optimizer = tf.keras.optimizers.Adam(
            current_lr, beta_1=0.0, beta_2=0.9
        )
        solar_gan.d_optimizer = tf.keras.optimizers.Adam(
            current_lr, beta_1=0.0, beta_2=0.9
        )

        for batch_data in train_dataset:

            d_steps = 3 if np.mean(epoch_d_loss) > 5 else 2
            for _ in range(d_steps):
                losses = solar_gan.train_step(batch_data)
                epoch_d_loss.append(float(losses["d_loss"]))
                epoch_g_loss.append(float(losses["g_loss"]))

        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)

        history["d_loss"].append(avg_d_loss)
        history["g_loss"].append(avg_g_loss)

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
            )
            print(f"Current learning rate: {current_lr:.6f}")

        if epoch >= min_epochs:
            current_loss = abs(avg_d_loss)
            if current_loss < best_loss * 1.1:
                best_loss = min(current_loss, best_loss)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return solar_gan, history


def generate_predictions(model, weather_features, scaler):

    weather_features = tf.cast(weather_features, tf.float32)
    batch_size = tf.shape(weather_features)[0]

    noise = tf.random.normal([batch_size, LATENT_DIM], dtype=tf.float32)

    predictions_scaled = model.generator([noise, weather_features], training=False)

    predictions_with_weather = np.concatenate(
        [predictions_scaled.numpy(), weather_features.numpy()], axis=1
    )
    predictions = scaler.inverse_transform(predictions_with_weather)[:, 0]

    return predictions


def evaluate_model(model, test_dataset, scaler):
    all_predictions = []
    all_true_values = []

    for test_pv, test_weather in test_dataset:

        batch_predictions = generate_predictions(model, test_weather, scaler)

        all_predictions.extend(batch_predictions)
        all_true_values.extend(test_pv.numpy().flatten())

    all_predictions = np.array(all_predictions)
    all_true_values = np.array(all_true_values)

    mse = np.mean((all_predictions - all_true_values) ** 2)
    mae = np.mean(np.abs(all_predictions - all_true_values))

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae:.4f}")

    return all_predictions, all_true_values


train_dataset, test_dataset, scaler = prepare_data(
    pre_gap_train_combined, pre_gap_test_combined, weather_features
)


num_weather_features = len(weather_features)
solar_gan, history = train_solar_gan(train_dataset, num_weather_features)
plot_training_history(history)


print("\nEvaluating model on test set...")
predictions, true_values = evaluate_model(solar_gan, test_dataset, scaler)

plt.figure(figsize=(12, 6))
plt.plot(true_values, label="True Values", marker="o")
plt.plot(predictions, label="Predicted Values", marker="x")
plt.xlabel("Sample Index")
plt.ylabel("PV Yield")
plt.legend()
plt.title("True vs Predicted PV Yield (First 100 samples)")
plt.show()
