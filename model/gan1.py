import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from helpers import (
    find_missing_date_ranges,
)

from enums import DatasetColumns, WeatherDatasetColumns

from sklearn.preprocessing import MinMaxScaler
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


def train_solar_gan(train_dataset, num_features):

    solar_gan = SolarGAN(LATENT_DIM, num_features)

    lr = 1e-4
    solar_gan.compile(
        g_optimizer=tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9),
        d_optimizer=tf.keras.optimizers.Adam(lr, beta_1=0.5, beta_2=0.9),
    )

    history = {"d_loss": [], "g_loss": []}

    best_loss = float("inf")
    patience = 50
    patience_counter = 0
    min_epochs = 500

    for epoch in range(EPOCHS):
        epoch_d_loss = []
        epoch_g_loss = []

        for batch_data in train_dataset:
            losses = solar_gan.train_step(batch_data)
            epoch_d_loss.append(float(losses["d_loss"]))
            epoch_g_loss.append(float(losses["g_loss"]))

        avg_d_loss = np.mean(epoch_d_loss)
        avg_g_loss = np.mean(epoch_g_loss)

        history["d_loss"].append(avg_d_loss)
        history["g_loss"].append(avg_g_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
            )

        if epoch >= min_epochs:

            current_loss = abs(avg_d_loss) + abs(avg_g_loss)
            if current_loss < best_loss * 1.1:
                best_loss = min(current_loss, best_loss)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return solar_gan, history


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
