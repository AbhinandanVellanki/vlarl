"""
Usage:
    pytest tests/test_tf.py -s
"""

import tensorflow as tf

def create_simple_mlp(input_size, hidden_size, output_size, num_layers=2):
    layers = [tf.keras.layers.InputLayer(input_shape=(input_size,))]
    layers.append(tf.keras.layers.Dense(hidden_size, activation='relu'))
    for _ in range(num_layers - 2):
        layers.append(tf.keras.layers.Dense(hidden_size, activation='relu'))
    layers.append(tf.keras.layers.Dense(output_size))
    return tf.keras.Sequential(layers)

def test_tensorflow_mlp_training():
    print("\nRunning simple MLP training test with TensorFlow...")

    # Check for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    device = "/gpu:0"

    print(f"Using device: {device}")

    with tf.device(device):
        # Parameters
        input_size = 768
        hidden_size = 512
        output_size = 768
        num_layers = 10
        batch_size = 1024
        num_epochs = 1000000
        learning_rate = 0.001

        # Model, Loss, and Optimizer
        model = create_simple_mlp(input_size, hidden_size, output_size, num_layers)
        criterion = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Dummy data
        dummy_input = tf.random.normal([batch_size, input_size])
        dummy_target = tf.random.normal([batch_size, output_size])

        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                outputs = model(dummy_input, training=True)
                loss = criterion(dummy_target, outputs)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.numpy():.4f}")

    print("Simple MLP training test with TensorFlow completed successfully.")
