import pathlib, json, numpy as np, tensorflow as tf
from keras import layers, models, callbacks, losses, optimizers, initializers

### 0. 輸出資料夾
MODEL_DIR   = pathlib.Path(r"C:\Users\elvis\Desktop\dnn-inference-113618028\model")
MODEL_DIR.mkdir(exist_ok=True)
H5_PATH     = MODEL_DIR / "fashion_mnist.h5"
WEIGHT_NPZ  = MODEL_DIR / "fashion_mnist.npz"
ARCH_JSON   = MODEL_DIR / "fashion_mnist.json"

### 1. 讀資料並標準化
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28*28).astype("float32")  / 255.0

### 2. 建立更深更寬的全連接網路（He 正則初始化 + ReLU）
he_init = initializers.HeNormal()
inputs = layers.Input(shape=(28*28,), name="input_flat")

# 多層 Dense，只用 ReLU activation
x = layers.Dense(1024, activation="relu", kernel_initializer=he_init, name="fc1")(inputs)
x = layers.Dense(512,  activation="relu", kernel_initializer=he_init, name="fc2")(x)
x = layers.Dense(512,  activation="relu", kernel_initializer=he_init, name="fc3")(x)
x = layers.Dense(256,  activation="relu", kernel_initializer=he_init, name="fc4")(x)
outputs = layers.Dense(10, activation="softmax", name="logits")(x)

model = models.Model(inputs, outputs, name="fashion_dense_deep")
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

### 3. 訓練（更長 epoch + LR 調度 + EarlyStopping）
es  = callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    x_train, y_train,
    epochs=80,
    batch_size=256,
    validation_split=0.1,
    callbacks=[es, rlr],
    verbose=2,
)

print("\nTest accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])

### 4. 儲存 .h5
model.save(H5_PATH)

### 5. 匯出 weights (.npz) & architecture (.json)
weight_dict = {}
for layer in model.layers:
    if isinstance(layer, layers.Dense):
        W, b = layer.get_weights()
        weight_dict[f"{layer.name}.kernel"] = W
        weight_dict[f"{layer.name}.bias"]   = b
np.savez_compressed(WEIGHT_NPZ, **weight_dict)

arch = []
for layer in model.layers:
    if isinstance(layer, layers.Dense):
        arch.append({
            "name": layer.name,
            "type": "Dense",
            "config": {"activation": layer.activation.__name__},
            "weights": [f"{layer.name}.kernel", f"{layer.name}.bias"],
        })
    elif isinstance(layer, layers.Flatten):
        arch.append({
            "name": layer.name,
            "type": "Flatten",
            "config": {},
            "weights": [],
        })

with open(ARCH_JSON, "w") as f:
    json.dump(arch, f, indent=2)

print(f"\n✅ 輸出完成：\n  ‣ {H5_PATH}\n  ‣ {WEIGHT_NPZ}\n  ‣ {ARCH_JSON}")
