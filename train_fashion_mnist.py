import pathlib, json, numpy as np, tensorflow as tf
from keras import layers, models, callbacks

### 0. 輸出資料夾
MODEL_DIR   = pathlib.Path("model")
MODEL_DIR.mkdir(exist_ok=True)
H5_PATH     = MODEL_DIR / "fashion_mnist.h5"
WEIGHT_NPZ  = MODEL_DIR / "fashion_mnist.npz"
ARCH_JSON   = MODEL_DIR / "fashion_mnist.json"

### 1. 讀資料並標準化
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0   # shape (60000, 28, 28)
x_test  = x_test.astype("float32")  / 255.0
# Keras Dense 只吃 2-D，因此用 Flatten
x_train = x_train.reshape(-1, 28*28)
x_test  = x_test.reshape(-1, 28*28)

### 2. 建立模型  (Flatten -> Dense 512 -> Dense 256 -> Dense 10 softmax)
inputs  = layers.Input(shape=(28*28,), name="input_flat")
x = layers.Dense(512, activation="relu", name="fc1")(inputs)
x = layers.Dense(256, activation="relu", name="fc2")(x)
outputs = layers.Dense(10, activation="softmax", name="logits")(x)

model = models.Model(inputs, outputs, name="fashion_dense")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

### 3. 訓練
es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
model.fit(
    x_train, y_train,
    epochs=40,
    batch_size=512,
    validation_split=0.1,
    callbacks=[es],
    verbose=2,
)

print("\nTest accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])

### 4. 儲存 .h5
model.save(H5_PATH)

### 5. 將權重與架構轉存為 .npz / .json
# 5-1   保存權重
weight_dict = {}
for layer in model.layers:
    if isinstance(layer, layers.Dense):
        W, b = layer.get_weights()
        weight_dict[f"{layer.name}.kernel"] = W
        weight_dict[f"{layer.name}.bias"]   = b
np.savez_compressed(WEIGHT_NPZ, **weight_dict)

# 5-2   產生架構 (符合 nn_predict.py 所需格式)
arch = []
for layer in model.layers:
    if isinstance(layer, layers.InputLayer):
        continue
    if isinstance(layer, layers.Dense):
        arch.append({
            "name"  : layer.name,
            "type"  : "Dense",
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

# 在前端推論時我們已把 Flatten 寫在第一層，因此保留即可
with open(ARCH_JSON, "w") as f:
    json.dump(arch, f, indent=2)

print(f"\n✅ 已輸出：\n  ‣ {H5_PATH}\n  ‣ {WEIGHT_NPZ}\n  ‣ {ARCH_JSON}")
