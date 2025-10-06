import numpy as np

# 1. F = G*m1*m2/(r+h)^2
def calc_force(m1:float, m2: float, r: float, h: float) -> float:
    """
    Calculates the gravitational pull made by an object.

    This function calculates the gravitational force exerted by a 
    mass (m1) into an object of mass (m2), taking into account
    distance between masses, and planets radius.

    Args:
        m1 (float): mass of first object (kg)
        m2 (float): mass of second object (kg)
        r (float): radius of the more massive object (m)
        h (float): height with respect to planets outer layer (m)

    Returns:
        f (float): gravitational pull made by the first object to the second (N)
    """
    G = 6.674*(10**(-11))
    R = max(r + h, 1e-12)                 # CAMBIO: asegurar que r+h > 0
    # if h>= 0:
    return G * m1 * m2 / (R**2)           # CAMBIO: devuelve F físico (no log(F))
    # elif r-h == 0:
    #     return 0
    # else: 
    #     return G * m1 * m2 * (R) / (r**3)

# 2.Parameters
samples = 1000

# Generate our synthetic data
m1 = np.random.uniform(1e20, 1e24, samples) # more massive object
m2 = np.random.uniform(1, 1e5, samples)     # less massive object
r  = np.random.uniform(1e4, 1e6, samples)   # m1 planet radius
h  = np.random.uniform(-r + 1e-9, 2000, samples) # CAMBIO: asegurar r+h>0 siempre

data = []
for i in range(samples):
    f = calc_force(m1[i], m2[i], r[i], h[i])  # CAMBIO: devuelve F (no log)
    data.append([m1[i], m2[i], r[i], h[i], f]) # CAMBIO: F última columna

# convert to np for AI model
data = np.array(data)
print(data[:5])


from sklearn.model_selection import train_test_split

# split into train_val 80% and test 20%
train_val_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# From the train set, segregate between train and validation
train_set, val_set = train_test_split(train_val_set, test_size=0.2, random_state=42)

# CAMBIO: crear R = r + h y usar 3 columnas [m1, m2, R]
R_train = train_set[:, 2] + train_set[:, 3]
R_val   = val_set[:, 2] + val_set[:, 3]
R_test  = test_set[:, 2] + test_set[:, 3]

x_train = np.c_[train_set[:, 0], train_set[:, 1], np.maximum(R_train, 1e-12)]  # CAMBIO
x_val   = np.c_[val_set[:, 0], val_set[:, 1], np.maximum(R_val, 1e-12)]        # CAMBIO
x_test  = np.c_[test_set[:, 0], test_set[:, 1], np.maximum(R_test, 1e-12)]     # CAMBIO

y_train = train_set[:, 4]   # CAMBIO: target = F
y_val   = val_set[:, 4]     # CAMBIO
y_test  = test_set[:, 4]    # CAMBIO


from sklearn.preprocessing import StandardScaler

# 3.Data preprocessing
EPS = 1e-12  # CAMBIO: para logs seguros

# CAMBIO: trabajar en log-espacio
x_train_log = np.log(x_train + EPS)
x_val_log   = np.log(x_val   + EPS)
x_test_log  = np.log(x_test  + EPS)

y_train_log = np.log(y_train.reshape(-1, 1) + EPS)
y_val_log   = np.log(y_val.reshape(-1, 1)   + EPS)
y_test_log  = np.log(y_test.reshape(-1, 1)  + EPS)

# CAMBIO: escalar logs (no valores crudos)
x_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train_log)
x_val_scaled = x_scaler.transform(x_val_log)
x_test_scaled = x_scaler.transform(x_test_log)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train_log)
y_val_scaled = y_scaler.transform(y_val_log)
y_test_scaled = y_scaler.transform(y_test_log)


# 4.Designing our model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


def build_functional_model(input_dim):
    """ Build a 4-layer MLP

    This is an example of how to build a model using the functional model from the keras library.
    The only functional block has three layers,1 input layer, two hidden layers and one output layer.

    Arg:
        input_dim (int): input dimension, number of features

    Returns:
        model (model): Functional model
    """

    inputs = Input(shape=(input_dim,), name="input_layer")
    x = Dense(20, activation="relu", name="hidden_layer_1")(inputs)
    outputs = Dense(1, activation="linear", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# 5. Compile the model
input_dim = 3  # CAMBIO: ahora solo m1, m2, R
model = build_functional_model(input_dim)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # CAMBIO: mse en log-space

history = model.fit(x=x_train_scaled, y=y_train_scaled,
                    validation_data=(x_val_scaled, y_val_scaled),
                    epochs=200)


# Evaluate our model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def plot_training_curves(history):
    """
    This function plots the training and validation curves.
    """
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"],label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_scatter(y_test, y_pred):
    """
    Plot the scatter plot with my test set
    """
    plt.figure()
    plt.scatter(y_test, y_pred, label="Test", alpha=0.3, s=10)
    mn = float(min(y_test.min(), y_pred.min()))
    mx = float(max(y_test.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "-r", label="Ideal (y=x)")
    plt.title("Scatter plot (True vs Predicted)")
    plt.xlabel("True F (N)")
    plt.ylabel("Predicted F (N)")
    plt.legend()
    plt.show()
    return r2_score(y_test, y_pred)


# CAMBIO: volver a F (desescalar y exp)
y_pred_scaled = model.predict(x_test_scaled).flatten()
y_pred_scaled = np.array(y_pred_scaled).reshape(-1,1)

y_pred_log = y_scaler.inverse_transform(y_pred_scaled)  # CAMBIO
y_pred = np.exp(y_pred_log)                             # CAMBIO: volver a F (N)

# y_test_log ya lo tenemos
y_true = np.exp(y_test_log)                             # CAMBIO: F real (N)

plot_training_curves(history)
r2 = plot_scatter(y_true, y_pred)


def further_evaluation(y_test, y_pred, r2):
    """
    Calculate all the metrics
    """
    metrics = {
        "mae" : mean_absolute_error(y_test, y_pred),
        "mse" : mean_squared_error(y_test, y_pred),
        "r2"  : r2,
    }
    return metrics 


metrics = further_evaluation(y_true, y_pred, r2)
print(metrics)
