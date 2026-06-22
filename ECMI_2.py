import numpy as np

#SIMULATED DATASET

data = np.load("ihdp_npci_1-1000.train.npz")

X = data["x"][:, :, 0]
T = data["t"][:, 0]
Y = data["yf"][:, 0]
MU0 = data["mu0"][:, 0]
MU1 = data["mu1"][:, 0]
TAU = MU1 - MU0

#T-LEARNER

from sklearn.ensemble import RandomForestRegressor

X_treated = X[T == 1]
Y_treated = Y[T == 1]

X_control = X[T == 0]
Y_control = Y[T == 0]


rf_treated = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_control = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_treated.fit(X_treated, Y_treated)
rf_control.fit(X_control, Y_control)

mu1_hat = rf_treated.predict(X)
mu0_hat = rf_control.predict(X)

tau_hat = mu1_hat - mu0_hat

print("True ATE      :", TAU.mean())
print("Estimated ATE :", tau_hat.mean())

from sklearn.metrics import mean_squared_error

error_T = np.sqrt(
    mean_squared_error(
        TAU,
        tau_hat
    )
)

print("T-Learner Error=", error_T)

#S-LEARNER

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X_with_T = np.column_stack([X, T])

s_learner = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

s_learner.fit(X_with_T, Y)

X_treat_all = np.column_stack([X, np.ones(len(X))])
X_control_all = np.column_stack([X, np.zeros(len(X))])

mu1_hat_s = s_learner.predict(X_treat_all)
mu0_hat_s = s_learner.predict(X_control_all)

tau_hat_s = mu1_hat_s - mu0_hat_s

print("\nTrue ATE        :", TAU.mean())
print("S-Learner ATE   :", tau_hat_s.mean())

error_S = np.sqrt(mean_squared_error(TAU, tau_hat_s))
print("S-Learner Error=:", error_S)


#X-LEARNER

D1 = Y_treated - rf_control.predict(X_treated)
D0 = rf_treated.predict(X_control) - Y_control

tau_model_treated = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

tau_model_control = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

tau_model_treated.fit(
    X_treated,
    D1
)

tau_model_control.fit(
    X_control,
    D0
)

tau1_hat = tau_model_treated.predict(X)
tau0_hat = tau_model_control.predict(X)

tau_hat_x = (
    tau1_hat +
    tau0_hat
) / 2

print("\nTrue ATE      :", TAU.mean())
print("X-Learner ATE :", tau_hat_x.mean())

error_X = np.sqrt(
    mean_squared_error(
        TAU,
        tau_hat_x
    )
)

print("X-Learner Error:", error_X)

print("\nError comparison")
print("T-Learner :", error_T)
print("S-Learner :", error_S)
print("X-Learner :", error_X)