import sys
from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestRegressor


def print_shape(name, arr):
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")


def load_one_simulation(np, file_path, k):
    data = np.load(file_path)

    x = data["x"][:, :, k]
    t = data["t"][:, k]
    mu0 = data["mu0"][:, k]
    mu1 = data["mu1"][:, k]
    yadd = data["yadd"]
    yf = data["yf"][:, k]
    ycf = data["ycf"][:, k]
    ate = data["ate"]

    print("\n--- Selected simulation dimensions ---")
    print_shape("ate", ate)
    print_shape("mu1", mu1)
    print_shape("mu0", mu0)
    print_shape("yadd", yadd)
    print_shape("yf", yf)
    print_shape("ycf", ycf)
    print_shape("t", t)
    print_shape("x", x)

    return x, t, mu0, mu1, yf, ycf


def random_forest_train(RandomForestRegressor, X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def random_forest_predict(model, X):
    return model.predict(X)


def main():
    # Basic fixed settings.
    k = 0 #Ordinal index of simulation to load (0-999).
    
    if k < 0 or k > 999:
        print("Error: k must be between 0 and 999.")
        sys.exit(1)

    base = Path(__file__).resolve().parent / "Simulated_outcomes"
    train_path = base / "ihdp_npci_1-1000.train.npz"

    if not train_path.exists():
        print("Could not find train file:")
        print(train_path)
        sys.exit(1)

    test_path = base / "ihdp_npci_1-1000.test.npz"

    if not test_path.exists():
        print("Could not find test file:")
        print(test_path)
        sys.exit(1)

    x_train, t_train, mu0_train, mu1_train, yf_train, ycf_train = load_one_simulation(np, train_path, k)
    x_test, t_test, mu0_test, mu1_test, yf_test, ycf_test = load_one_simulation(np, test_path, k)

    # Reconstruct noisy potential outcomes from (yf, ycf, t):
    # If t=1, then yf=Y1_noisy and ycf=Y0_noisy.
    # If t=0, then yf=Y0_noisy and ycf=Y1_noisy.
    y1_noisy_train = np.where(t_train == 1, yf_train, ycf_train)
    y0_noisy_train = np.where(t_train == 1, ycf_train, yf_train)

    y1_noisy_test = np.where(t_test == 1, yf_test, ycf_test)
    y0_noisy_test = np.where(t_test == 1, ycf_test, yf_test)

    # Ground-truth CATE without noise (used only for evaluation).
    cate_real_test = mu1_test - mu0_test

    # Use only x as model input.
    X_train = x_train
    X_test = x_test

    print("\n--- Dataset summary ---")
    print("Train samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])
    print("Number of features:", X_train.shape[1])
    print("Treated in train:", int((t_train == 1).sum()))
    print("Control in train:", int((t_train == 0).sum()))
    print("Treated in test:", int((t_test == 1).sum()))
    print("Control in test:", int((t_test == 0).sum()))

    print("\n--- Outcome models ---")
    print("Samples for Y1_noisy model:", X_train.shape[0])
    print("Samples for Y0_noisy model:", X_train.shape[0])

    # Model 1: predicts outcome if patient is treated (Y1_noisy).
    model_y1 = random_forest_train(RandomForestRegressor, X_train, y1_noisy_train)

    # Model 0: predicts outcome if patient is not treated (Y0_noisy).
    model_y0 = random_forest_train(RandomForestRegressor, X_train, y0_noisy_train)
    
    # Then we can predict on the test set and compute CATE estimate as difference of predictions.
    y1_hat = random_forest_predict(model_y1, X_test)
    y0_hat = random_forest_predict(model_y0, X_test)
    cate_est = y1_hat - y0_hat

    # We take mean squared error (MSE) as our evaluation metric
    mse_y1_noisy = np.mean((y1_hat - y1_noisy_test) ** 2)
    mse_y0_noisy = np.mean((y0_hat - y0_noisy_test) ** 2)
    mse_cate_vs_real = np.mean((cate_est - cate_real_test) ** 2)

    # Compare with noisy CATE from test data.
    cate_noisy_test = y1_noisy_test - y0_noisy_test
    mse_cate_vs_noisy = np.mean((cate_est - cate_noisy_test) ** 2)

    factual_pred = np.where(t_test == 1, y1_hat, y0_hat)
    mse_factual = np.mean((factual_pred - yf_test) ** 2)

    print("\n--- Results ---")
    print("MSE y1_hat vs true Y1_noisy:", float(mse_y1_noisy))
    print("MSE y0_hat vs true Y0_noisy:", float(mse_y0_noisy))
    print("MSE CATE_EST vs CATE_REAL (mu1-mu0):", float(mse_cate_vs_real))
    print("MSE CATE_EST vs noisy CATE:", float(mse_cate_vs_noisy))
    print("MSE factual prediction vs yf:", float(mse_factual))

    print("First 5 true Y0_noisy:", y0_noisy_test[:5])
    print("First 5 true mu0:", mu0_test[:5])
    print("First 5 y0_hat:", y0_hat[:5])

    print("First 5 true Y1_noisy:", y1_noisy_test[:5])
    print("First 5 true mu1:", mu1_test[:5])
    print("First 5 y1_hat:", y1_hat[:5])
    print("First 5 CATE_EST:", cate_est[:5])
    print("First 5 CATE_REAL:", cate_real_test[:5])
 


if __name__ == "__main__":
    main()
