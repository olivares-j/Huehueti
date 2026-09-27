"""Residual calibration diagnostics for the heteroscedastic PARSEC ANN.

This module deliberately has a different name from the training code so that
the diagnostic can be compared independently with the training pipeline.
"""

import os
import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mlp_model_probabilistic import create_custom_model


SIGMA_FLOOR = 1.0e-6


def reconstruct_model(mlp, activation_layers="sigmoid", activation_output="linear"):
    """Reconstruct a trained heteroscedastic ANN from a saved mlp dictionary."""
    n_features = len(mlp["features"])
    n_targets = len(mlp["targets"])

    model = create_custom_model(
        input_shape=n_features,
        output_shape=n_targets,
        num_layers=mlp["num_layers"],
        size_layers=mlp["size_layers"],
        activation_layers=activation_layers,
        activation_output=activation_output,
        seed=mlp["seed"],
    )
    model.set_weights(mlp["weights"])
    return model


def predict_mu_sigma(model, x):
    """Return mean photometry and positive ANN uncertainty."""
    pred = np.asarray(model.predict(x, verbose=0))
    n_targets = pred.shape[1] // 2
    mu = pred[:, :n_targets]
    sigma = np.maximum(pred[:, n_targets:], SIGMA_FLOOR)
    return mu, sigma


def calibration_statistics(y_true, mu, sigma):
    """Calculate per-band residual and Gaussian calibration diagnostics."""
    residual = mu - y_true
    z = residual / sigma

    rows = []
    for b in range(y_true.shape[1]):
        zb = z[:, b]
        rows.append({
            "band": b,
            "bias_mag": np.mean(residual[:, b]),
            "rmse_mag": np.sqrt(np.mean(residual[:, b] ** 2)),
            "mean_sigma_mag": np.mean(sigma[:, b]),
            "rms_z": np.sqrt(np.mean(zb ** 2)),
            "mean_z": np.mean(zb),
            "std_z": np.std(zb, ddof=1),
            "coverage_1sigma": np.mean(np.abs(zb) <= 1.0),
            "coverage_2sigma": np.mean(np.abs(zb) <= 2.0),
            "coverage_3sigma": np.mean(np.abs(zb) <= 3.0),
            "median_abs_z": np.median(np.abs(zb)),
        })

    return pd.DataFrame(rows), residual, z


def make_calibration_plots(dir_seed, df, residual, z, sigma, targets, log_age, log_l):
    """Create diagnostic plots without imposing a color/style scheme."""
    n_targets = len(targets)

    fig, axes = plt.subplots(n_targets, 1, figsize=(9, 4 * n_targets), squeeze=False)
    for b, target in enumerate(targets):
        ax = axes[b, 0]
        ax.scatter(sigma[:, b], np.abs(residual[:, b]), s=5, alpha=0.25)
        xx = np.linspace(np.nanmin(sigma[:, b]), np.nanmax(sigma[:, b]), 200)
        ax.plot(xx, xx, label="|residual| = sigma")
        ax.set_xlabel("Predicted $\\sigma$ [mag]")
        ax.set_ylabel(r"|residual| [mag]")
        ax.set_title(target)
        ax.legend()
    fig.tight_layout()
    fig.savefig(dir_seed+"residual_calibration_sigma.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(n_targets, 1, figsize=(9, 4 * n_targets), squeeze=False)
    for b, target in enumerate(targets):
        ax = axes[b, 0]
        ax.hist(z[:, b], bins=50, density=True)
        ax.axvline(-1, linestyle="--")
        ax.axvline(1, linestyle="--")
        ax.axvline(0, linestyle="-")
        ax.set_xlabel("standardized residual $z=(\\mu-y)/\\sigma$")
        ax.set_ylabel("density")
        ax.set_title(target)
    fig.tight_layout()
    fig.savefig(dir_seed+"residual_calibration_z.png", dpi=200)
    plt.close(fig)

    # Calibration as a function of the two ANN inputs.
    for b, target in enumerate(targets):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sc = axes[0].scatter(log_age, log_l, c=z[:, b], s=8)
        axes[0].set_xlabel("logAge")
        axes[0].set_ylabel("logL")
        axes[0].set_title(target + ": standardized residual")
        fig.colorbar(sc, ax=axes[0], label="z")

        sc = axes[1].scatter(log_age, log_l, c=sigma[:, b], s=8)
        axes[1].set_xlabel("logAge")
        axes[1].set_ylabel("logL")
        axes[1].set_title(target + ": predicted sigma [mag]")
        fig.colorbar(sc, ax=axes[1], label="sigma [mag]")
        fig.tight_layout()
        fig.savefig(dir_seed+"residual_calibration_domain_{0}.png".format(target), dpi=200)
        plt.close(fig)


def run_calibration(
    dir_seed,
    df_iso,
    file_mlp,
    validation_split=0.2,
    seed_split=0,
    max_label=1,
    age_range=None,
    activation_layers="sigmoid",
):
    """Run calibration on exactly the same held-out split used by training."""
    with open(file_mlp, "rb") as f:
        mlp = dill.load(f)

    features = mlp["features"]
    targets = mlp["targets"]

    # Reproduce the training transformation and split exactly.
    x = df_iso.loc[:, features].copy()
    y = df_iso.loc[:, targets].copy()

    for col in features:
        x[col] = (x[col] - mlp["mu_transform"][col]) / mlp["sd_transform"][col]

    _, x_val, _, y_val = train_test_split(
        x, y, test_size=validation_split, random_state=seed_split
    )

    model = reconstruct_model(mlp, activation_layers=activation_layers)
    mu, sigma = predict_mu_sigma(model, x_val.to_numpy())

    stats, residual, z = calibration_statistics(y_val.to_numpy(), mu, sigma)

    # Add human-readable band names.
    stats["band"] = targets
    stats.to_csv("residual_calibration_statistics.csv", index=False)

    make_calibration_plots(
        dir_seed,
        stats,
        residual,
        z,
        sigma,
        targets,
        x_val["logAge"].to_numpy() * mlp["sd_transform"]["logAge"] + mlp["mu_transform"]["logAge"],
        x_val["logL"].to_numpy() * mlp["sd_transform"]["logL"] + mlp["mu_transform"]["logL"],
    )

    return stats


if __name__ == "__main__":
    raise SystemExit(
        "Import run_calibration() and provide the same PARSEC input file and "
        "saved probabilistic mlp file used by the training run."
    )
