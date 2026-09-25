'''
Copyright 2023 Javier Olivares Romero

This file is part of Huehueti.

    Kalkayotl is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PyAspidistra is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PyAspidistra.  If not, see <http://www.gnu.org/licenses/>.
'''

#------------ LOAD LIBRARIES -------------------
import os
import sys
import numpy as np
import pandas as pn

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


#======================= Configuration ================================

# age_range = "15-25Myr"
# age_range = "15-220Myr"
# age_range = "20-220Myr"
age_range = "200-600Myr"
# age_range = "600-1000Myr"

# age_step = 0.025
# age_step = 0.05
# age_step = 0.1
age_step = 0.5
# age_step = 1

epochs = 500
# epochs = 1000

# trials = 100
trials = 50

experiment = (
    "Optuna_InverseTimeDecay_epochs_{0:1.0e}_trials_{1}_{2}myr"
    .format(epochs, trials, age_step)
)


dir_base = (
    "/home/jolivares/Repos/Huehueti@phanocles/"
    "validation/synthetic/PARSEC/{0}/"
    .format(age_range)
)

dir_fig = (
    "/home/jolivares/Dropbox/MisArticulos/"
    "BayesianAges/Isochrones/Method/Figures/{0}/"
    .format(age_range)
)

dir_fig += experiment
os.makedirs(dir_fig, exist_ok=True)


models = ["binaries"]


if age_range == "15-25Myr":
    list_of_ages = list(range(15, 27, 2))

elif age_range == "20-220Myr":
    list_of_ages = list(range(20, 240, 20))

elif age_range == "200-600Myr":
    list_of_ages = list(sum([[210], list(range(250, 600, 50)), [590]], []))

elif age_range == "600-1000Myr":
    list_of_ages = list(range(600, 1100, 100))

else:
    sys.exit("Undefined age range")


list_of_distances = [100]
list_of_n_stars = [15,30,50]
list_of_seeds = [0, 1, 2, 3]


# Main switches
do_process = True
do_plt_grp = True
do_plt_src = True
do_plt_bnr = True


file_data = dir_base + experiment + ".h5"

file_plt_grp = dir_fig + "/{0}-{1}.png"
file_plt_src = dir_fig + "/{0}-{1}.png"

base_obs_grp = "{0}/{1}/" + experiment + "/{2}/Global_statistics.csv"
base_obs_src = "{0}/{1}/" + experiment + "/{2}/Sources_statistics.csv"
base_syn_src = "{0}/{1}/inputs/{2}.csv"

base_name = "a{0:d}_d{1:d}_n{2:d}_s{3:d}"


#======================= Statistics ====================================

obs_grp_columns = [
    "Parameter",
    "mean",
    "sd",
    "hdi_2.5%",
    "hdi_97.5%",
    "r_hat",
    "ess_bulk",
    "ess_tail"
]

requested_src_parameters = ["mass","mass_secondary","mass_ratio"]

# Group-level diagnostics
sts_grp = [
    {"key": "err", "name": "Error [%]", "ylim": [-10, 10]},
    {"key": "unc", "name": "Uncertainty [%]", "ylim": [0, 5]},
    {"key": "crd", "name": "Credibility [%]", "ylim": [0, 101]},
]


# Source-level diagnostics
sts_src = [
    {"key": "err", "name": "Error [%]", "ylim": [-50, 50]},
    {"key": "unc", "name": "Uncertainty [%]", "ylim": [0, 50]},
    {"key": "crd", "name": "Credibility [%]", "ylim": [0, 101]},
]


#======================= Processing ====================================

if do_process:

    dfs_grp = []
    dfs_src = []

    for model in models:

        print(40 * "+" + " " + model + " " + 40 * "+")

        for age in list_of_ages:

            for distance in list_of_distances:

                for n_stars in list_of_n_stars:

                    for seed in list_of_seeds:

                        name = base_name.format(
                            age,
                            distance,
                            n_stars,
                            seed
                        )

                        print(
                            40 * "-" +
                            " " + name + " " +
                            40 * "-"
                        )

                        #------------- Files --------------------------

                        file_obs_grp = base_obs_grp.format(
                            dir_base,
                            model,
                            name
                        )

                        file_obs_src = base_obs_src.format(
                            dir_base,
                            model,
                            name
                        )

                        file_syn_src = base_syn_src.format(
                            dir_base,
                            model,
                            name
                        )

                        #------------------------------------------------

                        #------------- Observed group values ------------

                        df_obs_grp = pn.read_csv(
                            file_obs_grp,
                            usecols=obs_grp_columns
                        )

                        df_obs_grp.set_index(
                            "Parameter",
                            inplace=True
                        )

                        #------------------------------------------------

                        #------------- Observed source values -----------

                        # Read all available source-level parameters.
                        df_obs_src_raw = pn.read_csv(file_obs_src)

                        if "source_id" not in df_obs_src_raw.columns:
                            raise RuntimeError(
                                "source_id not found in " + file_obs_src
                            )

                        if "statistic" not in df_obs_src_raw.columns:
                            raise RuntimeError(
                                "statistic not found in " + file_obs_src
                            )

                        # All parameters except source_id/statistic.
                        source_parameters = [
                            col for col in df_obs_src_raw.columns
                            if col not in ["source_id", "statistic"]
                        ]

                        if len(source_parameters) == 0:
                            raise RuntimeError(
                                "No source-level parameters found in "
                                + file_obs_src
                            )

                        df_obs_src = df_obs_src_raw.set_index(
                            ["source_id", "statistic"]
                        )

                        df_obs_src = df_obs_src[
                            source_parameters
                        ]

                        df_obs_src = df_obs_src.unstack()

                        #------------------------------------------------

                        #------------- True source values ---------------

                        df_syn_src_raw = pn.read_csv(file_syn_src)

                        if "source_id" not in df_syn_src_raw.columns:
                            raise RuntimeError(
                                "source_id not found in " + file_syn_src
                            )

                        # Only process parameters available in both
                        # posterior statistics and synthetic catalogue.
                        true_parameters = [
                            par for par in source_parameters
                            if par in df_syn_src_raw.columns
                        ]

                        if len(true_parameters) == 0:
                            raise RuntimeError(
                                "No common source parameters found "
                                "between posterior and synthetic data:\n"
                                + file_obs_src + "\n" +
                                file_syn_src
                            )

                        df_true_src = df_syn_src_raw[
                            ["source_id"] + true_parameters
                        ].copy()

                        df_true_src.set_index(
                            "source_id",
                            inplace=True
                        )

                        df_true_src.columns = pn.MultiIndex.from_product(
                            [
                                true_parameters,
                                ["true"]
                            ]
                        )

                        df_true_src.replace(
                            to_replace=0.0,
                            value=np.nan,
                            inplace=True
                        )

                        #------------------------------------------------

                        #------------- Join source data -----------------

                        df_src = pn.merge(
                            left=df_obs_src,
                            right=df_true_src,
                            left_index=True,
                            right_index=True
                        )

                        #------------------------------------------------

                        #------------- Convergence ----------------------

                        if any(
                            df_obs_grp["r_hat"] > 1.05
                        ):

                            par = df_obs_grp.loc[
                                df_obs_grp["r_hat"] > 1.05
                            ]

                            print(
                                "WARNING: Convergence issues at:"
                            )

                            print(par)

                            if any(
                                df_obs_grp["r_hat"] > 1.5
                            ):

                                par = df_obs_grp.loc[
                                    df_obs_grp["r_hat"] > 1.5
                                ]

                                print(
                                    "Error: Convergence issues at:"
                                )

                                print(par)
                                print(300 * "<")

                        #------------------------------------------------

                        #------------- Group true value -----------------

                        df_true_grp = pn.DataFrame.from_dict(
                            data={"age": age},
                            orient="index",
                            columns=["true"]
                        )

                        df_true_grp.index.name = "Parameter"

                        #------------------------------------------------

                        #------------- Group join ------------------------

                        df_grp = pn.merge(
                            left=df_true_grp,
                            right=df_obs_grp,
                            left_index=True,
                            right_index=True
                        )

                        #------------------------------------------------

                        #------------- Source diagnostics ---------------

                        # df_obs_src has MultiIndex columns:
                        #   (Parameter, statistic)
                        #
                        # Stack Parameter so that each row corresponds to
                        # one source and one inferred parameter.

                        df_obs_long = df_obs_src.stack(
                            level=0,
                            future_stack=True
                        )

                        df_obs_long.index.names = [
                            "source_id",
                            "Parameter"
                        ]

                        # True values have the same parameter structure.
                        df_true_long = df_true_src.stack(
                            level=0,
                            future_stack=True
                        )

                        df_true_long.index.names = [
                            "source_id",
                            "Parameter"
                        ]

                        # Combine posterior statistics and true values.
                        df_src = pn.merge(
                            left=df_obs_long,
                            right=df_true_long,
                            left_index=True,
                            right_index=True
                        )

                        # Keep only parameters for which a true value exists.
                        df_src = df_src.loc[
                            df_src.index.get_level_values(
                                "Parameter"
                            ).isin(true_parameters)
                        ]

                        #------------------------------------------------
                        # Calculate diagnostics for every parameter.

                        df_src["err"] = (
                            100.0 *
                            (df_src["mean"] - df_src["true"]) /
                            df_src["true"]
                        )

                        df_src["unc"] = (
                            100.0 *
                            df_src["sd"] /
                            np.abs(df_src["true"])
                        )

                        df_src["crd"] = (
                            100.0 *
                            (
                                (df_src["true"] >= df_src["hdi_2.5%"]) &
                                (df_src["true"] <= df_src["hdi_97.5%"])
                            )
                        )

                        #------------------------------------------------

                        df_src = df_src[
                            ["err", "unc", "crd"]
                        ]

                        df_src = df_src.reset_index()

                        #------------------------------------------------
                        # Metadata.

                        df_src["Model"] = model
                        df_src["n_stars"] = n_stars
                        df_src["distance"] = distance
                        df_src["seed"] = seed
                        df_src["age"] = age

                        # Keep a stable index for HDF storage.
                        df_src.set_index(
                            [
                                "Model",
                                "age",
                                "distance",
                                "n_stars",
                                "seed"
                            ],
                            inplace=True
                        )

                        #------------------------------------------------
                        # Group-level diagnostics.

                        for tmp in [df_grp]:

                            tmp["err"] = tmp.apply(
                                lambda x:
                                100.0 *
                                (x["mean"] - x["true"]) /
                                x["true"],
                                axis=1
                            )

                            tmp["unc"] = tmp.apply(
                                lambda x:
                                100.0 *
                                (x["sd"] / np.abs(x["true"])),
                                axis=1
                            )

                            tmp["crd"] = tmp.apply(
                                lambda x:
                                100.0 *
                                (
                                    (x["true"] >= x["hdi_2.5%"]) &
                                    (x["true"] <= x["hdi_97.5%"])
                                ),
                                axis=1
                            )

                        df_grp = df_grp[
                            [
                                "err",
                                "unc",
                                "crd",
                                "r_hat",
                                "ess_bulk",
                                "ess_tail"
                            ]
                        ].copy()

                        # Group metadata.

                        df_grp["Model"] = model
                        df_grp["n_stars"] = n_stars
                        df_grp["distance"] = distance
                        df_grp["seed"] = seed
                        df_grp["age"] = age

                        df_grp.reset_index(
                            inplace=True,
                            drop=True
                        )

                        df_grp.set_index(
                            [
                                "Model",
                                "age",
                                "distance",
                                "n_stars",
                                "seed"
                            ],
                            inplace=True
                        )

                        #------------------------------------------------

                        dfs_grp.append(df_grp)
                        dfs_src.append(df_src)

    #------------ Concatenate ------------------------------------------

    df_grp = pn.concat(
        dfs_grp,
        ignore_index=False
    )

    df_src = pn.concat(
        dfs_src,
        ignore_index=False
    )

    #------------ Save -------------------------------------------------

    df_grp.to_hdf(
        file_data,
        key="df_grp"
    )

    df_src.to_hdf(
        file_data,
        key="df_src"
    )


#======================= Group-level plots =============================

if do_plt_grp:

    print("Plotting group-level parameters")

    df_grp = pn.read_hdf(
        file_data,
        key="df_grp"
    )

    df_grp.reset_index(inplace=True)

    for st in sts_grp:

        fg = sns.relplot(
            data=df_grp,
            x="age",
            y=st["key"],
            row="Model",
            style="n_stars",
            hue="distance",
            kind="line",
            palette="tab10",
            facet_kws={"margin_titles": True},
            legend="full",
            height=3.0,
            aspect=1.5
        )

        fg.set_xlabels("Age [Myr]")
        fg.set_ylabels(st["name"])
        fg.set_titles(
            row_template="Model: {row_name}"
        )

        fg.set(
            ylim=st["ylim"]
        )

        sns.move_legend(
            fg,
            loc="lower center",
            bbox_to_anchor=(.5, 1),
            ncol=2
        )

        plt.subplots_adjust(
            wspace=0.1
        )

        fg.savefig(
            file_plt_grp.format("Age",st["key"]),
            bbox_inches="tight",
            dpi=300
        )

        plt.close()


#======================= Source-level plots ============================

if do_plt_src:

    print("Plotting source-level parameters")

    df_src = pn.read_hdf(
        file_data,
        key="df_src"
    )

    df_src.reset_index(inplace=True)

    # All source-level parameters stored during processing.
    current_parameters = sorted(
        df_src["Parameter"].unique()
    )

    print(
        "Source-level parameters:",
        requested_src_parameters
    )

    # Produce one independent PNG for every parameter and diagnostic.
    for parameter in requested_src_parameters:

        assert parameter in current_parameters, "Error, requestd parameter {0} not in stored file!".format(parameter)

        df_parameter = df_src.loc[
            df_src["Parameter"] == parameter
        ].copy()

        for st in sts_src:

            fg = sns.relplot(
                data=df_parameter,
                x="age",
                y=st["key"],
                row="Model",
                style="n_stars",
                hue="distance",
                kind="line",
                palette="tab10",
                facet_kws={"margin_titles": True},
                legend="full",
                height=3.0,
                aspect=1.5
            )

            fg.set_xlabels("Age [Myr]")
            fg.set_ylabels(st["name"])

            fg.set_titles(
                row_template="Model: {row_name}"
            )

            fg.set(
                ylim=st["ylim"]
            )

            sns.move_legend(
                fg,
                loc="lower center",
                bbox_to_anchor=(.5, 1),
                ncol=2
            )

            plt.subplots_adjust(
                wspace=0.1
            )

            filename = file_plt_src.format(
                parameter,
                st["key"]
            )

            fg.savefig(
                filename,
                bbox_inches="tight",
                dpi=300
            )

            plt.close()

            print(
                "  saved:",
                filename
            )


#======================= Binary diagnostics ============================

if do_plt_bnr:

    print("Plotting binary diagnostics")

    df_src = pn.read_hdf(
        file_data,
        key="df_src"
    )

    df_src.reset_index(inplace=True)

    # The binary diagnostics require the secondary mass parameter.
    if "mass_secondary" not in df_src["Parameter"].unique():

        print(
            "WARNING: mass_secondary not found in source-level "
            "parameters. Binary diagnostics skipped."
        )

    else:

        #---------------------------------------------------------------
        # Read the true mass ratio from the synthetic catalogues.

        q_frames = []

        for model in models:

            for age in list_of_ages:

                for distance in list_of_distances:

                    for n_stars in list_of_n_stars:

                        for seed in list_of_seeds:

                            name = base_name.format(
                                age,
                                distance,
                                n_stars,
                                seed
                            )

                            file_syn_src = base_syn_src.format(
                                dir_base,
                                model,
                                name
                            )

                            df_syn = pn.read_csv(
                                file_syn_src
                            )

                            # q_actual is preferred because it represents
                            # the actual ratio after model-grid selection.
                            if "mass_ratio" in df_syn.columns:
                                q_column = "mass_ratio"

                            elif "mass_ratio_requested" in df_syn.columns:
                                q_column = "mass_ratio_requested"

                            else:
                                print(
                                    "WARNING: no mass_ratio or mass_ratio_requested "
                                    "found in " + file_syn_src
                                )
                                continue

                            if "source_id" not in df_syn.columns:
                                continue

                            df_q_tmp = df_syn[
                                ["source_id", q_column]
                            ].copy()

                            df_q_tmp.rename(
                                columns={
                                    q_column: "q_true"
                                },
                                inplace=True
                            )

                            df_q_tmp["Model"] = model
                            df_q_tmp["age"] = age
                            df_q_tmp["distance"] = distance
                            df_q_tmp["n_stars"] = n_stars
                            df_q_tmp["seed"] = seed

                            q_frames.append(
                                df_q_tmp
                            )

        if len(q_frames) == 0:

            print(
                "WARNING: no mass-ratio information found. "
                "Binary diagnostics skipped."
            )

        else:

            df_q = pn.concat(
                q_frames,
                ignore_index=True
            )

            #-----------------------------------------------------------
            # Select secondary-mass diagnostics.

            df_m2 = df_src.loc[
                df_src["Parameter"] == "mass_secondary"
            ].copy()

            df_m2 = pn.merge(
                df_m2,
                df_q,
                on=[
                    "Model",
                    "age",
                    "distance",
                    "n_stars",
                    "seed",
                    "source_id"
                ],
                how="inner"
            )

            df_m2 = df_m2.loc[
                np.isfinite(df_m2["q_true"]) &
                np.isfinite(df_m2["err"])
            ].copy()

            #-----------------------------------------------------------
            # 1. Relative M2 error versus true q, colour-coded by age.

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            scatter = ax.scatter(
                df_m2["q_true"],
                df_m2["err"],
                c=df_m2["age"],
                cmap="viridis",
                s=35,
                alpha=0.75,
                edgecolors="none"
            )

            ax.axhline(
                0.0,
                linestyle="--",
                linewidth=1.0
            )

            ax.set_xlabel(
                r"$q_{\rm true}$"
            )

            ax.set_ylabel(
                r"$100(M_{2,\rm inf}-M_{2,\rm true})/"
                r"M_{2,\rm true}$ [\%]"
            )

            ax.set_title(
                "Secondary-mass recovery"
            )

            cbar = fig.colorbar(
                scatter,
                ax=ax
            )

            cbar.set_label(
                "Age [Myr]"
            )

            ax.grid(
                alpha=0.2
            )

            fig.tight_layout()

            filename = (
                dir_fig +
                "/binary-M2-error-vs-q_true.png"
            )

            fig.savefig(
                filename,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            #-----------------------------------------------------------
            # 2. Same diagnostic separately for each number of sources.

            for n_stars in sorted(
                df_m2["n_stars"].unique()
            ):

                df_plot = df_m2.loc[
                    df_m2["n_stars"] == n_stars
                ]

                fig, ax = plt.subplots(
                    figsize=(7, 5)
                )

                scatter = ax.scatter(
                    df_plot["q_true"],
                    df_plot["err"],
                    c=df_plot["age"],
                    cmap="viridis",
                    s=35,
                    alpha=0.75,
                    edgecolors="none"
                )

                ax.axhline(
                    0.0,
                    linestyle="--",
                    linewidth=1.0
                )

                ax.set_xlabel(
                    r"$q_{\rm true}$"
                )

                ax.set_ylabel(
                    r"$100(M_{2,\rm inf}-M_{2,\rm true})/"
                    r"M_{2,\rm true}$ [\%]"
                )

                ax.set_title(
                    "Secondary-mass recovery "
                    + f"({n_stars} sources)"
                )

                cbar = fig.colorbar(
                    scatter,
                    ax=ax
                )

                cbar.set_label(
                    "Age [Myr]"
                )

                ax.grid(
                    alpha=0.2
                )

                fig.tight_layout()

                filename = (
                    dir_fig +
                    f"/binary-M2-error-vs-q_true-n{n_stars:d}.png"
                )

                fig.savefig(
                    filename,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close(fig)

                #-------------------------------------------------------
                # 3. Binned median and 16-84% interval versus q_true.

                df_bin = df_plot.copy()

                # Fixed q bins provide the same x-axis for all ages.
                q_edges = np.linspace(
                    0.0,
                    1.0,
                    11
                )

                df_bin["q_bin"] = pn.cut(
                    df_bin["q_true"],
                    bins=q_edges,
                    include_lowest=True
                )

                grouped = df_bin.groupby(
                    "q_bin",
                    observed=True
                )["err"]

                summary = grouped.agg(
                    median="median",
                    q16=lambda x: np.percentile(x, 16),
                    q84=lambda x: np.percentile(x, 84),
                    count="count"
                )

                summary["q"] = [
                    interval.mid
                    for interval in summary.index
                ]

                summary = summary.loc[
                    summary["count"] > 0
                ]

                fig, ax = plt.subplots(
                    figsize=(7, 5)
                )

                ax.plot(
                    summary["q"],
                    summary["median"],
                    marker="o",
                    linewidth=1.5,
                    label="Median"
                )

                ax.fill_between(
                    summary["q"],
                    summary["q16"],
                    summary["q84"],
                    alpha=0.25,
                    label="16–84%"
                )

                ax.axhline(
                    0.0,
                    linestyle="--",
                    linewidth=1.0
                )

                ax.set_xlabel(
                    r"$q_{\rm true}$"
                )

                ax.set_ylabel(
                    r"$100(M_{2,\rm inf}-M_{2,\rm true})/"
                    r"M_{2,\rm true}$ [\%]"
                )

                ax.set_title(
                    "Secondary-mass recovery "
                    + f"({n_stars} sources)"
                )

                ax.legend()

                ax.grid(
                    alpha=0.2
                )

                fig.tight_layout()

                filename = (
                    dir_fig +
                    f"/binary-M2-error-vs-q_true-binned"
                    f"-n{n_stars:d}.png"
                )

                fig.savefig(
                    filename,
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close(fig)

                print(
                    "  saved:",
                    filename
                )

            #-----------------------------------------------------------
            # 4. Overall binned diagnostic, combining all N.

            df_bin = df_m2.copy()

            q_edges = np.linspace(
                0.0,
                1.0,
                11
            )

            df_bin["q_bin"] = pn.cut(
                df_bin["q_true"],
                bins=q_edges,
                include_lowest=True
            )

            grouped = df_bin.groupby(
                "q_bin",
                observed=True
            )["err"]

            summary = grouped.agg(
                median="median",
                q16=lambda x: np.percentile(x, 16),
                q84=lambda x: np.percentile(x, 84),
                count="count"
            )

            summary["q"] = [
                interval.mid
                for interval in summary.index
            ]

            summary = summary.loc[
                summary["count"] > 0
            ]

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            ax.plot(
                summary["q"],
                summary["median"],
                marker="o",
                linewidth=1.5,
                label="Median"
            )

            ax.fill_between(
                summary["q"],
                summary["q16"],
                summary["q84"],
                alpha=0.25,
                label="16–84%"
            )

            ax.axhline(
                0.0,
                linestyle="--",
                linewidth=1.0
            )

            ax.set_xlabel(
                r"$q_{\rm true}$"
            )

            ax.set_ylabel(
                r"$100(M_{2,\rm inf}-M_{2,\rm true})/"
                r"M_{2,\rm true}$ [\%]"
            )

            ax.set_title(
                "Secondary-mass recovery"
            )

            ax.legend()

            ax.grid(
                alpha=0.2
            )

            fig.tight_layout()

            filename = (
                dir_fig +
                "/binary-M2-error-vs-q_true-binned.png"
            )

            fig.savefig(
                filename,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            print(
                "  saved:",
                filename
            )

#-----------------------------------------------------------------------