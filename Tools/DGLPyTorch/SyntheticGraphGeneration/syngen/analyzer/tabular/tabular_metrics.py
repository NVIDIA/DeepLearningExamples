# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import warnings
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dython.nominal import associations, numerical_encoding
from scipy import stats
from scipy.spatial import distance
from scipy.special import kl_div
from sklearn.decomposition import PCA

from syngen.utils.types import DataFrameType, ColumnType


warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
matplotlib._log.disabled = True
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


class TabularMetrics(object):
    def __init__(
        self,
        real: DataFrameType,
        fake: DataFrameType,
        categorical_columns: Optional[List] = [],
        nrows: Optional[int] = None,
        seed: Optional[int] = 123,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Args:
            real (DataFrameType): the original dataset
            fake (DataFrameType): the generated dataset
            categorical_columns (list): list of categorical columns in tabular data
            nrows (int): number of rows to use for evaluation (default: None), will use the minimum of real/fake data length
            seed (int): sets the random seed for reproducibility. (default: 123)
            verbose (bool): print intermediate results (default: False)
            debug (bool): debug mode (default: False)
        """
        assert all(c in fake.columns for c in real.columns) and len(
            real.columns
        ) == len(fake.columns), r"Real and fake have different columns."
        self.real = real
        self.fake = fake[real.columns]

        self.nrows = nrows
        self.seed = seed
        self.verbose = verbose
        self.debug = debug

        self.categorical_columns = categorical_columns
        self.numerical_columns = [
            column
            for column in real.columns
            if column not in categorical_columns
        ]
        # Make sure columns and their order are the same.
        if len(real.columns) == len(fake.columns):
            fake = fake[real.columns.tolist()]
        assert (
            real.columns.tolist() == fake.columns.tolist()
        ), "Columns in real and fake dataframe are not the same"

        # Make sure the number of samples is equal in both datasets.
        if nrows is None:
            self.nrows = min(len(self.real), len(self.fake))
        elif len(fake) >= nrows and len(real) >= nrows:
            self.nrows = nrows
        else:
            raise Exception(
                f"Make sure nrows < len(fake/real). len(real): {len(real)}, len(fake): {len(fake)}"
            )

        self.real = self.real.sample(self.nrows)
        self.fake = self.fake.sample(self.nrows)

        self.real.loc[:, self.categorical_columns] = (
            self.real.loc[:, self.categorical_columns]
            .fillna("[NAN]")
            .astype(str)
        )
        self.fake.loc[:, self.categorical_columns] = (
            self.fake.loc[:, self.categorical_columns]
            .fillna("[NAN]")
            .astype(str)
        )

        self.real.loc[:, self.numerical_columns] = self.real.loc[
            :, self.numerical_columns
        ].fillna(self.real[self.numerical_columns].mean())
        self.fake.loc[:, self.numerical_columns] = self.fake.loc[
            :, self.numerical_columns
        ].fillna(self.fake[self.numerical_columns].mean())

    def kl_divergence(self) -> float:
        def get_frequencies(real, synthetic):
            f_obs, f_exp = [], []
            real, synthetic = Counter(real), Counter(synthetic)
            for value in synthetic:
                if value not in real:
                    warnings.warn(
                        f"Unexpected value {value} in synthetic data."
                    )
                    real[value] += 1e-6  # Regularization to prevent NaN.

            for value in real:
                f_obs.append(synthetic[value] / sum(synthetic.values()))
            f_exp.append(real[value] / sum(real.values()))
            return f_obs, f_exp

        numerical_columns = self.numerical_columns
        # - continuous columns
        cont_scores = []
        for columns in combinations(numerical_columns, r=2):
            columns = list(columns)
            rd_cont = self.real[columns]
            rd_cont[pd.isna(rd_cont)] = 0.0
            rd_cont[pd.isna(rd_cont)] = 0.0
            column1, column2 = rd_cont.columns[:2]

            real, xedges, yedges = np.histogram2d(
                rd_cont[column1], rd_cont[column2]
            )
            fake, _, _ = np.histogram2d(
                self.fake[column1], self.fake[column2], bins=[xedges, yedges]
            )

            f_obs, f_exp = fake.flatten() + 1e-5, real.flatten() + 1e-5
            f_obs, f_exp = f_obs / np.sum(f_obs), f_exp / np.sum(f_exp)

            score = 1 / (1 + np.sum(kl_div(f_obs, f_exp)))
            cont_scores.append(score)

        # - discrete columns
        categorical_columns = self.categorical_columns
        cat_scores = []
        for columns in combinations(categorical_columns, r=2):
            columns = list(columns)
            real = self.real[columns].itertuples(index=False)
            fake = self.fake[columns].itertuples(index=False)

            f_obs, f_exp = get_frequencies(real, fake)
            score = 1 / (1 + np.sum(kl_div(f_obs, f_exp)))
            cat_scores.append(score)

        return np.nanmean(cont_scores + cat_scores)

    def correlation_correlation(
        self, comparison_metric: str = "pearsonr"
    ) -> float:
        """
        computes the column-wise correlation of each dataset, and outputs the
        `comparison_metric` score between the datasets.

        Args:
            comparison_metric (str): metric to be used to compare between the datasets
                see `scipy.stats`
        Returns:
            corr (float): correlation score
        """
        comparison_metric = getattr(stats, comparison_metric)
        total_metrics = pd.DataFrame()
        for ds_name in ["real", "fake"]:
            ds = getattr(self, ds_name)
            corr_df = associations(
                ds, nominal_columns=self.categorical_columns, nom_nom_assoc='theil', compute_only=True
            )
            values = corr_df['corr'].values
            values = values[~np.eye(values.shape[0], dtype=bool)].reshape(
                values.shape[0], -1
            )
            total_metrics[ds_name] = values.flatten()
        correlation_correlations = total_metrics
        corr, p = comparison_metric(
            total_metrics["real"], total_metrics["fake"]
        )
        if self.debug:
            print("\nColumn correlation between datasets:")
            print(total_metrics.to_string())
        return corr

    def statistical_correlation(self, comparison_metric="spearmanr") -> float:
        """
        computes correlation between basic statistics of each dataset for each column

        Args:
            comparison_metric (str): metric to be used to compare between the datasets
                see `scipy.stats`
        Returns:
            corr (float): correlation score

        """
        total_metrics = pd.DataFrame()
        comparison_metric = getattr(stats, comparison_metric)
        discrete_values = {
            c: self.real[c].unique() for c in self.categorical_columns
        }
        for ds_name in ["real", "fake"]:
            ds = getattr(self, ds_name)
            metrics = {}
            num_ds = ds.loc[:, self.numerical_columns]
            cat_ds = ds.loc[:, self.categorical_columns]
            for idx, value in num_ds.mean().items():
                metrics[f"mean_{idx}"] = value
            for idx, value in num_ds.median().items():
                metrics[f"median_{idx}"] = value
            for idx, value in num_ds.std().items():
                metrics[f"std_{idx}"] = value
            for idx, value in num_ds.var().items():
                metrics[f"variance_{idx}"] = value
            for cc in self.categorical_columns:
                cdf = ds[cc]
                v = cdf.value_counts(normalize=True)
                unique_vals = set(v.index)
                for d in discrete_values[cc]:
                    if d not in unique_vals:
                        metrics[f"count_{d}"] = 0.0
                    else:
                        metrics[f"count_{d}"] = v[d]
            total_metrics[ds_name] = metrics.values()

        total_metrics.index = metrics.keys()
        statistical_results = total_metrics

        if self.debug:
            print("\nBasic statistical attributes:")
            print(total_metrics.to_string())
        corr, p = comparison_metric(
            statistical_results["real"], statistical_results["fake"]
        )
        return corr

    def plot_cumsums(self, nr_cols=4, fname=None):
        """
        Plot the cumulative sums for all columns in the real and fake dataset.
        Height of each row scales with the length of the labels. Each plot contains the
        values of a real columns and the corresponding fake column.
        Args:
            fname: If not none, saves the plot with this file name.
        """
        nr_charts = len(self.real.columns)
        nr_rows = max(1, nr_charts // nr_cols)
        nr_rows = nr_rows + 1 if nr_charts % nr_cols != 0 else nr_rows

        max_len = 0
        # Increase the length of plots if the labels are long
        if not self.real.select_dtypes(include=["object"]).empty:
            lengths = []
            for d in self.real.select_dtypes(include=["object"]):
                lengths.append(
                    max(
                        [
                            len(x.strip())
                            for x in self.real[d].unique().tolist()
                        ]
                    )
                )
            max_len = max(lengths)

        row_height = 6 + (max_len // 30)
        fig, ax = plt.subplots(
            nr_rows, nr_cols, figsize=(16, row_height * nr_rows)
        )
        fig.suptitle("Cumulative Sums per feature", fontsize=16)
        if nr_rows == 1 and nr_cols == 1:
            axes = [ax]
        else:
            axes = ax.flatten()
        for i, col in enumerate(self.real.columns):
            r = self.real[col]
            f = self.fake.iloc[:, self.real.columns.tolist().index(col)]
            self.cdf(r, f, col, "Cumsum", ax=axes[i])
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if fname is not None:
            plt.savefig(fname)

        plt.show()

    def plot_mean_std(self, ax=None, fname=None) -> None:
        """
        Plot the means and standard deviations of each dataset.

        Args:
            ax: Axis to plot on. If none, a new figure is made.
            fname: If not none, saves the plot with this file name.
        """
        real = self.real
        fake = self.fake
        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(
                "Absolute Log Mean and STDs of numeric data\n", fontsize=16
            )

        ax[0].grid(True)
        ax[1].grid(True)
        real = real.select_dtypes(include=np.number).reset_index()
        fake = fake.select_dtypes(include=np.number).reset_index()
        real_mean = np.log(np.add(abs(real.mean()).values, 1e-5))
        fake_mean = np.log(np.add(abs(fake.mean()).values, 1e-5))
        min_mean = min(real_mean) - 1
        max_mean = max(real_mean) + 1
        line = np.arange(min_mean, max_mean)
        sns.lineplot(x=line, y=line, ax=ax[0])
        sns.scatterplot(x=real_mean, y=fake_mean, ax=ax[0])
        ax[0].set_title("Means of real and fake data")
        ax[0].set_xlabel("real data mean (log)")
        ax[0].set_ylabel("fake data mean (log)")

        real_std = np.log(np.add(real.std().values, 1e-5))
        fake_std = np.log(np.add(fake.std().values, 1e-5))
        min_std = min(real_std) - 1
        max_std = max(real_std) + 1
        line = np.arange(min_std, max_std)
        sns.lineplot(x=line, y=line, ax=ax[1])
        sns.scatterplot(x=real_std, y=fake_std, ax=ax[1])
        ax[1].set_title("Stds of real and fake data")
        ax[1].set_xlabel("real data std (log)")
        ax[1].set_ylabel("fake data std (log)")

        if fname is not None:
            plt.savefig(fname)

        if ax is None:
            plt.show()

    def convert_numerical(self, real, fake):
        """
        Convert categorical columns to numerical
        """
        for c in self.categorical_columns:
            if real[c].dtype == "object":
                real[c] = pd.factorize(real[c], sort=True)[0]
                fake[c] = pd.factorize(fake[c], sort=True)[0]
        return real, fake

    def cdf(
        self,
        real_data,
        fake_data,
        xlabel: str = "Values",
        ylabel: str = "Cumulative Sum",
        ax=None,
    ) -> None:
        """
        Plot continous density function on optionally given ax. If no ax, cdf is plotted and shown.
        Args:
            xlabel: Label to put on the x-axis
            ylabel: Label to put on the y-axis
            ax: The axis to plot on. If ax=None, a new figure is created.
        """

        x1 = np.sort(real_data)
        x2 = np.sort(fake_data)
        y = np.arange(1, len(real_data) + 1) / len(real_data)

        ax = ax if ax else plt.subplots()[1]

        axis_font = {"size": "14"}
        ax.set_xlabel(xlabel, **axis_font)
        ax.set_ylabel(ylabel, **axis_font)

        ax.grid()
        ax.plot(x1, y, marker="o", linestyle="none", label="Real", ms=8)
        ax.plot(x2, y, marker="o", linestyle="none", label="Fake", alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
        import matplotlib.ticker as mticker

        # If labels are strings, rotate them vertical
        if isinstance(real_data, pd.Series) and real_data.dtypes == "object":
            ticks_loc = ax.get_xticks()
            r_unique = real_data.sort_values().unique()
            if len(r_unique) > len(ticks_loc):
                import pdb; pdb.set_trace()
            ticks_loc = ticks_loc[: len(r_unique)]

            ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ax.set_xticklabels(r_unique, rotation="vertical")

        if ax is None:
            plt.show()

    def plot_correlation_difference(
        self,
        plot_diff: bool = True,
        cat_cols: list = None,
        annot=False,
        fname=None,
    ) -> None:
        """
        Plot the association matrices for the `real` dataframe, `fake` dataframe and plot the difference between them.
        Has support for continuous and categorical data types.
        All Object and Category dtypes are considered to be categorical columns if `cat_cols` is not passed.

        - Continuous - Continuous: Uses Pearson's correlation coefficient
        - Continuous - Categorical: Uses so called correlation ratio (https://en.wikipedia.org/wiki/Correlation_ratio) for both continuous - categorical and categorical - continuous.
        - Categorical - Categorical: Uses Theil's U, an asymmetric correlation metric for Categorical associations
        Args:
            plot_diff: Plot difference if True, else not
            cat_cols: List of Categorical columns
            boolean annot: Whether to annotate the plot with numbers indicating the associations.
        """
        real = self.real
        fake = self.fake
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        if cat_cols is None:
            cat_cols = real.select_dtypes(["object", "category"])
        if plot_diff:
            fig, ax = plt.subplots(1, 3, figsize=(24, 7))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(20, 8))

        real_corr = associations(
            real,
            nominal_columns=cat_cols,
            plot=False,
            nom_nom_assoc='theil',
            mark_columns=True,
            annot=annot,
            ax=ax[0],
            cmap=cmap,
        )["corr"]
        fake_corr = associations(
            fake,
            nominal_columns=cat_cols,
            plot=False,
            nom_nom_assoc='theil',
            mark_columns=True,
            annot=annot,
            ax=ax[1],
            cmap=cmap,
        )["corr"]

        if plot_diff:
            diff = abs(real_corr - fake_corr)
            sns.set(style="white")
            sns.heatmap(
                diff,
                ax=ax[2],
                cmap=cmap,
                vmax=0.3,
                square=True,
                annot=annot,
                center=0,
                linewidths=0.5,
                cbar_kws={"shrink": 0.5},
                fmt=".2f",
            )

        titles = (
            ["Real", "Fake", "Difference"] if plot_diff else ["Real", "Fake"]
        )
        for i, label in enumerate(titles):
            title_font = {"size": "18"}
            ax[i].set_title(label, **title_font)
        plt.tight_layout()

        if fname is not None:
            plt.savefig(fname)

        plt.show()

    def plot_pca(self, fname=None):
        """
        Plot the first two components of a PCA of real and fake data.
        Args:
            fname: If not none, saves the plot with this file name.
        """
        real, fake = self.convert_numerical(self.real, self.fake)

        pca_r = PCA(n_components=2)
        pca_f = PCA(n_components=2)

        real_t = pca_r.fit_transform(real)
        fake_t = pca_f.fit_transform(fake)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("First two components of PCA", fontsize=16)
        sns.scatterplot(ax=ax[0], x=real_t[:, 0], y=real_t[:, 1])
        sns.scatterplot(ax=ax[1], x=fake_t[:, 0], y=fake_t[:, 1])
        ax[0].set_title("Real data")
        ax[1].set_title("Fake data")

        if fname is not None:
            plt.savefig(fname)

        plt.show()

    def visual_evaluation(self, save_dir=None, **kwargs):
        """
        Plots mean, std, cumulative sum, correlation difference and PCA
        Args:
            save_dir: directory path to save images
            kwargs: any key word argument for matplotlib.
        """
        if save_dir is None:
            self.plot_mean_std()
            self.plot_cumsums()
            self.plot_correlation_difference(
                plot_diff=True, cat_cols=self.categorical_columns, **kwargs
            )
            self.plot_pca()
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            self.plot_mean_std(fname=save_dir / "mean_std.png")
            self.plot_cumsums(fname=save_dir / "cumsums.png")
            self.plot_correlation_difference(
                plot_diff=True,
                cat_cols=self.categorical_columns,
                fname=save_dir / "correlation_difference.png",
                **kwargs,
            )
            self.plot_pca(fname=save_dir / "pca.png")

    def evaluate(
        self, comparison_metric: str = "pearsonr"
    ) -> Dict[str, float]:
        """
        evaluate synthetic data

        Args:
            comparison_metric (str): metric to be used to compare between the datasets
                see `scipy.stats`

        Returns:
            results (dict<str, float>): dictionary containing computed metrics, <key> := metric_name, <value>:= score

        """
        statistical_correlation = self.statistical_correlation(
            comparison_metric
        )
        kl_divergence = self.kl_divergence()
        correlation_correlation = self.correlation_correlation()

        results = {
            "statistical_correlation": statistical_correlation,
            "kl_divergence": kl_divergence,
            "correlation_correlation": correlation_correlation,
        }

        return results


def dd_feat_heatmap(
    data,
    feat_name_col_info: Dict[str, ColumnType],
    src_col: str = "src",
    dst_col: str = "dst",
):
    src_degree = (
        data.groupby(src_col, as_index=False)
        .count()[[src_col, dst_col]]
        .rename(columns={dst_col: "src_degree"})
    )

    # - normalized src_degree
    src_degree_vals = src_degree["src_degree"].values
    normalized_src_degree = src_degree_vals / np.sum(src_degree_vals)
    src_degree.loc[:, "src_degree"] = normalized_src_degree

    # - normalized dst_degree
    dst_degree = (
        data.groupby(dst_col, as_index=False)
        .count()[[src_col, dst_col]]
        .rename(columns={src_col: "dst_degree"})
    )
    dst_degree_vals = dst_degree["dst_degree"].values
    normalized_dst_degree = dst_degree_vals / np.sum(dst_degree_vals)

    dst_degree.loc[:, "dst_degree"] = normalized_dst_degree

    # - merge
    data = data.merge(src_degree, how="outer", on=src_col)
    data = data.merge(dst_degree, how="outer", on=dst_col)

    # - normalize continuous columns
    for feat, col_info in feat_name_col_info.items():
        col_type = col_info["type"]
        min_ = col_info["min"]
        max_ = col_info["max"]
        if col_type == ColumnType.CONTINUOUS:
            vals = data[feat].values
            data.loc[:, feat] = (vals - min_) / (max_ - min_)

    # - plot heat maps
    def heat_map(x, y):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=10)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    heat_maps = []
    for feat in feat_name_col_info:
        heatmap, _ = heat_map(data["src_degree"].values, data[feat].values)
        heat_maps.append(heatmap)

    return heat_maps


def compute_dd_feat_js(
    real,
    fake,
    feat_name_col_info: Dict[str, ColumnType],
    src_col: str = "src",
    dst_col: str = "dst",
):

    col_info = {}
    for col_name, col_type in feat_name_col_info.items():
        if col_type == ColumnType.CONTINUOUS:
            min_ = real[col_name].min()
            max_ = real[col_name].max()
            col_info[col_name] = {"type": col_type, "min": min_, "max": max_}

        elif col_type == ColumnType.CATEGORICAL:
            # - none of the datsets align on categorical for now..
            pass

    real_heatmaps = dd_feat_heatmap(
        real, col_info, src_col=src_col, dst_col=dst_col
    )

    fake_heatmaps = dd_feat_heatmap(
        fake, col_info, src_col=src_col, dst_col=dst_col
    )

    heatmaps = list(zip(real_heatmaps, fake_heatmaps))
    score = 0.0
    for r, f in heatmaps:
        s = distance.jensenshannon(r, f, axis=1)  # - along feats
        np.nan_to_num(s, copy=False, nan=1.0)
        s = np.mean(s)
        score += s
    return score


def get_frequencies(real, synthetic):
    f_obs, f_exp = [], []
    real, synthetic = Counter(real), Counter(synthetic)
    for value in synthetic:
        if value not in real:
            warnings.warn(f"Unexpected value {value} in synthetic data.")
            real[value] += 1e-6  # Regularization to prevent NaN.

    for value in real:
        f_obs.append(synthetic[value] / sum(synthetic.values()))
        f_exp.append(real[value] / sum(real.values()))
    return f_obs, f_exp
