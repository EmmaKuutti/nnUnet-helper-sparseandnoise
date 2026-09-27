"""Create Nemenyi critical-difference diagrams from the project result tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare, studentized_range
import scikit_posthocs as sp


RAW_DATA_PATH = "segmentation_results_selected.csv"
RAW_DATA_DIRECTORY = Path(__file__).resolve().parent
STATS_DATA_PATH = "Classwise_Statistical_Results.xlsx"
OUTPUT_DIRECTORY = "critical_difference_diagrams"
ALPHA = 0.05
METRIC_DIRECTIONS = {"Dice": False, "HD95": True}


def read_table(path, **kwargs):
    """Read a CSV or Excel table using the same convention as table_inLAtex.py."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, **kwargs)
    return pd.read_excel(path, **kwargs)


def load_all_csv_results(directory):
    """Merge every segmentation CSV, retaining one copy of each dataset column."""
    directory = Path(directory)
    csv_paths = sorted(directory.glob("segmentation_results*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No segmentation_results*.csv files found in {directory}")

    merged = None
    known_columns = {"Case_ID", "Class"}
    for csv_path in csv_paths:
        table = pd.read_csv(csv_path)
        key_columns = ["Case_ID", "Class"]
        missing_keys = set(key_columns) - set(table.columns)
        if missing_keys:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing_keys)}")
        value_columns = [
            column
            for column in table.columns
            if column not in key_columns and column not in known_columns
        ]
        table = table[key_columns + value_columns]
        if table.duplicated(key_columns).any():
            raise ValueError(f"{csv_path} contains duplicate Case_ID/Class rows.")
        if merged is None:
            merged = table
        else:
            merged = merged.merge(table, on=key_columns, how="outer", validate="one_to_one")
        known_columns.update(value_columns)

    merged = merged.sort_values(["Case_ID", "Class"]).reset_index(drop=True)
    available_ids = {
        int(column.split("_")[0].replace("Dataset ", ""))
        for column in merged.columns
        if column.startswith("Dataset ") and "_" in column
    }
    missing_ids = sorted(set(range(1, 33)) - available_ids)
    if missing_ids:
        print(
            "Warning: no CSV columns were found for dataset(s): "
            + ", ".join(str(dataset_id) for dataset_id in missing_ids)
        )
    return merged


def get_metric_columns(frame, metric):
    """Return model columns containing the requested metric suffix."""
    metadata = {"Case_ID", "Class"}
    columns = [column for column in frame.columns if column not in metadata]
    metric_columns = [column for column in columns if metric.lower() in column.lower()]
    if not metric_columns:
        raise ValueError(f"No columns for metric '{metric}' were found.")
    return metric_columns


def clean_model_name(column, metric):
    return column.replace(f"_{metric}", "").replace(f"{metric}_", "")


def build_model_color_map(model_names):
    """Create one stable color per model without knowing model names in advance."""
    unique_names = sorted(set(model_names))
    color_map = plt.get_cmap("tab10", max(len(unique_names), 1))
    return {name: color_map(index) for index, name in enumerate(unique_names)}


def calculate_ranks(raw_data, metric, class_name):
    """Return mean ranks and the number of complete paired observations."""
    rank_values = calculate_rank_values(raw_data, metric, class_name)
    return rank_values.mean().sort_values(), len(rank_values)


def calculate_rank_values(raw_data, metric, class_name):
    """Return one rank per case and model for a metric/class combination."""
    class_data = raw_data if class_name is None else raw_data[raw_data["Class"] == class_name]
    columns = get_metric_columns(class_data, metric)
    values = class_data[columns].rename(
        columns={column: clean_model_name(column, metric) for column in columns}
    ).dropna()
    if values.empty:
        raise ValueError(f"No complete observations for {metric} and {class_name}.")

    lower_is_better = METRIC_DIRECTIONS[metric]
    return values.rank(axis=1, ascending=lower_is_better)


def nemenyi_sheet_name(metric, class_name):
    return f"{metric}_{class_name}"[:30] + "_Nemenyi"


def load_p_values(stats_path, metric, class_name, model_names):
    """Load and align a Nemenyi p-value matrix from the statistics workbook."""
    sheet_name = nemenyi_sheet_name(metric, class_name)
    p_values = pd.read_excel(stats_path, sheet_name=sheet_name, index_col=0)
    p_values.index = p_values.index.map(str)
    p_values.columns = p_values.columns.map(str)
    missing = set(model_names) - set(p_values.index)
    if missing:
        raise ValueError(f"{sheet_name} is missing models: {sorted(missing)}")
    return p_values.loc[model_names, model_names].apply(pd.to_numeric, errors="coerce")


def load_friedman_p_value(stats_path, metric, class_name):
    overview = pd.read_excel(stats_path, sheet_name="Friedman_Overview")
    row = overview[(overview["Metric"] == metric) & (overview["Class"] == class_name)]
    if row.empty:
        return None
    return float(row.iloc[0]["p-value"])


def calculate_combined_statistics(raw_data, metric, class_name=None):
    """Calculate Friedman and Nemenyi statistics for a class or all classes."""
    class_data = raw_data if class_name is None else raw_data[raw_data["Class"] == class_name]
    columns = get_metric_columns(class_data, metric)
    values = class_data[columns].rename(
        columns={column: clean_model_name(column, metric) for column in columns}
    ).dropna()
    model_names = list(values.columns)
    friedman_result = friedmanchisquare(*(values[name].to_numpy() for name in model_names))
    p_values = sp.posthoc_nemenyi_friedman(values)
    p_values.index = model_names
    p_values.columns = model_names
    return float(friedman_result.pvalue), p_values


def critical_difference(model_count, sample_count, alpha=ALPHA):
    """Calculate the Nemenyi critical difference for average ranks."""
    if model_count < 2 or sample_count < 1:
        raise ValueError("At least two models and one complete observation are required.")
    q_alpha = studentized_range.ppf(1 - alpha, model_count, np.inf) / math.sqrt(2)
    return q_alpha * math.sqrt(model_count * (model_count + 1) / (6 * sample_count))


def non_significant_groups(ranks, p_values, alpha=ALPHA):
    """Return maximal contiguous groups whose pairwise differences are non-significant."""
    ordered = list(ranks.sort_values().index)
    groups = []
    for start in range(len(ordered)):
        for end in range(start + 1, len(ordered)):
            group = ordered[start : end + 1]
            if all(
                pd.notna(p_values.loc[left, right])
                and p_values.loc[left, right] >= alpha
                for position, left in enumerate(group)
                for right in group[position + 1 :]
            ):
                groups.append(group)

    maximal = []
    for group in groups:
        if not any(set(group) < set(other) for other in groups):
            maximal.append(group)
    return maximal


def plot_diagram(
    ranks,
    p_values,
    metric,
    class_name,
    sample_count,
    output_path,
    alpha=ALPHA,
    friedman_p_value=None,
    color_map=None,
):
    cd = critical_difference(len(ranks), sample_count, alpha)
    groups = non_significant_groups(ranks, p_values, alpha)
    ordered_names = list(ranks.sort_values().index)
    max_rank = len(ordered_names) + 0.8
    rank_label_x = max_rank + 0.25
    clique_space = max(1.25, min(5.0, 0.32 * len(groups) + 0.8))
    baseline_name = ordered_names[0]
    baseline_color = "#2e8b57"
    similar_color = "#4f81bd"
    worse_color = "#c95b5b"

    figure_height = max(4.2, 2.7 + 0.42 * len(ordered_names) + 0.45 * clique_space)
    figure, axis = plt.subplots(figsize=(9.5, figure_height))
    axis.set_xlim(-1.25, rank_label_x + 0.65)
    axis.set_ylim(len(ordered_names) - 0.35, -clique_space - 1.15)
    axis.set_yticks([])
    axis.set_xlabel("Average rank (lower is better)")
    if friedman_p_value is None:
        p_text = "Friedman p unavailable"
    elif friedman_p_value < 0.001:
        p_text = "Friedman p < 0.001"
    else:
        p_text = f"Friedman p = {friedman_p_value:.3f}"
    axis.set_title(
        f"Critical Difference ({metric}, {class_name})\n"
        f"{p_text}, n = {sample_count} paired observations; baseline = {baseline_name}",
        pad=12,
    )
    axis.grid(axis="x", color="#e5e5e5", linewidth=0.8, zorder=0)

    for row, model_name in enumerate(ordered_names):
        rank = ranks[model_name]
        if model_name == baseline_name:
            bar_color = baseline_color
        else:
            p_value = p_values.loc[baseline_name, model_name]
            is_similar = (
                pd.notna(p_value) and p_value >= alpha
            ) or (
                pd.isna(p_value) and abs(rank - ranks[baseline_name]) <= cd
            )
            bar_color = similar_color if is_similar else worse_color
        axis.barh(
            row,
            rank,
            height=0.68,
            color=bar_color,
            edgecolor="#4b5563",
            linewidth=0.6,
            alpha=0.85,
            zorder=1,
        )
        axis.text(-0.08, row, model_name, ha="right", va="center", color="#333333")
        axis.text(
            rank_label_x,
            row,
            f"{rank:.2f}",
            ha="right",
            va="center",
            color="#333333",
            fontfamily="monospace",
        )

    # Each connector represents a maximal clique of models that is not
    # significantly different according to the Nemenyi p-value matrix.
    connector_color = "#c2410c"
    for group_index, group in enumerate(groups):
        left_rank = ranks[group].min()
        right_rank = ranks[group].max()
        connector_y = -0.45 - group_index * 0.27
        axis.plot(
            [left_rank, right_rank],
            [connector_y, connector_y],
            color=connector_color,
            linewidth=3,
            solid_capstyle="round",
            zorder=4,
        )
        axis.plot(
            [left_rank, left_rank],
            [connector_y, connector_y + 0.14],
            color=connector_color,
            linewidth=1.2,
            zorder=4,
        )
        axis.plot(
            [right_rank, right_rank],
            [connector_y, connector_y + 0.14],
            color=connector_color,
            linewidth=1.2,
            zorder=4,
        )

    axis.text(
        0,
        -clique_space + 0.08,
        "Orange connectors: models not significantly different (Nemenyi)",
        ha="left",
        va="center",
        fontsize=8,
        color="#7c2d12",
    )

    bracket_y = -clique_space - 0.35
    axis.plot([1, 1 + cd], [bracket_y, bracket_y], color="#444444", linewidth=1.5)
    axis.plot([1, 1], [bracket_y - 0.12, bracket_y + 0.12], color="#444444", linewidth=1.5)
    axis.plot(
        [1 + cd, 1 + cd],
        [bracket_y - 0.12, bracket_y + 0.12],
        color="#444444",
        linewidth=1.5,
    )
    axis.text(1 + cd / 2, bracket_y - 0.2, f"CD = {cd:.2f}", ha="center", va="bottom")
    axis.text(
        rank_label_x,
        -0.88,
        "Mean rank",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    axis.legend(
        handles=[
            Patch(facecolor=baseline_color, edgecolor="#4b5563", label="Best baseline"),
            Patch(facecolor=similar_color, edgecolor="#4b5563", label="Not significantly different"),
            Patch(facecolor=worse_color, edgecolor="#4b5563", label="Significantly worse"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.99),
        frameon=False,
        fontsize=8,
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_rank_distribution(rank_values, metric, class_name, output_path, color_map):
    """Plot per-case rank distributions in the style of a rank violin plot."""
    mean_ranks = rank_values.mean().sort_values()
    ordered_names = list(mean_ranks.index)
    distributions = [rank_values[name].to_numpy() for name in ordered_names]
    markers = ["s", "s", "D", "o", "o", "o", "s", "^", "o", "^"]
    x_positions = np.arange(1, len(ordered_names) + 1)

    figure_width = max(9.0, 1.0 * len(ordered_names))
    figure, axis = plt.subplots(figsize=(figure_width, 6.2))
    violins = axis.violinplot(
        distributions,
        positions=x_positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    colors = [color_map[model_name] for model_name in ordered_names]
    for body, color in zip(violins["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.48)

    rng = np.random.default_rng(7)
    for index, (position, model_name) in enumerate(zip(x_positions, ordered_names)):
        ranks_for_model = rank_values[model_name].to_numpy()
        jitter = rng.uniform(-0.13, 0.13, size=len(ranks_for_model))
        axis.scatter(
            position + jitter,
            ranks_for_model,
            marker=markers[index % len(markers)],
            s=24,
            color=colors[index % len(colors)],
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            zorder=3,
        )
        axis.plot(
            [position - 0.28, position + 0.28],
            [mean_ranks[model_name], mean_ranks[model_name]],
            color=colors[index % len(colors)],
            linewidth=2.2,
            zorder=4,
        )

    axis.set_title(f"Rank Distribution ({metric}, {class_name})", pad=14)
    axis.set_ylabel("Rank (1 = best)")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(ordered_names, rotation=45, ha="right")
    axis.set_ylim(len(ordered_names) + 0.5, 0.5)
    axis.set_yticks(np.arange(1, len(ordered_names) + 1))
    axis.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_diagrams(raw_path, stats_path, output_directory, alpha=ALPHA):
    return generate_selected_diagrams(raw_path, stats_path, output_directory, alpha)


def generate_selected_diagrams(
    raw_path,
    stats_path,
    output_directory,
    alpha=ALPHA,
    combine_classes=False,
    metric_filter=None,
    all_csv=False,
):
    raw_data = load_all_csv_results(RAW_DATA_DIRECTORY) if all_csv else read_table(raw_path)
    stats_path = Path(stats_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    model_names = []
    metrics = [metric_filter] if metric_filter else list(METRIC_DIRECTIONS)
    for metric in metrics:
        model_names.extend(
            clean_model_name(column, metric)
            for column in get_metric_columns(raw_data, metric)
        )
    color_map = build_model_color_map(model_names)

    generated_paths = []
    for metric in metrics:
        class_names = [None] if combine_classes else raw_data["Class"].dropna().unique()
        for class_name in class_names:
            rank_values = calculate_rank_values(raw_data, metric, class_name)
            ranks = rank_values.mean().sort_values()
            sample_count = len(rank_values)
            if combine_classes or all_csv:
                friedman_p_value, p_values = calculate_combined_statistics(
                    raw_data,
                    metric,
                    None if combine_classes else class_name,
                )
                class_label = "All_Classes"
                if not combine_classes:
                    class_label = class_name
            else:
                p_values = load_p_values(stats_path, metric, class_name, list(ranks.index))
                friedman_p_value = load_friedman_p_value(stats_path, metric, class_name)
                class_label = class_name
            filename = f"{metric}_{class_label}_critical_difference.png".replace(" ", "_")
            output_path = output_directory / filename
            plot_diagram(
                ranks,
                p_values,
                metric,
                class_label,
                sample_count,
                output_path,
                alpha,
                friedman_p_value,
                color_map,
            )
            generated_paths.append(output_path)
            distribution_path = output_directory / (
                f"{metric}_{class_label}_rank_distribution.png".replace(" ", "_")
            )
            plot_rank_distribution(
                rank_values,
                metric,
                class_label,
                distribution_path,
                color_map,
            )
            generated_paths.append(distribution_path)
    return generated_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=RAW_DATA_PATH, help="Raw CSV or Excel result table.")
    parser.add_argument("--stats", default=STATS_DATA_PATH, help="Friedman/Nemenyi Excel workbook.")
    parser.add_argument("--output-dir", default=OUTPUT_DIRECTORY, help="Directory for PNG diagrams.")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Significance level (default: 0.05).")
    parser.add_argument(
        "--combine-classes",
        action="store_true",
        help="Pool all classes into one rank analysis and one plot per metric.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_DIRECTIONS),
        help="Only generate diagrams for this metric.",
    )
    parser.add_argument(
        "--all-csv",
        action="store_true",
        help="Merge all segmentation_results*.csv files in the script directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    paths = generate_selected_diagrams(
        arguments.raw,
        arguments.stats,
        arguments.output_dir,
        arguments.alpha,
        arguments.combine_classes,
        arguments.metric,
        arguments.all_csv,
    )
    print(f"Generated {len(paths)} diagrams in '{Path(arguments.output_dir)}'.")