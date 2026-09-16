import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


def _get_significant_pairwise_differences(nemenyi_df, alpha=0.05):
    """Return model pairs with statistically significant differences from a Nemenyi matrix."""
    significant_pairs = []
    for i in range(len(nemenyi_df.index)):
        for j in range(i + 1, len(nemenyi_df.columns)):
            model_a = nemenyi_df.index[i]
            model_b = nemenyi_df.columns[j]
            p_value = nemenyi_df.iloc[i, j]
            if pd.notna(p_value) and p_value < alpha:
                significant_pairs.append((model_a, model_b, float(p_value)))
    return significant_pairs


def _get_better_model(sub_df, model_a, model_b, lower_is_better):
    """Return the better-performing model for a pair based on the metric direction."""
    mean_a = sub_df[model_a].mean()
    mean_b = sub_df[model_b].mean()

    if lower_is_better:
        if mean_a < mean_b:
            return model_a, mean_a, mean_b
        return model_b, mean_b, mean_a

    if mean_a > mean_b:
        return model_a, mean_a, mean_b
    return model_b, mean_b, mean_a


def run_classwise_friedman(input_file, output_excel="Classwise_Statistical_Results.xlsx"):
    """
    Reads segmentation results for multiple classes, runs class-wise Friedman 
    and Nemenyi tests, and exports everything into an Excel workbook.
    """
    # 1. Load Data (Supports CSV or Excel)
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)
    # Clean missing values to maintain paired samples
    df = df.dropna()

    # Identify metadata vs. model columns
    # Assumes 'Case_ID' and 'Class' exist; all other numerical columns are treated as models
    metadata_cols = ['Case_ID', 'Class']
    model_cols = [c for c in df.columns if c not in metadata_cols]

    classes = df['Class'].unique()
    metrics = [
        {'name': 'Dice', 'lower_is_better': False},
        {'name': 'HD95', 'lower_is_better': True}
    ]

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        overview_records = []
        significant_pair_records = []

        # 2. Iterate through each Metric and Class
        for metric in metrics:
            metric_name = metric['name']
            lower_is_better = metric['lower_is_better']

            for class_label in classes:
                # Filter rows for the current class
                class_df = df[df['Class'] == class_label]

                # Select only columns matching the current metric (e.g., ModelA_Dice or ModelA)
                # If metric names are embedded in columns, filter them; otherwise use model_cols directly
                target_cols = [c for c in model_cols if metric_name.lower() in c.lower()]
                if not target_cols:
                    target_cols = model_cols  # Fallback if metrics are in separate files/tables

                sub_df = class_df[target_cols].copy()

                # Clean column headers to display clean model names
                clean_names = {c: c.replace(f"_{metric_name}", "").replace(f"{metric_name}_", "") for c in target_cols}
                sub_df = sub_df.rename(columns=clean_names)

                # 3. Perform Friedman Test
                data_arrays = [sub_df[col].values for col in sub_df.columns]
                stat, p_val = friedmanchisquare(*data_arrays)
                is_sig = p_val < 0.05

                # Record Overview Summary
                overview_records.append({
                    'Metric': metric_name,
                    'Class': class_label,
                    'Sample Size (N)': len(sub_df),
                    'Friedman Chi2 Stat': round(stat, 4),
                    'p-value': p_val,
                    'Significant (a=0.05)': is_sig
                })

                # 4. Descriptive Stats & Ranks
                ranks = sub_df.rank(axis=1, ascending=lower_is_better).mean()
                stats_df = pd.DataFrame({
                    'Mean': sub_df.mean(),
                    'Std': sub_df.std(),
                    'Median': sub_df.median(),
                    'Mean Rank': ranks
                }).sort_values('Mean Rank')

                sheet_prefix = f"{metric_name}_{class_label}"[:30] # Excel sheet name length limit
                stats_df.round(4).to_excel(writer, sheet_name=f"{sheet_prefix}_Stats")

                # 5. Post-Hoc Nemenyi Pairwise p-values
                nemenyi_df = sp.posthoc_nemenyi_friedman(sub_df)
                significant_pairs = _get_significant_pairwise_differences(nemenyi_df)

                print(f"\n{metric_name} - {class_label}: significant model differences (alpha=0.05)")
                if significant_pairs:
                    for model_a, model_b, p_value in significant_pairs:
                        better_model, better_value, worse_value = _get_better_model(sub_df, model_a, model_b, lower_is_better)
                        print(
                            f"  - {model_a} vs {model_b}: p = {p_value:.4g}; "
                            f"better = {better_model} ({metric_name} mean = {better_value:.4g})"
                        )
                        significant_pair_records.append({
                            'Metric': metric_name,
                            'Class': class_label,
                            'Model A': model_a,
                            'Model B': model_b,
                            'p-value': p_value,
                            'Better model': better_model,
                            'Model A mean': sub_df[model_a].mean(),
                            'Model B mean': sub_df[model_b].mean(),
                            'Significant (a=0.05)': True
                        })
                else:
                    print("  - No significant pairwise differences between models.")
                    significant_pair_records.append({
                        'Metric': metric_name,
                        'Class': class_label,
                        'Model A': None,
                        'Model B': None,
                        'p-value': None,
                        'Better model': None,
                        'Model A mean': None,
                        'Model B mean': None,
                        'Significant (a=0.05)': False
                    })

                nemenyi_df.round(4).to_excel(writer, sheet_name=f"{sheet_prefix}_Nemenyi")

        # Save Main Summary Sheets
        pd.DataFrame(overview_records).to_excel(writer, sheet_name='Friedman_Overview', index=False)
        pd.DataFrame(significant_pair_records).to_excel(
            writer,
            sheet_name='Significant_Model_Pairs',
            index=False
        )

    print(f"\nAnalysis complete! Results successfully exported to: {output_excel}")


# ==========================================
# RUN ANALYSIS
# ==========================================
run_classwise_friedman('segmentation_results_selected.csv')