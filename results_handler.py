import pandas as pd
import numpy as np
from tabulate import tabulate

def display_and_save_results(results):
    table_data = []
    headers = ["Model", "Scenario"] + list(next(iter(results.values()))["No Attack"].keys())

    for model_name, scenarios in results.items():
        for scenario, metrics in scenarios.items():
            row = [model_name, scenario] + list(metrics.values())
            table_data.append(row)

    print("\nResults:")
    print(tabulate(table_data, headers=headers, floatfmt=".4f", tablefmt="grid"))

    results_df = pd.DataFrame(table_data, columns=headers)
    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
    results_df[numeric_columns] = results_df[numeric_columns].round(4)

    results_df.to_csv("recommender_system_results.csv", index=False)
    print("\nResults saved to 'recommender_system_results.csv'")