import pandas as pd

FILE_PROCESS = "/scratch.hpc/leonardo.meloni/CLAM/patching_results_20x/process_list_filtered.csv"
FILE_LABEL = "/scratch.hpc/leonardo.meloni/CLAM/scripts/label_her2.csv"

process_df = pd.read_csv(FILE_PROCESS)

process_df['cases.submitter_id'] = process_df['slide_id'].apply(lambda x: "-".join(str(x).split('-')[0:3]))

label_df = pd.read_csv(FILE_LABEL)

merged_df = pd.merge(process_df, label_df, on='cases.submitter_id', how='inner')

merged_df = merged_df[merged_df['her2_result'].isin(['Positive', 'Negative'])]

print(f"Total matched primary tumor slides: {len(merged_df)}\n")
print("Breakdown by HER2 Status and IHC Score:")
print("-" * 50)

summary = merged_df.groupby(['her2_result', 'ihc_score']).size().reset_index(name='count')

for _, row in summary.iterrows():
    print(f"HER2: {row['her2_result']:<10} | IHC Score: {row['ihc_score']:<5} | Slides: {row['count']}")

print("-" * 50)