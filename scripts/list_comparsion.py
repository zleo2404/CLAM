import pandas as pd

FILE_MANIFEST = "/scratch.hpc/leonardo.meloni/CLAM/scripts/filtered_manifest.txt"
FILE_CLINICAL = "/scratch.hpc/leonardo.meloni/CLAM/scripts/label_her2.csv"
FILE_OUTPUT = "/scratch.hpc/leonardo.meloni/CLAM/scripts/missing_slides.csv"

print("Loading...\n")

manifest_df = pd.read_csv(FILE_MANIFEST, sep='\t')

manifest_df['patient_id'] = manifest_df['filename'].apply(lambda x: "-".join(str(x).split('-')[0:3]))

clinical_df = pd.read_csv(FILE_CLINICAL)

clinical_patients = set(clinical_df['cases.submitter_id'].dropna())

missing_slides = manifest_df[~manifest_df['patient_id'].isin(clinical_patients)]

print(f"Total slides in List 1: {len(manifest_df)}")
print(f"Total patients in List 2: {len(clinical_patients)}")
print(f"Slides from List 1 MISSING in List 2: {len(missing_slides)}\n")

if len(missing_slides) > 0:
    print("Examples of orphan files (no clinical data):")
    print(missing_slides[['patient_id', 'filename']].head())
    
    missing_slides.to_csv(FILE_OUTPUT, index=False)
    print(f"\nThe complete list of {len(missing_slides)} missing slides has been saved to: '{FILE_OUTPUT}'")
else:
    print("Great news! All slides from List 1 have a record in List 2.")