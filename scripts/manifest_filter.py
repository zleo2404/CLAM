import json
import re

def filter_gdc_data(clinical_file, manifest_file, output_file):
    with open(clinical_file, 'r', encoding='utf-8') as f:
        clinical_data = json.load(f)

    valid_patients = set()

    for patient in clinical_data:
        submitter_id = patient.get('submitter_id')
        if not submitter_id:
            continue
            
        ihc_result = None
        fish_result = None

        for fu in patient.get('follow_ups', []):
            for test in fu.get('molecular_tests', []):
                if test.get('gene_symbol') == 'ERBB2':
                    method = test.get('molecular_analysis_method')
                    result = test.get('test_result')
                    
                    if method == 'IHC':
                        ihc_result = result
                    elif method == 'FISH':
                        fish_result = result

        is_valid = False
        if ihc_result in ['Positive', 'Negative']:
            is_valid = True
        elif fish_result in ['Positive', 'Negative']:
            is_valid = True

        if is_valid:
            valid_patients.add(submitter_id)

    print(f"Found {len(valid_patients)} patients meeting IHC/FISH clinical criteria.")

    saved_count = 0
    #seen_patients = set()

    with open(manifest_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        header = f_in.readline()
        f_out.write(header)

        for line in f_in:
            cols = line.strip().split('\t')
            if len(cols) < 2:
                continue
                
            filename = cols[1]

            if filename.endswith('.svs'):
                match = re.match(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-([0-9]{2}[A-Z])', filename)
                if match:
                    barcode = match.group(1)
                    sample_code = match.group(2)
                    
                    if sample_code.startswith('11') or sample_code.startswith('06'):
                        continue
#and barcode not in seen_patients
                    if barcode in valid_patients:
                        f_out.write(line)
                        #seen_patients.add(barcode)
                        saved_count += 1

    print(f"Filtering complete. Saved {saved_count} unique .svs files to the output manifest.")

if __name__ == "__main__":
    CLINICAL_JSON = '/scratch.hpc/leonardo.meloni/CLAM/scripts/clinical.project-tcga-brca.2026-08-26.json'
    MANIFEST_TXT = '/scratch.hpc/leonardo.meloni/CLAM/scripts/gdc_manifest.2026-08-26.132302.txt'
    OUTPUT_MANIFEST = '/scratch.hpc/leonardo.meloni/CLAM/scripts/filtered_manifest.txt'
    
    filter_gdc_data(CLINICAL_JSON, MANIFEST_TXT, OUTPUT_MANIFEST)