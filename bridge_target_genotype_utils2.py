import os
import subprocess


def intersect_sum_stats_and_bim_file(genotype_folder, prefix, full_sum_stats_file):
    # Extracting the name of the original sum stats file
    original_sum_stats_name = full_sum_stats_file.split('/')[-1].split('.')[0]
    bim_path = f'{genotype_folder}{prefix}.bim'
    # print(bim_path)
    bim_data = {}
    with open(bim_path, 'r') as bim_file:
        for line in bim_file:
            fields = line.strip().split()
            rs_id = fields[1]
            bim_data[rs_id] = fields
    # print(bim_data.keys())
    effect_data = {}
    with open(full_sum_stats_file, 'r') as effect_file:
        for line in effect_file:
            if not line.startswith("rsID"):
                fields = line.strip().split("\t")
                # print(fields)
                rs_id = fields[0]
                effect_data[rs_id] = fields[1:]
    # Intersect the dictionaries based on rsID
    # print(effect_data.keys(),"length effect data")

    intersected_data = {}
    for rs_id in bim_data:
        if rs_id in effect_data:
            bim_row = bim_data[rs_id]
            effect_row = effect_data[rs_id]
            intersected_data[rs_id] = bim_row + effect_row
    # print(len(intersected_data),"length intersected data")

    # Constructing the filtered sum stats file name based on the original sum stats file name
    filtered_sum_stats_file = f'intersect_{prefix}_{original_sum_stats_name}.txt'
    # Write the intersected data to a new file
    with open(filtered_sum_stats_file, 'w') as output_file:
        for rs_id in intersected_data:
            output_file.write('\t'.join(intersected_data[rs_id]) + '\n')
    print('Done filtering: ')
    return filtered_sum_stats_file

def intersect_sum_stats_weights_and_bim_file(genotype_folder, prefix, full_sum_stats_file, weights_file):
    # Extracting the name of the original sum stats file
    original_sum_stats_name = full_sum_stats_file.split('/')[-1].split('.')[0]
    bim_path = f'{genotype_folder}{prefix}.bim'

    # Load weights file into a set for efficient lookup
    weights_rsids = set()
    with open(weights_file, 'r') as weights:
        next(weights)  # Skip header
        for line in weights:
            rsid = line.strip().split(',')[0]
            weights_rsids.add(rsid)

    bim_data = {}
    with open(bim_path, 'r') as bim_file:
        for line in bim_file:
            fields = line.strip().split()
            rs_id = fields[1]
            if rs_id in weights_rsids:  # Filter based on weights file
                bim_data[rs_id] = fields

    effect_data = {}
    with open(full_sum_stats_file, 'r') as effect_file:
        header = next(effect_file)
        for line in effect_file:
            fields = line.strip().split(",")
            rs_id = fields[0]
            if rs_id in weights_rsids:  # Filter based on weights file
                effect_data[rs_id] = fields[1:]

    # Intersect the dictionaries based on rsID
    intersected_data = {}
    for rs_id in weights_rsids:
        if rs_id in bim_data and rs_id in effect_data:
            bim_row = bim_data[rs_id]
            effect_row = effect_data[rs_id]
            intersected_data[rs_id] = bim_row + effect_row

    # Constructing the filtered sum stats file name based on the original sum stats file name
    filtered_sum_stats_file = f'intersected_data_{original_sum_stats_name}.txt'
    # Write the intersected data to a new file
    with open(filtered_sum_stats_file, 'w') as output_file:
        output_file.write(header)  # Write the header
        for rs_id in intersected_data:
            output_file.write('\t'.join(intersected_data[rs_id]) + '\n')

    return filtered_sum_stats_file



def extract_snps_and_create_genotype_ld_matrix(genotype_folder, genotype_prefix, filtered_sum_stats_file):
    # Extracting the name of the filtered sum stats file

    filtered_sum_stats_name = filtered_sum_stats_file.split('/')[-1].split('.')[0]
    snps_list_file = f'snps_list_{filtered_sum_stats_name}.txt'
    subprocess.run(['awk', '{print $2}', filtered_sum_stats_file], stdout=open(snps_list_file, 'w'))

    plink_command_create_bed = f'plink --bfile {genotype_folder}{genotype_prefix} --extract {snps_list_file} --make-bed --silent --out {genotype_folder}filt_{filtered_sum_stats_name}'
    subprocess.run(plink_command_create_bed, shell=True)

    plink_command_recode = f'plink --bfile {genotype_folder}filt_{filtered_sum_stats_name} --recodeA --silent --out {genotype_folder}filt_{filtered_sum_stats_name}'
    subprocess.run(plink_command_recode, shell=True)

    ld_file_path = f'{genotype_folder}filt_{filtered_sum_stats_name}.ld'
    if not os.path.exists(ld_file_path):
        plink_command_ind_ld = f'plink --bfile {genotype_folder}{genotype_prefix} --r2 --ld-window-r2 0.2 --ld-window 1000 --out {genotype_folder}filt_{filtered_sum_stats_name}'
        subprocess.run(plink_command_ind_ld, shell=True)
    # plink_command_ind_ld = f'plink --bfile {genotype_folder}{genotype_prefix} --r2 inter-chr --out {genotype_folder}filt_{filtered_sum_stats_name}'
    # subprocess.run(plink_command_ind_ld, shell=True)
    print("Done creating the LD Matrix ")
    return f'{genotype_folder}filt_{filtered_sum_stats_name}',snps_list_file

def extract_snps_and_create_genotype_ld_matrix_ref_panel(genotype_folder, genotype_prefix, filtered_sum_stats_file):
    # Extracting the name of the filtered sum stats file

    filtered_sum_stats_name = filtered_sum_stats_file.split('/')[-1].split('.')[0]
    snps_list_file = f'snps_list_{filtered_sum_stats_name}.txt'
    subprocess.run(['awk', '{print $2}', filtered_sum_stats_file], stdout=open(snps_list_file, 'w'))

    plink_command_create_bed = f'plink --bfile {genotype_folder}{genotype_prefix} --extract {snps_list_file} --make-bed --silent --out {genotype_folder}filt_{filtered_sum_stats_name}'
    subprocess.run(plink_command_create_bed, shell=True)

    plink_command_recode = f'plink --bfile {genotype_folder}filt_{filtered_sum_stats_name} --recodeA --silent --out {genotype_folder}filt_{filtered_sum_stats_name}'
    subprocess.run(plink_command_recode, shell=True)

    ld_file_path = f'{genotype_folder}filt_{filtered_sum_stats_name}.ld'

    # plink_command_ind_ld = f'plink --bfile {genotype_folder}{genotype_prefix} --r2 inter-chr --out {genotype_folder}filt_{filtered_sum_stats_name}'
    # subprocess.run(plink_command_ind_ld, shell=True)
    print("Done creating the LD Matrix ")
    return f'{genotype_folder}filt_{filtered_sum_stats_name}',snps_list_file

def extract_snps_and_create_genotype(genotype_folder, genotype_prefix, filtered_sum_stats_file):
    # Extracting the name of the filtered sum stats file
    filtered_sum_stats_name = filtered_sum_stats_file.split('/')[-1].split('.')[0]
    snps_list_file = f'snps_list_{filtered_sum_stats_name}.txt'
    subprocess.run(['awk', '{print $2}', filtered_sum_stats_file], stdout=open(snps_list_file, 'w'))
    plink_command = f'plink --bfile {genotype_folder}{genotype_prefix} --extract {snps_list_file} --recodeA --silent --out {genotype_folder}filt_{filtered_sum_stats_name}'
    subprocess.run(plink_command, shell=True)
    return f'{genotype_folder}filt_{filtered_sum_stats_name}'