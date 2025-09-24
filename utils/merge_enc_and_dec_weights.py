import argparse
import pandas as pd


def merge_and_sum(file1, file2, output_file):
    # Read the datasets from the input files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Merge based on SNP_ID, Chromosome, and Position columns
    merged_df = pd.merge(df1, df2, on=["SNP_ID", "Chromosome", "Position"])

    # Sum the 'Weight' columns from both datasets into one column
    merged_df['Weight'] = df1['Weight'] + df2['Weight']

    # Drop the original 'Weight_x' and 'Weight_y' columns
    merged_df.drop(['Weight_x', 'Weight_y'], axis=1, inplace=True)

    # Write merged dataframe to output file
    merged_df.to_csv(output_file, index=False)


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Merge and sum based on SNP_ID, Chromosome, and Position columns")
    parser.add_argument("file1", help="Path to the first input file")
    parser.add_argument("file2", help="Path to the second input file")
    parser.add_argument("output_file", help="Path to the output file")
    args = parser.parse_args()

    # Call merge_and_sum function with command-line arguments
    merge_and_sum(args.file1, args.file2, args.output_file)


if __name__ == "__main__":
    main()
