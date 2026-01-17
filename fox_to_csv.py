import sys
import os
import pandas as pd
import numpy as np

def convert_fox_to_xy(input_filename):
    # 1. Determine output filename
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}.xy"

    try:
        # 2. Read the file
        # Added 'r' before '\s+' to fix the SyntaxWarning
        df = pd.read_csv(input_filename, sep=r'\s+', skiprows=1)

        if len(df.columns) < 4:
            print(f"Error: Format of '{input_filename}' not recognized.")
            return

        # 3. Extract data
        two_theta = df.iloc[:, 0]
        calc_intensity = df.iloc[:, 3]

        # 4. Create Random Background (Noise)
        # Random number between 0 and 8 for EVERY data point
        noise = np.random.uniform(0, 8, size=len(df))

        # 5. Calculate final simulated intensity
        # Intensity = (Calc * 500) + Random Noise
        final_intensity = (calc_intensity * 500) + noise

        # 6. Save to output file
        output_df = pd.DataFrame({
            '2Theta': two_theta,
            'Intensity': final_intensity
        })

        output_df.to_csv(output_filename, sep='\t', index=False, header=False)
        
        print(f"Success! Created '{output_filename}'")
        print("Applied random background noise (0-8) to data scaled by 500.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fox_to_csv.py <filename>")
    else:
        convert_fox_to_xy(sys.argv[1])