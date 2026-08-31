"""
Space Group Reflection Conditions Generator - Scientifically Rigorous Version

This script generates a JSON file containing the h,k,l reflection conditions
for all 230 space groups in all their 530+ standard settings.

This version uses the 'gemmi' library and reads from a corrected, verified
'settings_list.json' file.

This version includes the critical fix in `is_reflection_absent` to use
the fractional rotation matrix.
"""

import json
import numpy as np
import gemmi # Using Gemmi library
from collections import defaultdict

def is_reflection_absent(gemmi_ops, h, k, l):
    """
    Checks if a reflection (h, k, l) is systematically absent based on the
    symmetry operations of the space group.
    
    A reflection H is absent if, for any symmetry op (R, t):
    1. The reflection vector is invariant under the rotation: H.R = H
    2. The phase shift from the translation is not an integer: H.t != integer
    """
    h_vec = np.array([h, k, l])
    # gemmi.Op.DEN is the fractional base (e.g., 24)
    DEN = gemmi.Op.DEN 
    
    for op in gemmi_ops:
        # *** --- CRITICAL FIX --- ***
        # Get fractional rotation matrix. op.rot is an integer matrix
        # and must be divided by the denominator DEN.
        rot_frac = np.array(op.rot) / DEN 
        # *** --- END FIX --- ***
        
        # Get fractional translation vector
        trans_frac = np.array(op.tran) / DEN
        
        # Condition 1: The reflection must be invariant under the rotation part.
        # *** --- CRITICAL FIX --- ***
        # Use the fractional matrix `rot_frac` for the check.
        if np.allclose(h_vec @ rot_frac, h_vec):
        # *** --- END FIX --- ***
            
            # Condition 2: The phase shift from the translation
            ht = np.dot(h_vec, trans_frac)
            # Check if h.t is NOT an integer (with a small tolerance)
            if not np.isclose(ht, np.round(ht)):
                return True  # Reflection is absent
                
    return False # Reflection is present

def analyze_zone(gemmi_ops, zone_type, max_index=8):
    """
    Analyzes systematic absences for a specific reflection zone by identifying
    the mathematical conditions that govern the PRESENT reflections.
    """
    test_refs = []
    # Generate a comprehensive set of test reflections for the given zone
    # Use robust ranges that include 0 and negative values
    h_range = range(-max_index // 2, max_index)
    k_range = range(-max_index // 2, max_index)
    l_range = range(-max_index // 2, max_index)

    if zone_type == 'hkl':
        test_refs = [(h, k, l) for h in h_range for k in k_range for l in l_range if not (h==0 and k==0 and l==0)]
    elif zone_type == 'h00':
        test_refs = [(h, 0, 0) for h in h_range if h != 0]
    elif zone_type == '0k0':
        test_refs = [(0, k, 0) for k in k_range if k != 0]
    elif zone_type == '00l':
        test_refs = [(0, 0, l) for l in l_range if l != 0]
    elif zone_type == 'hk0':
        test_refs = [(h, k, 0) for h in h_range for k in k_range if not (h==0 and k==0)]
    elif zone_type == 'h0l':
        test_refs = [(h, 0, l) for h in h_range for l in l_range if not (h==0 and l==0)]
    elif zone_type == '0kl':
        test_refs = [(0, k, l) for k in k_range for l in l_range if not (k==0 and l==0)]
    elif zone_type == 'hhl':
         test_refs = [(h, h, l) for h in h_range for l in l_range if not (h==0 and l==0)]
    elif zone_type == 'hkk':
         test_refs = [(h, k, k) for h in h_range for k in k_range if not (h==0 and k==0)]
    elif zone_type == 'hll':
         test_refs = [(h, l, l) for h in h_range for l in l_range if not (h==0 and l==0)]

    if not test_refs:
        return None

    # Filter for reflections that are NOT systematically absent
    present_refs = [ref for ref in test_refs if not is_reflection_absent(gemmi_ops, *ref)]

    # If all test reflections are present, there are no special conditions
    if len(present_refs) == len(test_refs) or not present_refs:
        return None

    conditions = set()
    
    # --- Deduce the rules by analyzing the patterns in the PRESENT reflections ---
    h_vals = [r[0] for r in present_refs]
    k_vals = [r[1] for r in present_refs]
    l_vals = [r[2] for r in present_refs]
    
    if zone_type == 'hkl':
        if all((h + k) % 2 == 0 and (k + l) % 2 == 0 and (h + l) % 2 == 0 for h, k, l in present_refs):
            conditions.add("h+k, k+l, h+l=2n")
        elif all((h + k + l) % 2 == 0 for h, k, l in present_refs):
            conditions.add("h+k+l=2n")
        elif all((k + l) % 2 == 0 for h, k, l in present_refs):
            conditions.add("k+l=2n")
        elif all((h + l) % 2 == 0 for h, k, l in present_refs):
            conditions.add("h+l=2n")
        elif all((h + k) % 2 == 0 for h, k, l in present_refs):
            conditions.add("h+k=2n")
        elif all((-h + k + l) % 3 == 0 for h, k, l in present_refs):
            conditions.add("-h+k+l=3n")
        elif all((h - k + l) % 3 == 0 for h, k, l in present_refs):
            conditions.add("h-k+l=3n")

    elif zone_type == 'h00':
        if all(h % 4 == 0 for h in h_vals): conditions.add("h=4n")
        elif all(h % 2 == 0 for h in h_vals): conditions.add("h=2n")
    elif zone_type == '0k0':
        if all(k % 4 == 0 for k in k_vals): conditions.add("k=4n")
        elif all(k % 2 == 0 for k in k_vals): conditions.add("k=2n")
    elif zone_type == '00l':
        if all(l % 6 == 0 for l in l_vals): conditions.add("l=6n")
        elif all(l % 4 == 0 for l in l_vals): conditions.add("l=4n")
        elif all(l % 3 == 0 for l in l_vals): conditions.add("l=3n")
        elif all(l % 2 == 0 for l in l_vals): conditions.add("l=2n")
    
    elif zone_type == 'hk0':
        h2n = all(h % 2 == 0 for h in h_vals)
        k2n = all(k % 2 == 0 for k in k_vals)
        if h2n: conditions.add("h=2n")
        if k2n: conditions.add("k=2n")
        
        if all((h + k) % 4 == 0 for h, k in zip(h_vals, k_vals)):
            conditions.add("h+k=4n")
        elif all((h + k) % 2 == 0 for h, k in zip(h_vals, k_vals)):
            conditions.add("h+k=2n")
        
        # Cleanup redundant rules
        if h2n and k2n: conditions.discard("h+k=2n") # h=2n, k=2n is more specific
        if "h+k=4n" in conditions: conditions.discard("h+k=2n")

    elif zone_type == 'h0l':
        h2n = all(h % 2 == 0 for h in h_vals)
        l2n = all(l % 2 == 0 for l in l_vals)
        if h2n: conditions.add("h=2n")
        if l2n: conditions.add("l=2n")

        if all((h + l) % 4 == 0 for h, l in zip(h_vals, l_vals)):
            conditions.add("h+l=4n")
        elif all((h + l) % 2 == 0 for h, l in zip(h_vals, l_vals)):
            conditions.add("h+l=2n")

        # Cleanup redundant rules
        if h2n and l2n: conditions.discard("h+l=2n")
        if "h+l=4n" in conditions: conditions.discard("h+l=2n")

    elif zone_type == '0kl':
        k2n = all(k % 2 == 0 for k in k_vals)
        l2n = all(l % 2 == 0 for l in l_vals)
        if k2n: conditions.add("k=2n")
        if l2n: conditions.add("l=2n")
        
        if all((k + l) % 4 == 0 for k, l in zip(k_vals, l_vals)):
            conditions.add("k+l=4n")
        elif all((k + l) % 2 == 0 for k, l in zip(k_vals, l_vals)):
            conditions.add("k+l=2n")

        # Cleanup redundant rules
        if k2n and l2n: conditions.discard("k+l=2n")
        if "k+l=4n" in conditions: conditions.discard("k+l=2n")

    elif zone_type == 'hhl':
        if all((2*h + l) % 4 == 0 for h, l in zip(h_vals, l_vals)): conditions.add("2h+l=4n")
        if all((h + l) % 2 == 0 for h, l in zip(h_vals, l_vals)): conditions.add("h+l=2n")
        if all(l % 2 == 0 for l in l_vals): conditions.add("l=2n")
        
        # Cleanup
        if "2h+l=4n" in conditions: conditions.discard("l=2n") # 2h+l=4n is more specific

    elif zone_type == 'hkk':
        if all((h + 2*k) % 4 == 0 for h, k in zip(h_vals, k_vals)): conditions.add("h+2k=4n")
        if all((h + k) % 2 == 0 for h, k in zip(h_vals, k_vals)): conditions.add("h+k=2n")
        if all(h % 2 == 0 for h in h_vals): conditions.add("h=2n")
        
        # Cleanup
        if "h+2k=4n" in conditions: conditions.discard("h=2n") # h+2k=4n is more specific

    elif zone_type == 'hll':
         if all((h + 2*l) % 4 == 0 for h, l in zip(h_vals, l_vals)): conditions.add("h+2l=4n")
         if all((h + l) % 2 == 0 for h, l in zip(h_vals, l_vals)): conditions.add("h+l=2n")
         if all(h % 2 == 0 for h in h_vals): conditions.add("h=2n")

         # Cleanup
         if "h+2l=4n" in conditions: conditions.discard("h=2n") # h+2l=4n is more specific

    return sorted(list(conditions)) if conditions else None


def main():
    """Main execution function to generate the JSON database."""
    database = {}
    skipped_log_file = "skipped_symbols.log" # Log file for errors
    print("="*70)
    print("Rigorous Space Group Reflection Condition Database Generator")
    print("Using 'gemmi' and canonical 'settings_list.json'...")
    print(f"Skipped symbols will be logged to: {skipped_log_file}")
    print("="*70)

    try:
        with open(skipped_log_file, 'w') as f:
            f.write("Log of skipped space group symbols:\n")
            f.write("======================================\n")
    except IOError as e:
        print(f"CRITICAL ERROR: Could not write to log file {skipped_log_file}. Check permissions.")
        print(f"Error: {e}")
        return

    # Load the canonical settings list from the JSON file
    try:
        with open('settings_list.json', 'r') as f:
            settings_list = json.load(f)
    except FileNotFoundError:
        print("ERROR: 'settings_list.json' not found.")
        print("Please create this file in the same directory.")
        return
    except json.JSONDecodeError:
        print("ERROR: 'settings_list.json' is corrupted or not valid JSON.")
        return

    # Iterate over the correct list of settings
    for setting in settings_list:
        sg_number = setting["number"]
        hm_symbol = setting["symbol"]
        setting_name = setting["qualifier"] # This is "abc", "bca", "H", "R", or ""

        try:
            # Get the gemmi space group object using the symbol
            sg = gemmi.find_spacegroup_by_name(hm_symbol)
            
            if not sg:
                print(f"Warning: gemmi could not parse symbol '{hm_symbol}'. Skipping.")
                # Log the skipped symbol
                with open(skipped_log_file, 'a') as f:
                    f.write(f"Number: {sg_number}, Symbol: '{hm_symbol}', Qualifier: '{setting_name}'\n")
                continue
                
            if sg.number != sg_number:
                # Check if the parsed number matches the expected number
                print(f"Warning: Symbol '{hm_symbol}' (expected {sg_number}) was parsed as SG {sg.number}. Skipping.")
                with open(skipped_log_file, 'a') as f:
                    f.write(f"Number Mismatch: {sg_number}, Symbol: '{hm_symbol}', Parsed as: {sg.number}\n")
                continue


            # Check if this is the first time we've seen this space group number
            if str(sg_number) not in database:
                # Get the standard symbol for the top-level entry
                sg_std = gemmi.find_spacegroup_by_number(sg_number)
                database[str(sg_number)] = {
                    "number": sg_number,
                    "standard_symbol": sg_std.hm,
                    "crystal_system": sg_std.crystal_system_str(),
                    "point_group": sg_std.point_group_hm(),
                    "centrosymmetric": sg_std.is_centrosymmetric(),
                    "settings": []
                }
                print(f"\n--- Found SG {sg_number:3d}: {sg_std.hm} ({sg_std.crystal_system_str()}) ---")

            print(f"  Processing Setting: {hm_symbol:<12} (axes: {setting_name})")

            ops = sg.operations() # Get the gemmi operations
            
            zones = ['hkl', '0kl', 'h0l', 'hk0', 'hhl', 'hkk', 'hll', 'h00', '0k0', '00l']
            setting_conditions = {}
            for zone in zones:
                # Pass the gemmi operations list to analyze_zone
                conditions = analyze_zone(ops, zone)
                if conditions:
                    setting_conditions[zone] = conditions
            
            setting_data = {
                "symbol": hm_symbol,
                "description": setting_name, 
                "reflection_conditions": setting_conditions
            }
            database[str(sg_number)]["settings"].append(setting_data)
        
        except Exception as e:
            print(f"ERROR processing SG {sg_number} ({hm_symbol}): {e}")
            with open(skipped_log_file, 'a') as f:
                    f.write(f"CRITICAL ERROR: {sg_number}, Symbol: '{hm_symbol}', Error: {e}\n")
            continue

    output_file = "reflection_conditions_gemmi_final.json"
    with open(output_file, 'w') as f:
        # Sort the database by space group number
        sorted_database = dict(sorted(database.items(), key=lambda item: int(item[0])))
        json.dump({"space_groups": sorted_database}, f, indent=2)

    print("\n" + "="*70)
    print(f"Database generation complete. Saved to: {output_file}")
    print(f"Total space groups processed: {len(database)}")
    print(f"A log of skipped symbols was saved to: {skipped_log_file}")
    print("="*70)

if __name__ == "__main__":
    main()