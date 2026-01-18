# gemini_cctbx_generate_sg_json_v2.py
# 18 Jan 2026 - Fixed cctbx symbol recognition errors
#
# Changes from previous version:
# 1. Fixed "Space group symbol not recognized" error by adding "Hall:" prefix.
# 2. Preserves strict logic for d-glides (4n) and hhl zones.

import json
from collections import OrderedDict, defaultdict
from cctbx import sgtbx
import sys

def evaluate_rule(h, k, l, rule_str):
    """Evaluates if a reflection satisfies a textual rule."""
    try:
        expr = rule_str.replace("=2n", "%2==0") \
                       .replace("=4n", "%4==0") \
                       .replace("=6n", "%6==0") \
                       .replace("=3n", "%3==0") \
                       .replace("-h", "(-h)") \
                       .replace("-k", "(-k)") \
                       .replace("-l", "(-l)")
        
        context = {'h': h, 'k': k, 'l': l}
        return eval(expr, {}, context)
    except Exception:
        return False

def get_condition_string(present_values, axis_name):
    """Deduces axial conditions (2n, 4n, 6n)."""
    if not present_values: 
        return None
    
    vals = sorted(list(set(present_values)))
    if not vals: 
        return None
    
    # Strongest conditions first
    if all(v % 6 == 0 for v in vals): return f"{axis_name}=6n"
    if all(v % 4 == 0 for v in vals): return f"{axis_name}=4n"
    if all(v % 3 == 0 for v in vals): return f"{axis_name}=3n"
    if all(v % 2 == 0 for v in vals): return f"{axis_name}=2n"
    
    return None

def check_zonal_from_points(present_tuples, idx1, idx2):
    """
    Deduces zonal condition (h+k=4n, 2h+l=4n, etc) from a list of tuple indices.
    Priority:
    1. Complex/Diamond Glides (4n) - Specific sums involving coefficients
    2. Standard Glides (4n) - Simple sums/diffs
    3. Axial Restrictions (2n) - h=2n must be checked BEFORE h+k=2n
    4. Centering Sums (2n/3n) - Weakest conditions
    """
    if not present_tuples: 
        return None
    
    names = {0: 'h', 1: 'k', 2: 'l'}
    n1, n2 = names[idx1], names[idx2]
    
    v1 = [p[idx1] for p in present_tuples]
    v2 = [p[idx2] for p in present_tuples]
    sums = [p[idx1] + p[idx2] for p in present_tuples]
    diffs = [p[idx1] - p[idx2] for p in present_tuples]
    
    # --- 1. PRIORITY: Diamond/Complex Glides (4n) ---
    # These are very specific to space groups like I-43d or Fd-3m (0kl)
    combo_2v1_plus_v2 = [2*p[idx1] + p[idx2] for p in present_tuples]
    combo_2v1_minus_v2 = [2*p[idx1] - p[idx2] for p in present_tuples]
    combo_v1_plus_2v2 = [p[idx1] + 2*p[idx2] for p in present_tuples]
    
    if all(c % 4 == 0 for c in combo_2v1_plus_v2): 
        return f"2{n1}+{n2}=4n"
    if all(c % 4 == 0 for c in combo_2v1_minus_v2): 
        return f"2{n1}-{n2}=4n"
    if all(c % 4 == 0 for c in combo_v1_plus_2v2): 
        return f"{n1}+2{n2}=4n"

    # --- 2. PRIORITY: Standard Glide Sums (4n) ---
    # Check this before Axial 2n. 
    # Example: Fd-3m 0kl requires k+l=4n.
    if all(s % 4 == 0 for s in sums): 
        return f"{n1}+{n2}=4n"
    if all(d % 4 == 0 for d in diffs): 
        return f"{n1}-{n2}=4n"

    # --- 3. PRIORITY: Axial Conditions (2n) ---

    if all(v % 2 == 0 for v in v1): 
        return f"{n1}=2n"
    if all(v % 2 == 0 for v in v2): 
        return f"{n2}=2n"

    # --- 4. PRIORITY: Weak Centering Sums (3n, 2n) ---
    if all(s % 3 == 0 for s in sums): 
        return f"{n1}+{n2}=3n"
    if all(d % 3 == 0 for d in diffs): 
        return f"{n1}-{n2}=3n"

    if all(s % 2 == 0 for s in sums): 
        return f"{n1}+{n2}=2n"
    if all(d % 2 == 0 for d in diffs): 
        return f"{n1}-{n2}=2n"
    
    return None


def is_redundant(specific_rule, general_rules, axes_indices):
    """Checks if a specific rule is already covered by general HKL rules."""
    if not general_rules:
        return False

    valid_points_under_general = []
    
    # Scan range 1-24 to catch 4n/6n periodicity
    for i in range(1, 25):
        h, k, l = 0, 0, 0
        active_axes = [key for key, v in axes_indices.items() if v is None]
        
        if len(active_axes) == 1:  # Axial check
            if axes_indices['h'] is None: h = i
            elif axes_indices['k'] is None: k = i
            elif axes_indices['l'] is None: l = i
            
            if all(evaluate_rule(h, k, l, r) for r in general_rules):
                valid_points_under_general.append((h, k, l))
                
        elif len(active_axes) == 2:  # Zonal check
            for j in range(1, 25):
                if axes_indices.get('diagonal'): # hhl
                    h, k, l = i, i, j
                elif axes_indices['l'] == 0:  # hk0
                    h, k = i, j
                elif axes_indices['k'] == 0:  # h0l
                    h, l = i, j
                elif axes_indices['h'] == 0:  # 0kl
                    k, l = i, j
                
                if all(evaluate_rule(h, k, l, r) for r in general_rules):
                    valid_points_under_general.append((h, k, l))

    if not valid_points_under_general:
        return False

    derived_rule = None
    
    if len(active_axes) == 1:
        idx_map = {'h': 0, 'k': 1, 'l': 2}
        axis_char = active_axes[0]
        idx = idx_map[axis_char]
        vals = [p[idx] for p in valid_points_under_general]
        derived_rule = get_condition_string(vals, axis_char)
        
    elif len(active_axes) == 2:
        idx_map = {'h': 0, 'k': 1, 'l': 2}
        idx1 = idx_map[active_axes[0]]
        idx2 = idx_map[active_axes[1]]
        derived_rule = check_zonal_from_points(valid_points_under_general, idx1, idx2)

    return derived_rule == specific_rule

def analyze_systematic_absences(space_group_info):
    sg = space_group_info.group()
    
    # --- 1. HKL Conditions (Centering) ---
    present_hkl = []
    for h in range(1, 8): 
        for k in range(1, 8):
            for l in range(1, 8):
                if not sg.is_sys_absent((h, k, l)):
                    present_hkl.append((h, k, l))
    
    hkl_rules = []
    if present_hkl:
        if all((h+k) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("h+k=2n")
        if all((h+l) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("h+l=2n")
        if all((k+l) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("k+l=2n")
        if all((h+k+l) % 2 == 0 for h, k, l in present_hkl): hkl_rules.append("h+k+l=2n")
        
        # Rhombohedral check (Obverse vs Reverse)
        if all((-h+k+l) % 3 == 0 for h, k, l in present_hkl): hkl_rules.append("-h+k+l=3n")
        if all((h-k+l) % 3 == 0 for h, k, l in present_hkl): hkl_rules.append("h-k+l=3n")
        
        # F-centering cleanup (only if strictly F and not I)
        if "h+k=2n" in hkl_rules and "h+l=2n" in hkl_rules and "k+l=2n" in hkl_rules:
            if "h+k+l=2n" not in hkl_rules:
                hkl_rules = ["h+k=2n", "h+l=2n", "k+l=2n"]

    conditions = OrderedDict()
    if hkl_rules:
        conditions['hkl'] = hkl_rules

    # --- 2. Zonal Conditions ---
    zones = [
        ('0kl', {'h': 0, 'k': None, 'l': None}, 1, 2),
        ('h0l', {'h': None, 'k': 0, 'l': None}, 0, 2),
        ('hk0', {'h': None, 'k': None, 'l': 0}, 0, 1),
        # Detect conditions on the diagonal (e.g., I-43d)
        ('hhl', {'h': None, 'k': None, 'l': None, 'diagonal': True}, 0, 2)
    ]
    
    for zone_name, axes_map, idx1, idx2 in zones:
        points = []
        for i in range(1, 20):
            for j in range(1, 20):
                hkl = [0, 0, 0]
                if zone_name == 'hhl':
                    hkl[0] = i
                    hkl[1] = i
                    hkl[2] = j
                else:
                    hkl[idx1] = i
                    hkl[idx2] = j
                
                if not sg.is_sys_absent(tuple(hkl)):
                    points.append(tuple(hkl))
        
        rule = check_zonal_from_points(points, idx1, idx2)
        if rule:
            if not is_redundant(rule, hkl_rules, axes_map):
                conditions[zone_name] = [rule]

    # --- 3. Axial Conditions ---
    axes_defs = [
        ('h00', {'h': None, 'k': 0, 'l': 0}, 0, 'h'),
        ('0k0', {'h': 0, 'k': None, 'l': 0}, 1, 'k'),
        ('00l', {'h': 0, 'k': 0, 'l': None}, 2, 'l')
    ]
    
    for axis_name, axes_map, idx, label in axes_defs:
        points = []
        for i in range(1, 30):
            hkl = [0, 0, 0]
            hkl[idx] = i
            if not sg.is_sys_absent(tuple(hkl)):
                points.append(hkl[idx])
        
        rule = get_condition_string(points, label)
        if rule:
            is_covered = False
            if is_redundant(rule, hkl_rules, axes_map):
                is_covered = True
            
            if not is_covered:
                # Check coverage by zones
                parents = []
                if axis_name == 'h00': parents = ['hk0', 'h0l', 'hhl']
                elif axis_name == '0k0': parents = ['hk0', '0kl']
                elif axis_name == '00l': parents = ['h0l', '0kl', 'hhl']
                
                for p in parents:
                    if p in conditions:
                        if is_redundant(rule, conditions[p], axes_map):
                            is_covered = True
                            break
            
            if not is_covered:
                conditions[axis_name] = [rule]
                
    return conditions

def generate_all_space_groups():
    all_data = defaultdict(lambda: {
        "number": 0, "standard_symbol": "", "crystal_system": "",
        "point_group": "", "centrosymmetric": False, "settings": []
    })
    
    iterator = sgtbx.space_group_symbol_iterator()
    processed_settings = defaultdict(set)
    
    print("Processing settings...")
    count = 0
    
    while True:
        try:
            symbols = iterator.next()
        except StopIteration:
            break
            
        if symbols.number() == 0: 
            break
        
        try:
            sg_num = symbols.number()
            hm_symbol = symbols.hermann_mauguin()
            hall = symbols.hall().strip() # Clean formatting
            
            uid = hall # Hall is the unique identifier for setting
            if uid in processed_settings[sg_num]: 
                continue
            processed_settings[sg_num].add(uid)
            
            # --- FIX: Explicitly tell cctbx this is a Hall symbol ---
            # Without "Hall:", cctbx attempts to parse things like "-P 1" or "P 2y" 
            # as Hermann-Mauguin and fails.
            sg_info = sgtbx.space_group_info(symbol=f"Hall: {hall}")
            sg = sg_info.group()
            
            if all_data[str(sg_num)]["number"] == 0:
                all_data[str(sg_num)]["number"] = sg_num
                all_data[str(sg_num)]["standard_symbol"] = sg_info.type().lookup_symbol()
                all_data[str(sg_num)]["crystal_system"] = str(sg.crystal_system()).lower()
                all_data[str(sg_num)]["point_group"] = str(sg.point_group_type())
                all_data[str(sg_num)]["centrosymmetric"] = sg.is_centric()

            conditions = analyze_systematic_absences(sg_info)
            desc = symbols.qualifier() if symbols.qualifier() else "standard"
            clean_sym = hm_symbol.replace(" ", "")
            
            all_data[str(sg_num)]["settings"].append({
                "symbol": clean_sym,
                "description": desc,
                "hall": hall,
                "reflection_conditions": conditions
            })
            
            count += 1
            if count % 50 == 0: 
                print(f"Processed {count} settings...", end="\r")
                sys.stdout.flush()

        except Exception as e:
            print(f"\n[!] Error processing SG {sg_num}: {e}")
            sys.stdout.flush()
            continue
            
    return all_data

def main():
    print("="*60)
    print("Space Group Generator - Fixed Symbol Parsing")
    print("="*60)
    
    data = generate_all_space_groups()
    
    total_settings = sum(len(x['settings']) for x in data.values())
    print(f"\nFinal processing complete.")
    print(f"Total space group numbers: {len(data)}")
    print(f"Total settings: {total_settings}")
    
    sorted_data = OrderedDict()
    for k in sorted(data.keys(), key=int):
        sorted_data[k] = data[k]
        sorted_data[k]["settings"].sort(key=lambda x: x["symbol"])
        
    output_wrapper = OrderedDict()
    output_wrapper["space_groups"] = sorted_data
        
    outfile = 'space_groups_final_v2.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(output_wrapper, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved to {outfile}")
    print("="*60)

if __name__ == "__main__":
    main()