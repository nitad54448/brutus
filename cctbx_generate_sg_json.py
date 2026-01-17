# cctbx_generate_sg_json.py
# version 17 janv 2026
# output wrapped in a space_group, required by brutus.html and worker-logic.js
# this version outputs 530 space groups, reflections pruned, i.e. h00, h=2n may already be included as hk0 h+k=2n, 
# ITC rules are more verbose, but not needed, (I think...)

import json
from collections import OrderedDict, defaultdict
from cctbx import sgtbx, crystal, miller
from cctbx.array_family import flex
import sys

# logic of cctbx

def evaluate_rule(h, k, l, rule_str):
    """Evaluates if a reflection satisfies a textual rule."""
    try:
        # Convert standard crystallographic rules to Python expressions
        expr = rule_str.replace("=2n", "%2==0") \
                       .replace("=4n", "%4==0") \
                       .replace("=6n", "%6==0") \
                       .replace("=3n", "%3==0") \
                       .replace("-h", "(-h)")
        
        context = {'h': h, 'k': k, 'l': l}
        return eval(expr, {}, context)
    except Exception:
        return False

def get_condition_string(present_values, axis_name):
    """Deduces the condition string (2n, 4n, etc) from a list of integer values."""
    if not present_values: return None
    
    # need unique values to determine the pattern
    vals = sorted(list(set(present_values)))
    
    if not vals: return None
    
    # Check strongest conditions first
    if all(v % 6 == 0 for v in vals): return f"{axis_name}=6n"
    if all(v % 4 == 0 for v in vals): return f"{axis_name}=4n"
    if all(v % 3 == 0 for v in vals): return f"{axis_name}=3n"
    if all(v % 2 == 0 for v in vals): return f"{axis_name}=2n"
    
    return None

def check_zonal_from_points(present_tuples, idx1, idx2):
    """Deduces zonal condition (h+k=2n, etc) from a list of tuple indices."""
    if not present_tuples: return None
    
    names = {0: 'h', 1: 'k', 2: 'l'}
    n1, n2 = names[idx1], names[idx2]
    
    v1 = [p[idx1] for p in present_tuples]
    v2 = [p[idx2] for p in present_tuples]
    sums = [p[idx1] + p[idx2] for p in present_tuples]
    
    # Check for simple axial conditions appearing in the zone
    if all(v % 2 == 0 for v in v1): return f"{n1}=2n"
    if all(v % 2 == 0 for v in v2): return f"{n2}=2n"
    
    # Check coupled conditions
    if all(s % 4 == 0 for s in sums): return f"{n1}+{n2}=4n"
    if all(s % 2 == 0 for s in sums): return f"{n1}+{n2}=2n"
    
    return None

def is_redundant(specific_rule, general_rules, axes_indices):
    """
    Determines if a 'specific_rule' (e.g., h00: h=2n) is redundant 
    because it is already enforced by 'general_rules' (e.g., hkl: h+k=2n).
    """
    if not general_rules:
        return False

    valid_points_under_general = []
    
    # Range 1-12 is sufficient to detect 2,3,4,6 periodicity
    rng = range(1, 13)
    
    for i in rng:
        h, k, l = 0, 0, 0
        
        # Determine which axes vary (value=None) and which are fixed (value=0)
        active_axes = [k for k, v in axes_indices.items() if v is None]
        
        if len(active_axes) == 1: # Axial (h00, 0k0, 00l)
            if axes_indices['h'] is None: h = i
            elif axes_indices['k'] is None: k = i
            elif axes_indices['l'] is None: l = i
            
            passes = True
            for rule in general_rules:
                if not evaluate_rule(h, k, l, rule):
                    passes = False
                    break
            if passes:
                valid_points_under_general.append((h, k, l))
                
        elif len(active_axes) == 2: # Zonal (hk0, etc)
            for j in range(1, 13):
                h, k, l = 0, 0, 0
                if axes_indices['l'] == 0: # hk0
                    h, k = i, j
                elif axes_indices['k'] == 0: # h0l
                    h, l = i, j
                elif axes_indices['h'] == 0: # 0kl
                    k, l = i, j
                
                passes = True
                for rule in general_rules:
                    if not evaluate_rule(h, k, l, rule):
                        passes = False
                        break
                if passes:
                    valid_points_under_general.append((h, k, l))

    if not valid_points_under_general:
        return False

    # Re-derive rule from allowed points
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


# main paert


def analyze_systematic_absences(space_group_info):
    sg = space_group_info.group()
    
    # 1. HKL Conditions (Centering)
    present_hkl = []
    for h in range(1, 6):
        for k in range(1, 6):
            for l in range(1, 6):
                if not sg.is_sys_absent((h, k, l)):
                    present_hkl.append((h, k, l))
    
    hkl_rules = []
    
    if not present_hkl: return {} 

    if all((h+k)%2==0 for h,k,l in present_hkl): hkl_rules.append("h+k=2n")
    if all((h+l)%2==0 for h,k,l in present_hkl): hkl_rules.append("h+l=2n")
    if all((k+l)%2==0 for h,k,l in present_hkl): hkl_rules.append("k+l=2n")
    if all((h+k+l)%2==0 for h,k,l in present_hkl): hkl_rules.append("h+k+l=2n")
    if all((-h+k+l)%3==0 for h,k,l in present_hkl): hkl_rules.append("-h+k+l=3n")
    
    # Consolidate F-centering
    if "h+k=2n" in hkl_rules and "h+l=2n" in hkl_rules and "k+l=2n" in hkl_rules:
        hkl_rules = ["h+k=2n", "h+l=2n", "k+l=2n"]

    conditions = OrderedDict()
    if hkl_rules:
        conditions['hkl'] = hkl_rules

    # 2. Zonal Conditions
    zones = [
        ('0kl', {'h': 0, 'k': None, 'l': None}, 1, 2),
        ('h0l', {'h': None, 'k': 0, 'l': None}, 0, 2),
        ('hk0', {'h': None, 'k': None, 'l': 0}, 0, 1)
    ]
    
    for zone_name, axes_map, idx1, idx2 in zones:
        points = []
        for i in range(1, 15):
            for j in range(1, 15):
                hkl = [0,0,0]
                hkl[idx1] = i; hkl[idx2] = j
                if not sg.is_sys_absent(tuple(hkl)):
                    points.append(tuple(hkl))
        
        rule = check_zonal_from_points(points, idx1, idx2)
        if rule:
            if not is_redundant(rule, hkl_rules, axes_map):
                conditions[zone_name] = [rule]

    # 3. Axial Conditions
    axes_defs = [
        ('h00', {'h': None, 'k': 0, 'l': 0}, 0, 'h'),
        ('0k0', {'h': 0, 'k': None, 'l': 0}, 1, 'k'),
        ('00l', {'h': 0, 'k': 0, 'l': None}, 2, 'l')
    ]
    
    for axis_name, axes_map, idx, label in axes_defs:
        points = []
        for i in range(1, 25):
            hkl = [0,0,0]
            hkl[idx] = i
            if not sg.is_sys_absent(tuple(hkl)):
                points.append(hkl[idx])
        
        rule = get_condition_string(points, label)
        if rule:
            is_covered = False
            
            if is_redundant(rule, hkl_rules, axes_map):
                is_covered = True
            
            if not is_covered:
                parents = []
                if axis_name == 'h00': parents = ['hk0', 'h0l']
                elif axis_name == '0k0': parents = ['hk0', '0kl']
                elif axis_name == '00l': parents = ['h0l', '0kl']
                
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
    
    print("Processing...")
    count = 0
    
    while True:
        try:
            symbols = iterator.next()
        except StopIteration:
            break
            
        if symbols.number() == 0: break
        
        try:
            sg_num = symbols.number()
            hm_symbol = symbols.hermann_mauguin()
            hall = symbols.hall()
            
            uid = f"{hm_symbol}_{hall}"
            if uid in processed_settings[sg_num]: continue
            processed_settings[sg_num].add(uid)
            
            sg_info = sgtbx.space_group_info(symbol=hm_symbol)
            sg = sg_info.group()
            
            if all_data[str(sg_num)]["number"] == 0:
                all_data[str(sg_num)]["number"] = sg_num
                all_data[str(sg_num)]["standard_symbol"] = sg_info.type().lookup_symbol()
                all_data[str(sg_num)]["crystal_system"] = str(sg.crystal_system()).lower()
                all_data[str(sg_num)]["point_group"] = str(sg.point_group_type())
                all_data[str(sg_num)]["centrosymmetric"] = sg.is_centric()

            conditions = analyze_systematic_absences(sg_info)
            desc = symbols.qualifier() if symbols.qualifier() else "standard"
            
            all_data[str(sg_num)]["settings"].append({
                "symbol": hm_symbol.replace(" ", ""),
                "description": desc,
                "reflection_conditions": conditions
            })
            
            count += 1
            if count % 10 == 0: 
                print(f"Processed {count} settings...", end="\r")
                sys.stdout.flush()

        except Exception as e:
            print(f"\n[!] Error processing SG {sg_num} ({hm_symbol}): {e}")
            sys.stdout.flush()
            continue
            
    return all_data

def main():
    print("="*60)
    print("Space Group Generator")
    print("="*60)
    
    data = generate_all_space_groups()
    
    print(f"\nFinal processing complete. Total settings: {sum(len(x['settings']) for x in data.values())}")
    
    # Sort the dictionary keys (Space group numbers)
    sorted_data = OrderedDict()
    for k in sorted(data.keys(), key=int):
        sorted_data[k] = data[k]
        sorted_data[k]["settings"].sort(key=lambda x: x["symbol"])
        
    # WRAP IN "space_groups" PROPERTY
    output_wrapper = OrderedDict()
    output_wrapper["space_groups"] = sorted_data
        
    outfile = 'space_groups_final.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(output_wrapper, f, indent=2, ensure_ascii=False)
        
    print(f"Saved to {outfile}")

if __name__ == "__main__":
    main()