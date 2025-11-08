import re
import yaml
from typing import List, Dict, Any
from collections import Counter

import pydagmc
import openmc

# Handle both direct execution and module import
try:
    from .parse import parse
    from .openmc_conversion import get_openmc_materials
except ImportError:
    from parse import parse
    from openmc_conversion import get_openmc_materials



def parse_mesh_region_comments(filename: str) -> List[Dict[str, Any]]:
    """Parse mesh region/pseudo-cell information from MCNP comment blocks.
    
    Parameters
    ----------
    filename : str
        Path to MCNP input file (e.g., "configb.mcnp.i")
    
    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries, where each dictionary contains information
        for one mesh region/pseudo-cell block
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    regions = []
    current_block = []
    in_mesh_section = False
    
    for line in lines:
        # Check if we're entering the mesh region section
        if 'Mesh Region/Pseudo-Cell Information' in line:
            in_mesh_section = True
            continue
        
        # If we're in the mesh section
        if in_mesh_section:
            # Check if we hit a non-comment line (end of mesh section)
            if not line.strip().startswith('c'):
                break
            
            # Check if this is a separator (blank comment line or new region start)
            if line.strip() == 'c' or 'Attila Region #' in line:
                # Process the previous block if it exists
                if current_block:
                    region_data = parse_block('\n'.join(current_block))
                    if region_data:
                        regions.append(region_data)
                    current_block = []
            
            # Add line to current block
            if line.strip() != 'c':
                current_block.append(line)
    
    # Don't forget the last block
    if current_block:
        region_data = parse_block('\n'.join(current_block))
        if region_data:
            regions.append(region_data)
    
    return regions


def parse_block(block: str) -> Dict[str, Any]:
    """Parse a single mesh region block into a dictionary."""
    region_data = {}
    
    # Parse Attila Region #
    if m := re.search(r'c\s+Attila Region #\s*:\s*(\d+)', block):
        region_data['attila_region_id'] = int(m.group(1))
    
    # Parse Attila Region Name
    if m := re.search(r'c\s+Attila Region Name\s*:\s*"([^"]+)"', block):
        region_data['attila_region_name'] = m.group(1)
    
    # Parse Abaqus Part #
    if m := re.search(r'c\s+Abaqus Part #\s*:\s*(\d+)', block):
        region_data['abaqus_part_id'] = int(m.group(1))
    
    # Parse Abaqus Part Name
    if m := re.search(r'c\s+Abaqus Part Name\s*:\s*"([^"]+)"', block):
        region_data['abaqus_part_name'] = m.group(1)
    
    # Parse MCNP Pseudo-cell #
    if m := re.search(r'c\s+MCNP Pseudo-cell #\s*:\s*(\d+)', block):
        region_data['mcnp_pseudocell_id'] = int(m.group(1))
    
    # Parse Material
    if m := re.search(r'c\s+Material\s*:\s*"([^"]+)"', block):
        region_data['material_name'] = m.group(1)
    
    # Parse MCNP Material (optional)
    if m := re.search(r'c\s+MCNP Material\s*:\s*(\w+)', block):
        region_data['mcnp_material'] = m.group(1)
    
    # Parse Density (optional)
    if m := re.search(r'c\s+Density\s*:\s*([\d.e+-]+)\s*g/cc', block):
        region_data['density_gcc'] = float(m.group(1))
    
    # Parse Mesh Data - Meshed Volume
    if m := re.search(r'c\s+Meshed Volume\s*:\s*([\d.e+-]+)\s*cm\*\*3', block):
        region_data['meshed_volume_cm3'] = float(m.group(1))
    
    # Parse Mesh Data - # Cells
    if m := re.search(r'c\s+#\s*Cells\s*:\s*(\d+)', block):
        region_data['num_cells'] = int(m.group(1))
    
    # Parse Mesh Data - % of Total Cells
    if m := re.search(r'c\s+%\s*of Total Cells\s*:\s*([\d.]+)%', block):
        region_data['percent_total_cells'] = float(m.group(1))
    
    return region_data


def build_volume_material_maps(
    rtt_file: str,
    mcnp_file: str,
    h5m_file: str = 'mesh.h5m',
    export_h5m_file: bool = True,
    export_materials: bool = True,
    export_group_map: bool = True,
    validate: bool = True
) -> tuple[List[Dict[str, Any]], List[openmc.Material], Dict[tuple, List[int]]]:
    """Build volume, material, and group maps from RTT and MCNP input files.
    
    This function processes RTT mesh files and MCNP input files to create:
    1. A volume map linking RTT volumes to MCNP pseudo-cells and materials
    2. A material map with OpenMC materials configured with names and densities
    3. A group map organizing volumes by material for OpenMC geometry
    4. An H5M file with materials assigned to volumes (if requested)
    
    Parameters
    ----------
    rtt_file : str
        Path to RTT mesh file (e.g., "model.rtt")
    mcnp_file : str
        Path to MCNP input file (e.g., "input.mcnp.i")
    h5m_file : str, optional
        Path for output H5M file with material assignments (default: "mesh.h5m")
    export_h5m_file : bool, optional
        If True, export H5M file with materials assigned to volumes (default: True)
    export_materials : bool, optional
        If True, export materials to materials.xml (default: True)
    export_group_map : bool, optional
        If True, export group map to group_map.yaml (default: True)
    validate : bool, optional
        If True, perform validation checks on volume and material consistency (default: True)
    
    Returns
    -------
    volume_map : List[Dict[str, Any]]
        List of dictionaries containing volume information with keys:
        - 'pydagmc_id': RTT/DAGMC volume ID
        - 'mcnp_pseudocell_id': MCNP pseudo-cell ID
        - 'attila_region_name': Region name from Attila
        - 'material_name': Material name
        - 'mcnp_material_id': MCNP material ID (if not void)
        - 'material_density': Tuple of (units, value) or None
    
    material_map : List[openmc.Material]
        List of OpenMC Material objects with names and densities set
    
    group_map : Dict[tuple, List[int]]
        Dictionary mapping (material_label, material_id) to list of volume IDs
        Example: {('mat:Steel', 5): [1, 2, 3], ('mat:void', 0): [4, 5]}
    
    Side Effects
    ------------
    - Writes materials.xml if export_materials=True
    - Writes group_map.yaml if export_group_map=True
    - Writes H5M file with material assignments if export_h5m_file=True
    
    Raises
    ------
    ValueError
        If material names are inconsistent for the same MCNP material ID
    NotImplementedError
        If material densities are inconsistent (requires material duplication)
    
    Examples
    --------
    >>> volume_map, material_map, group_map = build_volume_material_maps(
    ...     'model.rtt', 'input.mcnp.i'
    ... )
    
    >>> # Export only specific outputs
    >>> volume_map, material_map, group_map = build_volume_material_maps(
    ...     'model.rtt', 'input.mcnp.i',
    ...     h5m_file='custom_mesh.h5m',
    ...     export_h5m_file=True,
    ...     export_materials=False,
    ...     export_group_map=False
    ... )
    """
    # Load RTT model and extract volumes
    model = pydagmc.Model(rtt_file)
    rtt_vols = model.volumes
    
    # Parse MCNP file
    mcnp_parsed = parse(mcnp_file)
    mcnp_vols = mcnp_parsed[0]
    
    # Parse mesh region comments from MCNP file
    comment_vols = parse_mesh_region_comments(mcnp_file)
    
    # Validation checks if requested
    if validate:
        _validate_volumes(rtt_vols, comment_vols)
    
    # Build volume map
    volume_map = _build_volume_map(rtt_vols, comment_vols, mcnp_vols)
    
    # Build and configure material map
    mcnp_materials = mcnp_parsed[2]['materials']
    material_map = list(get_openmc_materials(mcnp_materials).values())
    _configure_materials(material_map, volume_map)
    
    # Build group map (volume-material associations)
    group_map = _build_group_map(volume_map, material_map)
    
    # Export materials if requested
    if export_materials:
        openmc.Materials(material_map).export_to_xml()
    
    # Export group map if requested
    if export_group_map:
        _export_group_map(group_map)

    if export_h5m_file:
        model = pydagmc.Model(rtt_file)
        # assign materials to volumes
        model.add_groups(group_map)
        model.mb.write_file(h5m_file)
    
    return volume_map, material_map, group_map


def _validate_volumes(rtt_vols, comment_vols):
    """Validate consistency between RTT volumes and comment volumes."""
    rtt_vol_ids = [vol.id for vol in rtt_vols]
    comment_vol_ids = [c['attila_region_id'] for c in comment_vols]
    
    counts = Counter(comment_vol_ids)
    repeated = [num for num, c in counts.items() if c > 1]
    
    if repeated:
        print(f"Warning: Repeated Attila region IDs in comments: {repeated}")
        for num in repeated:
            print(f"  ID {num} appears {counts[num]} times")
    
    missing = []
    for c in comment_vols:
        attila_id = c['attila_region_id']
        if attila_id not in rtt_vol_ids:
            missing.append(attila_id)
    
    if missing:
        print(f"Warning: {len(missing)} Attila region IDs from comments not found in RTT volumes")


def _build_volume_map(rtt_vols, comment_vols, mcnp_vols):
    """Build volume map linking RTT volumes to MCNP data."""
    # Create lookup dictionaries for O(1) access
    comment_by_attila_id = {c['attila_region_id']: c for c in comment_vols}
    mcnp_vol_by_id = {m['id']: m for m in mcnp_vols}
    
    volume_map = []
    
    for r in rtt_vols:
        d = {'pydagmc_id': r.id}
        
        # O(1) lookup for comment data
        if r.id in comment_by_attila_id:
            c = comment_by_attila_id[r.id]
            d['mcnp_pseudocell_id'] = c['mcnp_pseudocell_id']
            d['attila_region_name'] = c['attila_region_name']
            d['material_name'] = c['material_name']
            
            # Safely extract mcnp_material_id
            if 'mcnp_material' in c:
                d['mcnp_material_id'] = int(c['mcnp_material'][1:])
            else:
                d['mcnp_material'] = None
            
            # O(1) lookup for mcnp volume data
            if 'mcnp_pseudocell_id' in d and d['mcnp_pseudocell_id'] in mcnp_vol_by_id:
                m = mcnp_vol_by_id[d['mcnp_pseudocell_id']]
                
                if m['density'] is not None:
                    value = abs(m['density'])
                    units = 'atom/b-cm' if m['density'] > 0 else 'g/cm3'
                    d['material_density'] = (units, value)
                else:
                    d['material_density'] = None
        
        volume_map.append(d)
    
    return volume_map


def _configure_materials(material_map, volume_map):
    """Configure material names and densities from volume map."""
    # Check for inconsistencies and build material info lookup
    material_info_by_id = {}
    
    for v in volume_map:
        if 'mcnp_material_id' in v:
            mat_id = v['mcnp_material_id']
            if mat_id not in material_info_by_id:
                material_info_by_id[mat_id] = {'names': set(), 'densities': set()}
            
            material_info_by_id[mat_id]['names'].add(v['material_name'])
            material_info_by_id[mat_id]['densities'].add(v['material_density'])
    
    # Check for inconsistencies
    for mat_id, info in material_info_by_id.items():
        if len(info['names']) > 1:
            raise ValueError(
                f"Material name inconsistency for MCNP material ID {mat_id}: {info['names']}"
            )
        
        if len(info['densities']) > 1:
            msg = (
                f"Material density inconsistency for MCNP material ID {mat_id}: "
                f"{info['densities']}.\nNeed to duplicate the material in OpenMC"
            )
            raise NotImplementedError(msg)
    
    # Create lookup dictionary for O(1) access to materials
    material_by_id = {m.id: m for m in material_map}
    
    # Update material names and densities
    for mat_id, info in material_info_by_id.items():
        if mat_id in material_by_id:
            m = material_by_id[mat_id]
            m.name = info['names'].pop()
            
            density = info['densities'].pop()
            if density is not None:
                m.set_density(density[0], density[1])


def _build_group_map(volume_map, material_map):
    """Build group map organizing volumes by material."""
    group_map = {}
    
    # Create material name to material lookup for O(1) access
    material_by_name = {m.name: m for m in material_map}
    
    for v in volume_map:
        mat_name = v.get('material_name')
        
        # Check if this material exists in our material_map
        if mat_name and mat_name in material_by_name:
            m = material_by_name[mat_name]
            key = (f'mat:{m.name}', m.id)
        else:
            # No matching material found - it's void
            key = ('mat:void', 0)
        
        if key not in group_map:
            group_map[key] = []
        group_map[key].append(v['pydagmc_id'])
    
    return group_map


def _export_group_map(group_map, filename='group_map.yaml'):
    """Export group map to YAML file.
    
    Preserves tuple keys exactly as they are in the Python dictionary.
    """
    # Convert numpy types to Python types for YAML serialization
    yaml_group_map = {
        key: [int(v) for v in vol_ids]
        for key, vol_ids in group_map.items()
    }
    
    # Use unsafe dump to allow Python tuples
    with open(filename, 'w') as f:
        yaml.dump(yaml_group_map, f, default_flow_style=False, sort_keys=False)
    
    print(f"Group map exported to: {filename}")
    print(f"Note: Load with yaml.unsafe_load() or yaml.full_load() to preserve tuples")


# Command line interface
if __name__ == '__main__':
    import argparse
    import sys
    
    # Check if being run as a script (not as module)
    if __package__ is None:
        print("Error: This script should be run as a module.")
        print("\nPlease use one of the following:")
        print("  python -m openmc_mcnp_adapter.rtt_comments_reader <rtt_file> <mcnp_file>")
        print("  cd /path/to/openmc_mcnp_adapter && python -m src.openmc_mcnp_adapter.rtt_comments_reader <rtt_file> <mcnp_file>")
        print("\nOr from within Python:")
        print("  from openmc_mcnp_adapter.rtt_comments_reader import build_volume_material_maps")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description='Build volume, material, and group maps from RTT and MCNP files.'
    )
    parser.add_argument('rtt_file', help='Path to RTT mesh file')
    parser.add_argument('mcnp_file', help='Path to MCNP input file')
    parser.add_argument('--no-export', action='store_true', help='Skip materials.xml export')
    parser.add_argument('--no-group-map', action='store_true', help='Skip group_map.json export')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation checks')
    
    args = parser.parse_args()
    
    volume_map, material_map, group_map = build_volume_material_maps(
        rtt_file=args.rtt_file,
        mcnp_file=args.mcnp_file,
        export_materials=not args.no_export,
        export_group_map=not args.no_group_map,
        validate=not args.no_validate
    )
    
    print(f"\nProcessed {len(volume_map)} volumes")
    print(f"Found {len(material_map)} materials")
    print(f"Created {len(group_map)} material groups")