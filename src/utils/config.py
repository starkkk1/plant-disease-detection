import os
import yaml

def merge_dicts(dict1, dict2):
    """
    Recursively merges dict2 into dict1.
    """
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.
    If the configuration file specifies a 'defaults' key, it will recursively
    load the base config and merge the current config overrides onto it.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
        
    if 'defaults' in config:
        base_path = config['defaults']
        # Resolve path relative to config_path's directory if it is not absolute
        if not os.path.isabs(base_path):
            config_dir = os.path.dirname(config_path)
            # Try resolving relative to the config file's directory
            candidate_path = os.path.join(config_dir, os.path.basename(base_path))
            if os.path.exists(candidate_path):
                base_path = candidate_path
            elif os.path.exists(base_path):
                pass
            else:
                base_path = os.path.join(config_dir, base_path)
                
        base_config = load_config(base_path)
        # Remove defaults key from config
        config.pop('defaults', None)
        config = merge_dicts(base_config, config)
        
    return config
