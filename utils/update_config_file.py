import yaml
import argparse

def update_config(config_path, updates):
    """
    Update the given YAML configuration file with new values.

    Parameters:
    - config_path (str): Path to the YAML file.
    - updates (dict): Dictionary containing key-value pairs to update in the config.

    Returns:
    - None
    """
    # Load the existing configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Apply updates
    for key, value in updates.items():
        if key in config:
            print(f"Updating '{key}' from '{config[key]}' to '{value}'")
        else:

            print(f"Adding new key '{key}' with value '{value}'")
        config[key] = value

    # Save the updated configuration back to the file
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    print(f"Configuration updated successfully in {config_path}")

# Command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update YAML configuration file with specified key-value pairs.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--updates", nargs="+", required=True, help="Key-value pairs to update, e.g., key1=value1 key2=value2.")

    args = parser.parse_args()

    # Parse updates into a dictionary
    updates_dict = {}
    for update in args.updates:
        key, value = update.split("=")
        updates_dict[key] = value

    # Update the configuration file
    update_config(args.config_file, updates_dict)
