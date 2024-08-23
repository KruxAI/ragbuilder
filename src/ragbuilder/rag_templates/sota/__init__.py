import os
import importlib

# Get the directory of this file
module_dir = os.path.dirname(__file__)

# Track whether any modules are found and imported
modules_found = False

# Iterate over all files in the directory
for filename in os.listdir(module_dir):
    # Only consider .py files and skip __init__.py itself
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]  # Strip .py extension
        # Dynamically import the module
        importlib.import_module(f'.{module_name}', package=__name__)
        modules_found = True  # Mark that at least one module was found

# If no modules were found, print a message or handle as needed
if not modules_found:
    print("No Python modules found in the 'byor' folder.")
