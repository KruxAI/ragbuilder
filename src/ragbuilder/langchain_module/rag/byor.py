def LangchainByor(file_path):
    """
    Reads the contents of a file and returns it.
    
    Parameters:
    file_path (str): The path to the file to be read.
    
    Returns:
    str: The contents of the file.
    """
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
        return contents
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"