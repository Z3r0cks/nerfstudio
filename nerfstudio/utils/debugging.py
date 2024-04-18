from nerfstudio.utils.deb_config_path import DEBUG_PATH

class Debugging:
   
   """ Class for debugging purposes. """
   
   def __init__(self, verbose: bool = False):
      self.verbose = verbose
   
   @staticmethod   
   def write_to_file(data, filename: str) -> None:
      """ write prompt to file. Use deb_config_path.DEBUG_PATH as path on your system.
      
      Args: 
         data: the data to write to the file, the name of the file.
         filename: the name of the file.
      """
      file_path = f"{DEBUG_PATH}/{filename}.txt"
      with open(file_path, "w") as f:
         f.write(str(data))