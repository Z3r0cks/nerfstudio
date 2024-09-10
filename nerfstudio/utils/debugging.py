# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
      try:
         from nerfstudio.utils.deb_config_path import DEBUG_PATH
      except ImportError:
         print(ImportError)
         return
      
      file_path = f"{DEBUG_PATH}/{filename}.txt"
      with open(file_path, "w") as f:
         f.write(str(data))
         
   @staticmethod
   def print_call_stack() -> None:
      """Print the entire call stack."""
      from nerfstudio.utils.rich_utils import CONSOLE
      import inspect
      stack = inspect.stack()
      style = {"text": "yellow bold", "element": "bold green", "stack": "bold blue"}
      
      CONSOLE.print("\nTotal stack depth: ", len(stack)-1, style=style["stack"])
      for index, frame in enumerate(stack[1:], start=1):
         frame_info = inspect.getframeinfo(frame[0])
         CONSOLE.print(f"Stack level:{index}", style=style["text"])
         CONSOLE.print(f"Function: ", style=style["text"], end="")
         CONSOLE.print(frame_info.function, style=style["element"])
         CONSOLE.print(f"in file: ", style=style["text"],end="" )
         CONSOLE.print(frame_info.filename, style=style["element"])
         CONSOLE.print(f"on line: {frame_info.lineno} \n", style=style["text"])
         
   @staticmethod        
   def log(name: str, data) -> None:
      from nerfstudio.utils.rich_utils import CONSOLE
      
      CONSOLE.print(name, style="bold yellow")
      CONSOLE.print(data, style="bold green")