# settings.py
import os
from os.path import join, dirname
from dotenv import load_dotenv
 
dotenv_path = join(dirname(__file__), 'app.env')
load_dotenv(dotenv_path)
 
# Accessing variables.
status = os.getenv('EPOCHS')
 
# Using variables.
print(status)