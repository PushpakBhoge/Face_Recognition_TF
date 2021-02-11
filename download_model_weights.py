from zipfile import ZipFile 
import gdown
import os

# original link - https://drive.google.com/file/d/1SH-9ApSaD78OS-AdKqZeyLMLHe-_GA8F/view?usp=sharing

url = 'https://drive.google.com/uc?id=1SH-9ApSaD78OS-AdKqZeyLMLHe-_GA8F'
output_file = "model_weights.zip"

gdown.download(url, output_file, quiet=False)

# opening the zip file in READ mode 
with ZipFile(output_file, 'r') as zip: 
    zip.printdir() 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('weights extracted Done!')

os.remove(output_file)
