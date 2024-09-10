import os
from datasets import Dataset
from tqdm import tqdm

folder_directory = "../bookcorpus"
file_list = [ os.path.join(folder_directory, file) for file in os.listdir(folder_directory) if file.endswith(".txt")]

data = []
for file_path in tqdm(file_list):
    with open(file_path, "r") as file:
        file_content = file.read()
        data.append({"text": file_content})
        
dataset = Dataset.from_list(data)
dataset.save_to_disk("/home/user19/bag/6.BERT/dataset")
