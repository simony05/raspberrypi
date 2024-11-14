import os
model_file = "saved_model/simplified_integer_quant.tflite"
os.chdir("metadata_model")
curr = os.getcwd()
print("Current working directory:", curr)
with open("shape_labels.txt", "r") as file:
    text = file.read()
print(text)

