import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Code for deducing the class weights between light and dark skintone. Useful for addressing class imbalance in code.

MetaData=pd.read_csv("myimages/metadata.csv")
def fitzpatrick_converter(entry):
    if entry == "I":
        return 0
    elif entry == "II":
        return 0
    elif entry == "III":
        return 1
    elif entry == "IV":
        return 1
    elif entry == "V":
        return 1
    elif entry == "VI":
        return 1
    else:
        return "Error"
MetaData["SkinTone"]=MetaData["fitzpatrick_skin_type"].apply(lambda x: fitzpatrick_converter(x))
y=MetaData["SkinTone"]

classimblanace=compute_class_weight(class_weight="balanced", classes=y.unique(), y=y)
print(classimblanace)

print(MetaData.columns)

print(MetaData[["pixels_x", "pixels_y"]].columns)
count = len(MetaData[(MetaData["pixels_x"] < 224) | (MetaData["pixels_y"] < 224)])
print(count)
print(MetaData.shape[0])
