data_dir = "../data/NCT-CRC/"
csv_name = "files.csv"
csv_path = data_dir + csv_name
from script.data_processing.image_transforms import get_transforms
transforms = get_transforms()
bins = 1
backbone = "resnet18"