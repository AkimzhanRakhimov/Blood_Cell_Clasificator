# archive unpacking
import zipfile
ZIP="/content/drive/MyDrive/blood_cells_dataset/archive.zip"
with zipfile.ZipFile("/content/drive/MyDrive/blood_cells_dataset/archive.zip","r") as zip_ref:
  zip_ref.extractall("/content/drive/MyDrive/blood_cells_dataset/")

