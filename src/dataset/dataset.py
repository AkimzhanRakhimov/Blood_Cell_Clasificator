label_classes={
    "EOSINOPHIL":0,
    "LYMPHOCYTE":1,
    "MONOCYTE":2,
    "NEUTROPHIL":3
}
# dataset processing class
class CellDataset(Dataset):
  def __init__(self,root_dir,label_classes,transform=None):
    self.transform=transform
    self.samples=[]

    for cell_type,label in label_classes.items():
      directory=Path(root_dir)/cell_type
      for file_path in directory.iterdir():
        self.samples.append((file_path,label))
  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    image_path,label=self.samples[index]
    image=Image.open(image_path).convert("RGB")

    if self.transform:
      image=self.transform(image)
    return image, label
