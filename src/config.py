# config
device="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=32
torch.manual_seed(42)
torch.cuda.manual_seed(42)

drive.mount("/content/drive")
dataset_path="/content/drive/MyDrive/blood_cells_dataset/dataset2-master/dataset2-master/images/TRAIN/"
MODEL_SAVE_PATH="/content/drive/MyDrive/blood_cells_dataset/dataset2-master/dataset2-master/images/model.pth"
