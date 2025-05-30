### dataset/

- `kitti.py`: Contains dataset classes and utility for loading the KITTI dataset in PyTorch.

  - `KITTIDataset`: Basic dataset loader.
  - `FilteredKITTIDataset`: Loads images/labels based on IDs listed in `train.txt` or `val.txt`.
  - `collate_fn`: Custom collate function to handle variable number of annotations.
