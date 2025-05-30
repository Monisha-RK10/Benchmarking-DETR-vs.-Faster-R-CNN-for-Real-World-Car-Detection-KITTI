### dataset/

- `kitti.py`: Contains dataset classes and utility for loading the KITTI dataset for Faster R-CNN in PyTorch.

  - `KITTIDataset`: Basic dataset loader.
  - `FilteredKITTIDataset`: Loads images/labels based on IDs listed in `train.txt` or `val.txt`.
  - `collate_fn`: Custom collate function to handle variable number of annotations.

- `kitti_detr.py`: Contains dataset classes and utility for loading the KITTI dataset for DETR in PyTorch.

  - `CocoDetection': Parent class to load image file and raw annotations from JSON.
  - 'custom class': Uses parent class to load raw data and DetrImageProcessor processor to prep for model.
  - `collate_fn`: Custom collate function to pad the images to the largest H, W in the batch and creates a pixel mask.

