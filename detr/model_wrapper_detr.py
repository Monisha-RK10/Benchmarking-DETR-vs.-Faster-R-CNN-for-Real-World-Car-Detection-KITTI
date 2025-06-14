# Step 4 for DETR
# Plugging the HuggingFace DETR model into the Lightning pipeline (more info @ https://lightning.ai/docs/pytorch/stable/common/lightning_module.html).
# Different learning rates to improve convergence and stability.
# Resnet (ImageNet, no aggressive update) -> small learning rate to fine-tune gently.
# Transformer (need to learn actively) -> higher learning rate to adapt faster.

import pytorch_lightning as pl                                                                     # Lightweight training wrapper over raw PyTorch 
from transformers import DetrForObjectDetection
import torch

CHECKPOINT = "facebook/detr-resnet-50"                                                             # Finetuning everything (backbone + transformer + head)

id2label = {0: "car"}
label2id = {v: k for k, v in id2label.items()}

class Detr(pl.LightningModule):                                                                    # DETR: subclass of pl.LightningModule to load and call the model, train, validate, and optimize it
    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader, val_dataloader):           # Extending the parent class
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.lr = lr                                                                                # Goes in configure_optimizers
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self._train_dataloader = train_dataloader                                                   # Pass them to .fit() later or define train_dataloader()/val_dataloader() methods 
        self._val_dataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):                                                    # Overriding the forward() method
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)                         # pixel value: tensor shape [B, 3, H, W], pixel mask: tensor shape [B, H, W]

    def common_step(self, batch, batch_idx):                                                        # Custom helper method, not overriding, not part of PyTorch Lightning’s API
        pixel_values = batch["pixel_values"]                                                        # Comes from the 'custom collate_fn()'
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)       # Calls DETR's forward() (inherited from HuggingFace)
        loss = outputs.loss                                                                         # Scalar 
        loss_dict = outputs.loss_dict                                                               # Breakdown of loss components (e.g., class loss, bbox loss, GIoU)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):                                                      # Overriding the train() method
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training/loss", loss, on_step=True, on_epoch=True, prog_bar=True)                 # on_step: log every batch, on_epoch: average over the epoch
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())
        return loss                                                                                 # Lightning uses this to perform loss.backward() and optimizer step internally

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)              # Tracks losses in TensorBoard
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad] 
            },                                                                                      # name (n), parameter (p). Example: ("backbone.body.layer1.0.conv1.weight", Parameter(...))
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self._train_dataloader                                                               # internal use only, exposes internal dataloader

    def val_dataloader(self):
        return self._val_dataloader                                                                 # internal use only
