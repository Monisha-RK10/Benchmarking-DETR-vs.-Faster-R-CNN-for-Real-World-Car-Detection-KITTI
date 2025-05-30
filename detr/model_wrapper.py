import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

# Step 5
# Plugging the HuggingFace DETR model into the Lightning pipeline.
# Example: Exposing the forward of the wrapped HuggingFace model.
# Different learning rates to improve convergence and stability.
# Resnet (ImageNet, no aggressive update) -> small learning rate to fine-tune gently.
# Transformer (need to learn actively) -> higher learning rate to adapt faster.

CHECKPOINT = "facebook/detr-resnet-50"

id2label = {0: "car"}
label2id = {v: k for k, v in id2label.items()}

class Detr(pl.LightningModule): # DETR: subclass of pl.LightningModule to load and call the model, train, validate, and optimize it
    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader=None, val_dataloader=None):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"validation_{k}", v.item())
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader
