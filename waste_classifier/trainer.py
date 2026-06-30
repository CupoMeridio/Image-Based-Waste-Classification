"""
Waste Classifier Trainer - Modulo principale per l'addestramento
Gestisce dataset, modelli, training e salvataggio risultati
"""

import json
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import random
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich import box
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import copy

__version__ = "1.2.0"



def set_global_seed(seed: int = 42):
    """
    Imposta il seed globale per PyTorch, NumPy e Python standard,
    garantendo la massima riproducibilità possibile tra run diversi.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    """
    Garantisce che ogni worker di un DataLoader riceva un seed NumPy univoco.
    Evita il problema del rumore correlato nell'augmentation (stessi numeri 
    casuali generati in parallelo da worker forked).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class FocalLoss(nn.Module):
    """
    Focal Loss per classificazione multi-classe.

    Riduce il peso degli esempi "facili" (già classificati correttamente con
    alta confidenza) in modo che il training si concentri sugli esempi difficili
    e sulle classi minoritarie.

    Riferimento: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Args:
        alpha (Tensor, opzionale): Vettore di pesi per classe (uno per classe).
            Solitamente l'inverso della frequenza di ogni classe. Se None,
            nessuna ponderazione per classe viene applicata.
        gamma (float): Esponente del termine di focalizzazione (1 - pt)^gamma.
            gamma=0 equivale alla CrossEntropy standard. Valore consigliato: 2.0.
        reduction (str): 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha      # Tensor di forma [num_classes] o None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calcola la CE per ogni campione senza riduzione
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")

        # pt = probabilità assegnata alla classe corretta
        pt = torch.exp(-ce_loss)

        # Termine modulante: abbassa la loss per esempi già classificati bene
        focal_term = (1.0 - pt) ** self.gamma

        # Applica i pesi per classe (alpha), se forniti
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device).gather(0, targets)
            focal_term = focal_term * alpha_t

        loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"gamma={self.gamma}, "
            f"alpha={'set' if self.alpha is not None else 'None'}, "
            f"reduction='{self.reduction}')"
        )


def build_criterion(
    use_focal: bool,
    class_weights: Optional[torch.Tensor],
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    Costruisce la funzione di loss in base alle impostazioni dell'utente.

    Args:
        use_focal (bool): Se True, usa FocalLoss; altrimenti CrossEntropyLoss.
        class_weights (Tensor | None): Pesi delle classi calcolati da
            ``Trainer.compute_class_weights``. Usati come *alpha* nella
            FocalLoss e come *weight* nella CrossEntropyLoss.
        gamma (float): Parametro di focalizzazione (ignorato se use_focal=False).
        label_smoothing (float): Applicato solo alla CrossEntropyLoss.

    Returns:
        nn.Module: Criterio pronto per essere passato a ``Trainer.train_model``.
    """
    if use_focal:
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=gamma,
        )
        print(f"[Loss] FocalLoss attiva  (gamma={gamma}, "
              f"class_weights={'sì' if class_weights is not None else 'no'})")
        if label_smoothing > 0.0:
            print(f"[Warning] label_smoothing={label_smoothing} ignorato: "
                  f"non supportato nativamente da FocalLoss.")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        print(f"[Loss] CrossEntropyLoss attiva "
              f"(label_smoothing={label_smoothing}, "
              f"class_weights={'sì' if class_weights is not None else 'no'})")
    return criterion


def _process_memory_mb() -> Optional[float]:
    """Restituisce la RAM usata dal processo, se psutil è disponibile."""
    try:
        import psutil
    except ImportError:
        return None

    return psutil.Process().memory_info().rss / (1024 ** 2)


class ResourceTracker:
    """Misura durata, memoria GPU e RAM di processo nelle fasi principali."""

    def __init__(self, device: torch.device):
        self.device = device
        self.records: List[Dict[str, Any]] = []
        self._phase_starts: Dict[str, float] = {}

    def _cuda_memory(self) -> Dict[str, Optional[float]]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {
                "gpu_allocated_mb": None,
                "gpu_reserved_mb": None,
                "gpu_peak_allocated_mb": None,
                "gpu_peak_reserved_mb": None,
                "gpu_total_mb": None,
            }

        return {
            "gpu_allocated_mb": torch.cuda.memory_allocated(self.device) / (1024 ** 2),
            "gpu_reserved_mb": torch.cuda.memory_reserved(self.device) / (1024 ** 2),
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(self.device) / (1024 ** 2),
            "gpu_peak_reserved_mb": torch.cuda.max_memory_reserved(self.device) / (1024 ** 2),
            "gpu_total_mb": torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2),
        }

    def start(self, phase: str):
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._phase_starts[phase] = time.perf_counter()

    def stop(self, phase: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        started_at = self._phase_starts.pop(phase, None)
        record: Dict[str, Any] = {
            "phase": phase,
            "duration_seconds": None if started_at is None else time.perf_counter() - started_at,
            "device": str(self.device),
            "process_ram_mb": _process_memory_mb(),
        }
        record.update(self._cuda_memory())
        if extra:
            record.update(extra)

        self.records.append(record)
        return record


class AddGaussianNoise:
    """Aggiunge rumore gaussiano a un'immagine PIL."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() >= self.p:
            return img
        img_np = np.array(img, dtype=np.float32) / 255.0
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        noisy = img_np + np.random.normal(self.mean, self.std, img_np.shape)
        noisy = np.clip(noisy, 0.0, 1.0)
        return Image.fromarray((noisy * 255).astype(np.uint8))


class RandomAffineWithReflectPad:
    """
    Alternativa a RandomAffine che usa padding 'reflect' invece di fill=0.

    Applica padding simmetrico prima della trasformazione affine in modo che
    le aree vuote create dalla rotazione/traslazione siano riempite con pixel
    reali riflessi dal bordo, eliminando gli artefatti neri nel CenterCrop finale.

    Strategia:
        1. Padding reflect di `pad` pixel su tutti e 4 i lati.
        2. RandomAffine sulla versione padded.
        3. CenterCrop per tornare alle dimensioni originali (rimuove il padding).
    """

    def __init__(
        self,
        degrees: float = 0,
        translate: Optional[Tuple[float, float]] = None,
        scale: Optional[Tuple[float, float]] = None,
        pad: int = 64,
        p: float = 0.5,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.pad = pad
        self.p = p
        self._pad_fn = transforms.Pad(pad, padding_mode="reflect")
        self._affine = transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            interpolation=interpolation,
            fill=0,
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() >= self.p:
            return img
        w, h = img.size
        padded = self._pad_fn(img)           # (W+2*pad) × (H+2*pad)
        rotated = self._affine(padded)       # applica trasformazione sull'immagine padded
        return transforms.CenterCrop((h, w))(rotated)  # ritaglia alle dimensioni originali

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"degrees={self._affine.degrees}, "
            f"translate={self._affine.translate}, "
            f"scale={self._affine.scale}, "
            f"pad={self.pad}, "
            f"p={self.p})"
        )


class AdaptiveAugmentationDataset(Dataset):
    """Dataset con augmentation adattiva per classe."""
    
    def __init__(
        self,
        subset_data: Subset,
        class_names_map: Dict[int, str],
        augmentation_strategies: Dict[str, Dict],
        resize: int = 256,
        crop: int = 224,
        is_train: bool = True,
        global_aug_p: float = 0.6,
    ):
        self.subset_data = subset_data
        self.class_names_map = class_names_map
        self.is_train = is_train
        self.global_aug_p = global_aug_p

        self.base_transform = self._build_base_transform(resize, crop)
        self.class_transforms = {
            name.lower(): self._build_augmentation_transform(params, resize, crop)
            for name, params in augmentation_strategies.items()
        }
        self.default_aug_transform = self._build_augmentation_transform({}, resize, crop)

    @staticmethod
    def _build_base_transform(resize: int = 256, crop: int = 224) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _build_augmentation_transform(
        params: Dict, resize: int = 256, crop: int = 224
    ) -> transforms.Compose:
        # La pipeline parte dal resize e applica solo le trasformazioni abilitate
        # nella configurazione della classe corrente.
        steps = [
            transforms.Resize(resize),
        ]

        rotation = params.get("rotation_range", 0)
        if rotation > 0:
            p = params.get("rotation_p", 0.5)
            steps.append(RandomAffineWithReflectPad(degrees=rotation, pad=64, p=p))

        shift_x = params.get("width_shift_range", 0)
        shift_y = params.get("height_shift_range", 0)
        if shift_x > 0 or shift_y > 0:
            p = params.get("shift_p", 0.5)
            steps.append(RandomAffineWithReflectPad(degrees=0, translate=(shift_x, shift_y), pad=64, p=p))

        zoom = params.get("zoom_range", 0)
        if zoom > 0:
            p = params.get("zoom_p", 0.5)
            steps.append(RandomAffineWithReflectPad(degrees=0, scale=(1 - zoom, 1 + zoom), pad=64, p=p))

        if params.get("horizontal_flip", False):
            p = params.get("horizontal_flip_p", 0.5)
            steps.append(transforms.RandomHorizontalFlip(p=p))

        brightness = params.get("brightness_range")
        if brightness is not None:
            p = params.get("brightness_p", 0.5)
            steps.append(transforms.RandomApply(
                [transforms.ColorJitter(brightness=tuple(brightness))], p=p
            ))

        if params.get("add_noise", False):
            p = params.get("add_noise_p", 0.5)
            noise_std = params.get("add_noise_std", 0.1)
            steps.append(AddGaussianNoise(std=noise_std, p=p))

        # Il crop finale uniforma tutte le immagini alla dimensione richiesta
        # dal modello e rimuove eventuali bordi introdotti dalle trasformazioni.
        steps += [
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        return transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.subset_data)

    def __getitem__(self, idx: int):
        img, label = self.subset_data[idx]

        if not self.is_train:
            return self.base_transform(img), label

        if torch.rand(1).item() >= self.global_aug_p:
            return self.base_transform(img), label

        # Ogni classe può avere una strategia diversa; se manca, usa quella base.
        class_name = self.class_names_map[label].lower()
        aug_transform = self.class_transforms.get(class_name, self.default_aug_transform)
        return aug_transform(img), label

class ModelFactory:
    """Factory per creare modelli di classificazione."""
    
    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int,
        device: torch.device,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> nn.Module:
        """Crea un modello specificato."""
        
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            # In feature extraction si congela il backbone e si addestra solo
            # il classificatore finale adattato al numero di classi del dataset.
            if pretrained:
                for p in model.parameters():
                    p.requires_grad = False
            num_ftrs = model.classifier[1].in_features
            model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b2(weights=weights)
            # Stessa strategia di EfficientNet-B0: backbone congelato,
            # classificatore sostituito.
            if pretrained:
                for p in model.parameters():
                    p.requires_grad = False
            num_ftrs = model.classifier[1].in_features
            model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b3(weights=weights)
            # B3 usa lo stesso layout del classificatore delle altre EfficientNet.
            if pretrained:
                for p in model.parameters():
                    p.requires_grad = False
            num_ftrs = model.classifier[1].in_features
            model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            # MobileNet ha un classificatore con indici diversi rispetto a
            # EfficientNet, quindi si sostituisce l'ultimo layer lineare.
            if pretrained:
                for p in model.parameters():
                    p.requires_grad = False
            in_feats = model.classifier[3].in_features
            model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
            model.classifier[3] = nn.Linear(in_feats, num_classes)
            
        else:
            raise ValueError(f"Modello non supportato: {model_name}")
        
        return model.to(device)
    
    @staticmethod
    def unfreeze_for_finetuning(
        model: nn.Module,
        model_name: str,
        blocks: Optional[List[int]] = None,
        n_last_blocks: int = 3,
    ):
        """Sblocca blocchi per il fine-tuning.
        
        Args:
            model: Il modello PyTorch.
            model_name: Nome del modello ('efficientnet_b0', 'mobilenet_v3_small', ecc.).
            blocks: Lista di indici dei blocchi da sbloccare (solo EfficientNet).
                    Default: [5, 6, 7] se non specificato.
            n_last_blocks: Numero di ultimi blocchi da sbloccare (solo MobileNet).
        """
        if "efficientnet" in model_name:
            # Nel fine-tuning si riaprono solo gli ultimi blocchi scelti, evitando
            # di aggiornare tutto il backbone con un dataset piccolo.
            if blocks is None:
                blocks = [5, 6, 7]
            for block_idx in blocks:
                for p in model.features[block_idx].parameters():
                    p.requires_grad = True
        elif "mobilenet" in model_name:
            # Per MobileNet la configurazione indica quanti blocchi finali
            # rendere addestrabili.
            blocchi = list(model.features)
            for blk in blocchi[-n_last_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

        # Il classificatore resta sempre addestrabile in entrambe le fasi.
        for p in model.classifier.parameters():
            p.requires_grad = True

class Trainer:
    """Gestisce il training dei modelli."""
    
    def __init__(self, config: Dict, experiment_dir: Path):
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {
            "train_loss": [], "train_bal_acc": [],
            "val_loss": [], "val_bal_acc": []
        }
        self.resource_tracker = ResourceTracker(self.device)
        self.stop_requested = False
        # Lo scaler viene creato una sola volta e usato solo su CUDA.
        self._scaler: Optional[torch.amp.GradScaler] = (
            torch.amp.GradScaler(self.device.type)
            if self.device.type == "cuda"
            else None
        )
        
    def compute_class_weights(self, train_indices: np.ndarray, targets: np.ndarray, n_classes: int) -> torch.Tensor:
        """Calcola i pesi delle classi per bilanciamento."""
        # Le classi meno frequenti ricevono peso maggiore nella CrossEntropy.
        counts = np.bincount(targets[train_indices], minlength=n_classes)
        total = len(train_indices)
        weights = torch.tensor(
            [total / (n_classes * max(counts[i], 1)) for i in range(n_classes)],
            dtype=torch.float32,
        )
        return weights.to(self.device)
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loader: DataLoader,
        criterion: nn.Module,
        use_amp: bool = True,
    ) -> Tuple[float, float]:
        """Addestra per un'epoca."""
        model.train()
        running_loss = 0.0
        total_samples = 0
        all_preds: List[int] = []
        all_labels: List[int] = []

        scaler = self._scaler if use_amp else None
        device_type = self.device.type

        for inputs, labels in tqdm(loader, desc="  Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            # Con AMP su GPU si riduce l'uso di memoria mantenendo stabile il
            # backward tramite GradScaler; su CPU si usa il percorso standard.
            if scaler is not None:
                with torch.amp.autocast(device_type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # La balanced accuracy evita che classi molto presenti dominino la metrica.
        train_loss = running_loss / total_samples
        train_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        return train_loss, train_bal_acc
    
    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Valuta il modello."""
        model.eval()
        running_loss = 0.0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for inputs, labels in tqdm(loader, desc="  Valutazione", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / total_samples
        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
        return val_loss, val_bal_acc
    
    def train_model(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        epochs: int = 30,
        patience: int = 5,
        min_delta: float = 0.001,
        save_path: str = "best_model.pth",
        initial_best: float = 0.0,
        scheduler: Optional[Any] = None,
        use_amp: bool = True,
        phase_name: Optional[str] = None,
    ) -> Dict:
        """Training completo con early stopping."""

        # Resetta il flag prima di ogni fase: un'interruzione manuale durante
        # la FE non deve bloccare automaticamente la FT o i fold successivi.
        self.stop_requested = False

        best_val_bal_acc = initial_best
        epochs_no_improve = 0
        best_weights = copy.deepcopy(model.state_dict())

        history: Dict[str, List[float]] = {
            "train_loss": [], "train_bal_acc": [],
            "val_loss": [], "val_bal_acc": []
        }

        phase = phase_name or "training"
        self.resource_tracker.start(phase)
        epochs_run = 0
        for epoch in range(1, epochs + 1):
            if self.stop_requested:
                print("\n[!] Training interrotto manualmente dall'utente.")
                break
            epochs_run = epoch
            # Ogni epoca alterna addestramento e validazione, poi aggiorna
            # metriche, scheduler ed eventuale checkpoint migliore.
            train_loss, train_bal_acc = self.train_epoch(
                model, optimizer, train_loader, criterion, use_amp
            )
            val_loss, val_bal_acc = self.evaluate(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["train_bal_acc"].append(train_bal_acc)
            history["val_loss"].append(val_loss)
            history["val_bal_acc"].append(val_bal_acc)
            for k in history:
                self.history[k].append(history[k][-1])

            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:02d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}  Bal-Acc: {train_bal_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}  Bal-Acc: {val_bal_acc:.4f} | "
                  f"lr: {lr_now:.2e}")

            if scheduler is not None:
                # ReduceLROnPlateau richiede la metrica di validazione; gli altri
                # scheduler avanzano solo in base all'epoca.
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_bal_acc)
                else:
                    scheduler.step()

            if val_bal_acc > best_val_bal_acc + min_delta:
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f"  -> Nuovo miglior modello salvato! (Bal-Acc: {val_bal_acc:.4f})")
                epochs_no_improve = 0
                best_val_bal_acc = val_bal_acc
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping dopo {epoch} epoche.")
                    break

        # A fine training il modello torna ai pesi migliori osservati in validazione.
        model.load_state_dict(best_weights)

        # Garantisce che il file checkpoint esista sempre su disco, anche se questa
        # fase non ha mai superato initial_best (es. FT che non batte la FE).
        # Senza questa riga, qualsiasi codice che carica save_path (Grad-CAM,
        # predict, ecc.) andrebbe in crash con FileNotFoundError.
        torch.save(best_weights, save_path)

        self.resource_tracker.stop(phase, {
            "epochs_run": epochs_run,
            "best_val_bal_acc": best_val_bal_acc,
            "batch_size": getattr(train_loader, "batch_size", None),
            "use_amp": use_amp,
        })
        return history

    @torch.no_grad()
    def benchmark_inference(
        self,
        model: nn.Module,
        loader: DataLoader,
        phase_name: str = "inference_benchmark",
        num_runs: int = 50,
        warmup_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Misura throughput puro della GPU/CPU (escludendo il data loading).
        Usa un singolo batch tenuto in memoria per misurare i veri FPS hardware.
        """
        model.eval()
        
        # Prende un singolo batch reale per avere forma e tipo esatti
        try:
            inputs, _ = next(iter(loader))
        except StopIteration:
            return {}
            
        inputs = inputs.to(self.device)
        batch_size = inputs.size(0)

        # Warmup: forza l'inizializzazione di cuDNN e l'autotune di PyTorch
        # così da non inquinare i tempi misurati successivamente.
        for _ in range(warmup_runs):
            _ = model(inputs)
            
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        self.resource_tracker.start(phase_name)

        # Misurazione pura
        for _ in range(num_runs):
            _ = model(inputs)
            
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        total_samples = batch_size * num_runs
        record = self.resource_tracker.stop(phase_name, {
            "samples": total_samples,
            "batches": num_runs,
            "batch_size": batch_size,
        })
        duration = record.get("duration_seconds") or 0.0
        record["samples_per_second"] = total_samples / duration if duration > 0 else None
        
        return record

class ExperimentManager:
    """Gestisce gli esperimenti e il salvataggio dei risultati."""
    
    def __init__(self, base_dir: str = "experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_experiment_dir(self, model_name: str, custom_name: Optional[str] = None) -> Path:
        """Crea una directory per l'esperimento con timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name:
            exp_name = f"{model_name}_{custom_name}_{timestamp}"
        else:
            exp_name = f"{model_name}_{timestamp}"
        
        exp_dir = self.base_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Ogni esperimento separa pesi, grafici e log per rendere i risultati
        # confrontabili e riproducibili.
        (exp_dir / "models").mkdir(exist_ok=True)
        (exp_dir / "plots").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        
        return exp_dir
    
    def save_config(self, config: Dict, exp_dir: Path):
        """Salva la configurazione."""
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def save_history(self, history: Dict, exp_dir: Path, filename: str = "history.json"):
        """Salva la cronologia del training."""
        history_path = exp_dir / "logs" / filename
        # Converte eventuali tipi numpy in float Python, serializzabili in JSON.
        serializable_history = {}
        for key, values in history.items():
            serializable_history[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)

    def save_resource_usage(self, records: List[Dict[str, Any]], exp_dir: Path):
        """Salva durata, memoria e throughput misurati durante l'esperimento."""
        resource_path = exp_dir / "logs" / "resource_usage.json"
        with open(resource_path, 'w') as f:
            json.dump(records, f, indent=2)
    
    def save_training_curves(self, history: Dict, exp_dir: Path, model_name: str):
        """Salva i grafici delle curve di training."""
        plots_dir = exp_dir / "plots"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # L'asse X parte da 1 (epoca 1, 2, 3, …) anziché da 0.
        epochs_range = range(1, len(history["train_loss"]) + 1)

        # Il primo grafico confronta la loss di training e validazione.
        axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#3B82F6")
        axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#EF4444")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoca")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Il secondo grafico mostra la metrica principale scelta per il dataset sbilanciato.
        axes[1].plot(epochs_range, history["train_bal_acc"], label="Train Bal-Acc", color="#3B82F6")
        axes[1].plot(epochs_range, history["val_bal_acc"], label="Val Bal-Acc", color="#EF4444")
        axes[1].set_title("Balanced Accuracy")
        axes[1].set_xlabel("Epoca")
        axes[1].set_ylabel("Balanced Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Training Curves - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: List[str], exp_dir: Path, model_name: str):
        """Salva la matrice di confusione."""
        plots_dir = exp_dir / "plots"
        
        cm = confusion_matrix(y_true, y_pred)
        # La matrice normalizzata evidenzia gli errori percentuali per classe.
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        im1 = axes[0].imshow(cm_norm, interpolation='nearest', cmap='Blues')
        axes[0].set_title("Confusion Matrix (Normalizzata)")
        plt.colorbar(im1, ax=axes[0])
        axes[0].set_xticks(range(len(class_names)))
        axes[0].set_yticks(range(len(class_names)))
        axes[0].set_xticklabels(class_names, rotation=45, ha='right')
        axes[0].set_yticklabels(class_names)
        
        im2 = axes[1].imshow(cm, interpolation='nearest', cmap='Oranges')
        axes[1].set_title("Confusion Matrix (Assoluta)")
        plt.colorbar(im2, ax=axes[1])
        axes[1].set_xticks(range(len(class_names)))
        axes[1].set_yticks(range(len(class_names)))
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].set_yticklabels(class_names)
        
        plt.suptitle(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / f"{model_name}_confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: List[str], exp_dir: Path):
        """Salva il report di classificazione."""
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        report_path = exp_dir / "logs" / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        return report

def extract_dataset(zip_path: str, extract_to: str, dataset_name: str) -> Path:
    """Estrae il dataset da un file zip se non già presente."""
    dataset_path = Path(extract_to) / dataset_name
    
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"Dataset già presente in {dataset_path}")
        return dataset_path
    
    print(f"Estrazione dataset da {zip_path}...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Dataset estratto in {dataset_path}")
    return dataset_path


def analyze_dataset(dataset_path: Path) -> Dict:
    """Analizza la struttura del dataset."""
    class_names = sorted([
        d.name for d in dataset_path.iterdir() if d.is_dir()
    ])
    
    stats = {}
    total = 0
    
    for class_name in class_names:
        class_path = dataset_path / class_name
        count = len([
            f for f in class_path.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
        ])
        stats[class_name] = count
        total += count
    
    return {
        "class_names": class_names,
        "class_counts": stats,
        "total_images": total,
        "num_classes": len(class_names),
    }


def get_default_augmentation_strategies() -> Dict[str, Dict]:
    """Restituisce le strategie di augmentation predefinite per classe."""
    return {
        "plastic": {
            "rotation_range": 20, "rotation_p": 0.7,
            "horizontal_flip": True, "horizontal_flip_p": 0.5,
            "zoom_range": 0.15, "zoom_p": 0.7,
            "width_shift_range": 0.1, "height_shift_range": 0.1, "shift_p": 0.7,
            "brightness_range": [0.3, 0.9], "brightness_p": 0.6,
            "add_noise": False, "add_noise_std": 0.1, "add_noise_p": 0.0,
        },
        "organic": {
            "rotation_range": 20, "rotation_p": 0.7,
            "zoom_range": 0.2, "zoom_p": 0.8,
            "width_shift_range": 0.1, "height_shift_range": 0.1, "shift_p": 0.7,
            "brightness_range": [0.5, 2.0], "brightness_p": 0.8,
            "add_noise": True, "add_noise_std": 0.1, "add_noise_p": 0.6,
        },
        "metal": {
            "rotation_range": 15, "rotation_p": 0.6,
            "zoom_range": 0.1, "zoom_p": 0.6,
            "width_shift_range": 0.05, "height_shift_range": 0.05, "shift_p": 0.6,
            "brightness_range": [0.7, 1.2], "brightness_p": 0.5,
            "add_noise": False, "add_noise_std": 0.1, "add_noise_p": 0.0,
        },
        "battery": {
            "rotation_range": 10, "rotation_p": 0.5,
            "zoom_range": 0.1, "zoom_p": 0.5,
            "brightness_range": [0.5, 1.5], "brightness_p": 0.4,
            "add_noise": False, "add_noise_std": 0.1, "add_noise_p": 0.0,
        },
        "undifferentiated": {
            "rotation_range": 25, "rotation_p": 0.8,
            "horizontal_flip": True, "horizontal_flip_p": 0.6,
            "zoom_range": 0.2, "zoom_p": 0.8,
            "width_shift_range": 0.15, "height_shift_range": 0.15, "shift_p": 0.8,
            "brightness_range": [0.6, 1.3], "brightness_p": 0.7,
            "add_noise": True, "add_noise_std": 0.1, "add_noise_p": 0.7,
        },
        "clothing": {
            "rotation_range": 15, "rotation_p": 0.5,
            "horizontal_flip": True, "horizontal_flip_p": 0.5,
            "zoom_range": 0.1, "zoom_p": 0.5,
            "brightness_range": [0.8, 1.2], "brightness_p": 0.4,
            "add_noise": False, "add_noise_std": 0.1, "add_noise_p": 0.0,
        },
        "papery": {
            "rotation_range": 20, "rotation_p": 0.6,
            "horizontal_flip": True, "horizontal_flip_p": 0.5,
            "zoom_range": 0.15, "zoom_p": 0.6,
            "width_shift_range": 0.1, "height_shift_range": 0.1, "shift_p": 0.5,
            "add_noise": False, "add_noise_std": 0.1, "add_noise_p": 0.0,
        },
        "glass": {
            "rotation_range": 15, "rotation_p": 0.5,
            "zoom_range": 0.1, "zoom_p": 0.5,
            "brightness_range": [0.4, 1.7], "brightness_p": 0.7,
            "add_noise": False, "add_noise_std": 0.1, "add_noise_p": 0.0,
        },
    }

def get_advanced_stratification_labels(dataset_path: str, dataset_samples: List[Tuple[str, int]]) -> List[str]:
    """
    Genera le etichette combinate 'MacroClass (SubClass)' per la stratificazione.
    """
    labels = []
    root = Path(dataset_path)
    for path_str, _ in dataset_samples:
        p = Path(path_str)
        rel_path = p.relative_to(root)
        parts = rel_path.parts
        
        if len(parts) < 2:
            raise ValueError(
                f"Struttura del dataset non valida: il file '{path_str}' si trova "
                f"direttamente nella root anziché in una sottocartella di classe."
            )
            
        macro_class = parts[0]
        if len(parts) > 2:
            sub_class = "/".join(parts[1:-1])
            labels.append(f"{macro_class} ({sub_class})")
        else:
            labels.append(macro_class)
    return labels

def advanced_stratified_split(
    dataset_path: str, 
    dataset_samples: List[Tuple[str, int]],
    split_type: str = "static", 
    train_ratio: float = 0.7, 
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[int], List[int], Optional[List[int]], List[str]]:
    """
    Esegue lo split preservando la distribuzione delle sottoclassi.
    Ritorna: train_indices, test_indices, val_indices (se split_type=='static' altrimenti None), stratification_labels
    """
    labels = get_advanced_stratification_labels(dataset_path, dataset_samples)
    
    # Check for labels with less than 2 samples, as train_test_split will crash
    label_counts = Counter(labels)
    rare_labels = [label for label, count in label_counts.items() if count < 2]
    if rare_labels:
        raise ValueError(
            f"Errore nello split stratificato: Le seguenti sottoclassi hanno meno di 2 campioni "
            f"e non possono essere suddivise in modo sicuro: {rare_labels}. "
            "Aggiungi più immagini a queste categorie o rimuovile dal dataset."
        )

    indices = list(range(len(dataset_samples)))
    
    if split_type == "static":
        test_size = 1.0 - train_ratio
        # Primo split: Train vs Temp (Val+Test)
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            indices, labels, 
            test_size=test_size, 
            random_state=random_seed, 
            stratify=labels
        )
        # Secondo split: divide Temp a metà per avere Val e Test
        try:
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx, y_temp, 
                test_size=0.5, 
                random_state=random_seed, 
                stratify=y_temp
            )
        except ValueError as e:
            print(f"[Warning] Stratificazione annidata fallita (sottoclassi troppo piccole). Fallback su split casuale per Val/Test.")
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx, y_temp, 
                test_size=0.5, 
                random_state=random_seed, 
                stratify=None
            )
            
        return train_idx, test_idx, val_idx, labels
    else:
        # K-Fold prepara: divisione solo in Train e Test
        test_size = test_ratio
        train_idx, test_idx, _, _ = train_test_split(
            indices, labels, 
            test_size=test_size, 
            random_state=random_seed, 
            stratify=labels
        )
        return train_idx, test_idx, None, labels

def analyze_dataset_with_rich(
    stratification_labels: List[str], 
    train_idx: List[int], 
    test_idx: List[int], 
    val_idx: Optional[List[int]] = None
):
    """
    Stampa l'analisi dettagliata del dataset usando rich, stile vecchio progetto.
    """
    # Force a larger width and ANSI terminal output to prevent Jupyter HTML clipping
    console = Console(force_terminal=True, width=200)
    
    y_train = [stratification_labels[i] for i in train_idx]
    y_test = [stratification_labels[i] for i in test_idx]
    
    conteggio_train = Counter(y_train)
    conteggio_test = Counter(y_test)
    
    if val_idx is not None:
        y_val = [stratification_labels[i] for i in val_idx]
        conteggio_val = Counter(y_val)
        tot_val = len(val_idx)
    else:
        conteggio_val = Counter()
        tot_val = 0
        
    tot_train = len(train_idx)
    tot_test = len(test_idx)
    tot_completo = tot_train + tot_test + tot_val
    
    table = Table(
        title="Analisi Stratificazione del Dataset",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        title_style="bold magenta"
    )
    table.add_column("Categoria / Sottocategoria", style="magenta", no_wrap=True)
    table.add_column("Train", justify="right", style="green", header_style="bold green", no_wrap=True)
    if val_idx is not None:
        table.add_column("Validation", justify="right", style="yellow", header_style="bold yellow", no_wrap=True)
    table.add_column("Test", justify="right", style="cyan", header_style="bold cyan", no_wrap=True)
    
    macro_corrente = None
    
    for cat in sorted(set(stratification_labels)):
        macro = cat.split(" (")[0]
        
        if macro != macro_corrente:
            if macro_corrente is not None:
                table.add_section()
            macro_corrente = macro
            
        qta_train = conteggio_train[cat]
        qta_test = conteggio_test[cat]
        qta_val = conteggio_val[cat] if val_idx is not None else 0
        
        totale_cat = qta_train + qta_test + qta_val
        
        def fmt(qta, tot_cat, tot_split):
            if tot_split == 0: return "0 (0%)"
            pct_loc = (qta / tot_cat * 100) if tot_cat > 0 else 0
            pct_glob = (qta / tot_split * 100) if tot_split > 0 else 0
            return f"{qta:,}  ({pct_loc:.0f}% | {pct_glob:.1f}%)"
            
        label = f"  ↳ {cat.split('(')[1].rstrip(')')}" if "(" in cat else cat
        
        if val_idx is not None:
            table.add_row(label, fmt(qta_train, totale_cat, tot_train),
                          fmt(qta_val, totale_cat, tot_val),
                          fmt(qta_test, totale_cat, tot_test))
        else:
            table.add_row(label, fmt(qta_train, totale_cat, tot_train),
                          fmt(qta_test, totale_cat, tot_test))
                          
    table.add_section()
    
    def fmt_tot(n, tot):
        if tot == 0: return "0 (0%)"
        return f"{n:,}  ({n/tot*100:.1f}%)"
        
    if val_idx is not None:
        table.add_row("TOTALE", fmt_tot(tot_train, tot_completo),
                      fmt_tot(tot_val, tot_completo),
                      fmt_tot(tot_test, tot_completo),
                      style="bold magenta")
    else:
        table.add_row("TOTALE", fmt_tot(tot_train, tot_completo),
                      fmt_tot(tot_test, tot_completo),
                      style="bold magenta")
                      
    console.print(table)

def print_dataset_structure_with_rich(dataset_path: str):
    """
    Stampa la struttura iniziale del dataset prima dello split (Table 1 del vecchio progetto).
    """
    # Force a larger width and ANSI terminal output to prevent Jupyter HTML clipping
    console = Console(force_terminal=True, width=200)
    
    if not os.path.exists(dataset_path):
        console.print(f"[red]Errore: Il percorso {dataset_path} non esiste.[/red]")
        return
        
    etichette = sorted(
        nome for nome in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, nome))
    )
    
    righe = []
    totale_complessivo = 0
    # Add all standard PyTorch extensions
    estensioni_immagini = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    
    for etichetta in etichette:
        percorso_macro = os.path.join(dataset_path, etichetta)
        totale_macro = 0
        sottocartelle = []
        ha_sottocartelle_effettive = False
        
        for dirpath, _, files in os.walk(percorso_macro):
            n = sum(1 for f in files if f.lower().endswith(estensioni_immagini))
            if n > 0:
                totale_macro += n
                percorso_relativo = os.path.relpath(dirpath, percorso_macro)
                if percorso_relativo != ".":
                    ha_sottocartelle_effettive = True
                    sottocartelle.append((percorso_relativo, n))
                    
        righe.append((etichetta, totale_macro, sottocartelle if ha_sottocartelle_effettive else []))
        totale_complessivo += totale_macro
        
    table = Table(
        title="Struttura Dataset",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        title_style="bold magenta"
    )
    table.add_column("Categoria", style="magenta", no_wrap=True)
    table.add_column("N. immagini", justify="right", style="green", header_style="bold green", no_wrap=True)
    table.add_column("% totale", justify="right", style="cyan", header_style="bold cyan", no_wrap=True)
    
    for etichetta, conteggio, sottocartelle in righe:
        pct = f"{conteggio / totale_complessivo * 100:.1f}%" if totale_complessivo > 0 else "0%"
        table.add_row(etichetta, f"{conteggio:,}", pct, style="bold")
        for nome_sub, cnt_sub in sottocartelle:
            pct_sub = f"{cnt_sub / totale_complessivo * 100:.1f}%" if totale_complessivo > 0 else "0%"
            table.add_row(f"  ↳ {nome_sub}", f"{cnt_sub:,}", pct_sub)
        table.add_section()
        
    table.add_row("TOTALE", f"{totale_complessivo:,}", "100.0%", style="bold magenta")
    console.print(table)

