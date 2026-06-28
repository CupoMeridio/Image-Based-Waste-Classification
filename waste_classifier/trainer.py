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
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import copy

__version__ = "1.1.0"


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
            steps.append(transforms.RandomApply(
                [transforms.RandomAffine(degrees=rotation)], p=p
            ))

        shift_x = params.get("width_shift_range", 0)
        shift_y = params.get("height_shift_range", 0)
        if shift_x > 0 or shift_y > 0:
            p = params.get("shift_p", 0.5)
            steps.append(transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, translate=(shift_x, shift_y))], p=p
            ))

        zoom = params.get("zoom_range", 0)
        if zoom > 0:
            p = params.get("zoom_p", 0.5)
            steps.append(transforms.RandomApply(
                [transforms.RandomAffine(degrees=0, scale=(1 - zoom, 1 + zoom))], p=p
            ))

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
            steps.append(AddGaussianNoise(p=p))

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
            for p in model.parameters():
                p.requires_grad = False
            num_ftrs = model.classifier[1].in_features
            model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b3(weights=weights)
            # B3 usa lo stesso layout del classificatore delle altre EfficientNet.
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
        label_smoothing: float = 0.0,
        phase_name: Optional[str] = None,
    ) -> Dict:
        """Training completo con early stopping.
        
        Args:
            label_smoothing: Se > 0, sovrascrive il criterion con uno identico
                             ma con label_smoothing attivo. Ignorato se il
                             criterion passato non è CrossEntropyLoss.
        """
        if label_smoothing > 0.0 and isinstance(criterion, nn.CrossEntropyLoss):
            # Il label smoothing rende il modello meno sicuro su una singola
            # classe e può aiutare la generalizzazione.
            criterion = nn.CrossEntropyLoss(
                weight=criterion.weight,
                label_smoothing=label_smoothing,
            )

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

            if val_bal_acc > best_val_bal_acc:
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                print(f"  -> Nuovo miglior modello salvato! (Bal-Acc: {val_bal_acc:.4f})")

            # L'early stopping conta le epoche senza miglioramenti sufficienti.
            if val_bal_acc > best_val_bal_acc + min_delta:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping dopo {epoch} epoche.")
                    break

            best_val_bal_acc = max(best_val_bal_acc, val_bal_acc)

        # A fine training il modello torna ai pesi migliori osservati in validazione.
        model.load_state_dict(best_weights)
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
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Misura throughput e memoria durante inferenza su un loader."""
        model.eval()
        self.resource_tracker.start(phase_name)

        total_samples = 0
        total_batches = 0
        for batch_idx, (inputs, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(self.device)
            _ = model(inputs)
            total_samples += inputs.size(0)
            total_batches += 1

        record = self.resource_tracker.stop(phase_name, {
            "samples": total_samples,
            "batches": total_batches,
            "batch_size": getattr(loader, "batch_size", None),
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
    
    def save_model_weights(self, model: nn.Module, exp_dir: Path, filename: str = "model.pth"):
        """Salva i pesi del modello."""
        weights_path = exp_dir / "models" / filename
        torch.save(model.state_dict(), weights_path)
    
    def save_training_curves(self, history: Dict, exp_dir: Path, model_name: str):
        """Salva i grafici delle curve di training."""
        plots_dir = exp_dir / "plots"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Il primo grafico confronta la loss di training e validazione.
        axes[0].plot(history["train_loss"], label="Train Loss", color="#3B82F6")
        axes[0].plot(history["val_loss"], label="Val Loss", color="#EF4444")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoca")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Il secondo grafico mostra la metrica principale scelta per il dataset sbilanciato.
        axes[1].plot(history["train_bal_acc"], label="Train Bal-Acc", color="#3B82F6")
        axes[1].plot(history["val_bal_acc"], label="Val Bal-Acc", color="#EF4444")
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
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
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
            "add_noise": False, "add_noise_p": 0.0,
        },
        "organic": {
            "rotation_range": 20, "rotation_p": 0.7,
            "zoom_range": 0.2, "zoom_p": 0.8,
            "width_shift_range": 0.1, "height_shift_range": 0.1, "shift_p": 0.7,
            "brightness_range": [0.5, 2.0], "brightness_p": 0.8,
            "add_noise": True, "add_noise_p": 0.6,
        },
        "metal": {
            "rotation_range": 15, "rotation_p": 0.6,
            "zoom_range": 0.1, "zoom_p": 0.6,
            "width_shift_range": 0.05, "height_shift_range": 0.05, "shift_p": 0.6,
            "brightness_range": [0.7, 1.2], "brightness_p": 0.5,
            "add_noise": False, "add_noise_p": 0.0,
        },
        "battery": {
            "rotation_range": 10, "rotation_p": 0.5,
            "zoom_range": 0.1, "zoom_p": 0.5,
            "brightness_range": [0.5, 1.5], "brightness_p": 0.4,
            "add_noise": False, "add_noise_p": 0.0,
        },
        "undifferentiated": {
            "rotation_range": 25, "rotation_p": 0.8,
            "horizontal_flip": True, "horizontal_flip_p": 0.6,
            "zoom_range": 0.2, "zoom_p": 0.8,
            "width_shift_range": 0.15, "height_shift_range": 0.15, "shift_p": 0.8,
            "brightness_range": [0.6, 1.3], "brightness_p": 0.7,
            "add_noise": True, "add_noise_p": 0.7,
        },
        "clothing": {
            "rotation_range": 15, "rotation_p": 0.5,
            "horizontal_flip": True, "horizontal_flip_p": 0.5,
            "zoom_range": 0.1, "zoom_p": 0.5,
            "brightness_range": [0.8, 1.2], "brightness_p": 0.4,
            "add_noise": False, "add_noise_p": 0.0,
        },
        "papery": {
            "rotation_range": 20, "rotation_p": 0.6,
            "horizontal_flip": True, "horizontal_flip_p": 0.5,
            "zoom_range": 0.15, "zoom_p": 0.6,
            "width_shift_range": 0.1, "height_shift_range": 0.1, "shift_p": 0.5,
            "add_noise": False, "add_noise_p": 0.0,
        },
        "glass": {
            "rotation_range": 15, "rotation_p": 0.5,
            "zoom_range": 0.1, "zoom_p": 0.5,
            "brightness_range": [0.4, 1.7], "brightness_p": 0.7,
            "add_noise": False, "add_noise_p": 0.0,
        },
    }
