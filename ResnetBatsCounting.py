"""
BatCount-ResNet: Stratified 5-fold CV with internal validation for checkpoint selection
- Pretrained ResNet18/34/50/101 (ImageNet)
- Transfer learning (fine-tune all layers)
- Inner validation split (10% of dev) for early stopping + LR scheduling + checkpoint
- Test fold used ONLY once per fold for reporting
- Saves per-fold artifacts (history, plots, per-class metrics, best checkpoint)
- Saves per-model summaries + aggregate summaries across models
- Benchmarks throughput/latency (end-to-end and model-only) for batch sizes 32 and 1

"""

import os
import json
import time
import random
from datetime import datetime
from statistics import median

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.models as tv_models
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

# ============================================================
# Hyperparameters & Config
# ============================================================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 12
KFOLDS = 5
EARLY_STOP_PATIENCE = 7

# If you keep this, add to paper; otherwise set to 0.0
LABEL_SMOOTHING = 0.1

# Inner validation fraction from the dev set (outer_train)
INNER_VAL_FRACTION = 0.10

DATA_PATH = "Data/Final Testing Dataset/"
SAVE_DIR = "SavedModels/New"

# DataLoader settings
NUM_WORKERS = max(4, (os.cpu_count() or 4))
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()
PERSISTENT_WORKERS = False  # can set True if stable workers + prefetch_factor is not None
PREFETCH_FACTOR = None      # set int like 2 if you enable persistent_workers=True

SEED = 42

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For speed; if you need stricter determinism, set to False and use deterministic algos
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

# ============================================================
# Matplotlib publication style
# ============================================================
def set_publication_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Run output directories
# ============================================================
RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_ROOT = os.path.join(SAVE_DIR, RUN_ID)
os.makedirs(RUN_ROOT, exist_ok=True)

def model_dir(model_name: str) -> str:
    return os.path.join(RUN_ROOT, model_name)

def fold_dir(model_name: str, fold: int) -> str:
    return os.path.join(model_dir(model_name), f"fold{fold:02d}")

def aggregate_dir() -> str:
    return os.path.join(RUN_ROOT, "_aggregate")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def remove_empty_dirs(root: str) -> None:
    # bottom-up, and check the real filesystem state (not os.walk's cached dirnames)
    for dirpath, _, _ in os.walk(root, topdown=False):
        try:
            if not os.listdir(dirpath):   # <-- THIS is the key change
                os.rmdir(dirpath)
        except OSError:
            pass

# ============================================================
# Transforms (train vs eval)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(15, scale=(0.8, 1.2), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# Datasets (mirror datasets for consistent ordering)
# ============================================================
base_dataset = datasets.ImageFolder(root=DATA_PATH, transform=None)  # for indexing + labels
CLASS_NAMES = base_dataset.classes
DISPLAY_NAMES = CLASS_NAMES

train_dataset_full = datasets.ImageFolder(root=DATA_PATH, transform=train_transform)
eval_dataset_full  = datasets.ImageFolder(root=DATA_PATH, transform=eval_transform)

labels = np.array(base_dataset.targets)
kf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

# ============================================================
# Torchvision weights helper (robust across versions)
# ============================================================
def get_imagenet_weights_enum(model_name: str):
    try:
        mapping = {
            "resnet18":  tv_models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34":  tv_models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50":  tv_models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": tv_models.ResNet101_Weights.IMAGENET1K_V1,
        }
        return mapping[model_name]
    except Exception:
        # Older torchvision may accept str enums
        return "IMAGENET1K_V1"

# ============================================================
# Model wrapper
# ============================================================
class CustomResNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int = NUM_CLASSES, dropout_p: float = 0.3):
        super().__init__()
        weights = get_imagenet_weights_enum(model_name)
        base_fn = getattr(models, model_name)
        self.model = base_fn(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

resnet_models = {
    "ResNet18":  lambda: CustomResNet("resnet18"),
    "ResNet34":  lambda: CustomResNet("resnet34"),
    "ResNet50":  lambda: CustomResNet("resnet50"),
    "ResNet101": lambda: CustomResNet("resnet101"),
}

# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model: nn.Module, loader: DataLoader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(targets.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            y_prob.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = {}
    for i in range(NUM_CLASSES):
        key = str(i)
        if key in report:
            per_class[i] = {
                "precision": report[key]["precision"],
                "recall": report[key]["recall"],
                "f1": report[key]["f1-score"],
            }
        else:
            per_class[i] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return avg_loss, acc, per_class, np.array(y_true), np.array(y_pred), np.array(y_prob)

# ============================================================
# Plotting helpers
# ============================================================
def _savefig(out_base: str) -> None:
    plt.savefig(out_base + ".png")
    plt.savefig(out_base + ".pdf")
    plt.close()

def plot_learning_curve(train_losses, val_losses, model_name, fold, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} — Fold {fold} Learning Curve")
    plt.legend(frameon=False)
    plt.tight_layout()
    _savefig(os.path.join(out_dir, "learning_curve"))

def plot_confusion_matrix(y_true, y_pred, model_name, fold, out_dir):
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DISPLAY_NAMES)
    disp.plot(cmap="Blues", colorbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Fold {fold} Confusion Matrix")
    plt.tight_layout()
    _savefig(os.path.join(out_dir, "confusion_matrix"))

def plot_roc_curve(y_true, y_prob, model_name, fold, out_dir):
    plt.figure(figsize=(6, 5))
    plotted = False
    for i in range(NUM_CLASSES):
        if (y_true == i).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{DISPLAY_NAMES[i]} (AUC={roc_auc:.3f})")
        plotted = True

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} — Fold {fold} ROC")
    if plotted:
        plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    _savefig(os.path.join(out_dir, "roc"))

def plot_precision_recall_curve(y_true, y_prob, model_name, fold, out_dir):
    plt.figure(figsize=(6, 5))
    plotted = False
    for i in range(NUM_CLASSES):
        if (y_true == i).sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
        plt.plot(recall, precision, label=f"{DISPLAY_NAMES[i]}")
        plotted = True

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} — Fold {fold} Precision–Recall")
    if plotted:
        plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    _savefig(os.path.join(out_dir, "precision_recall"))

# ============================================================
# Training (Stratified 5-fold CV + internal validation)
# ============================================================
def train_model(model_builder, model_name: str):
    all_fold_metrics = []
    fold_rows = []

    m_dir = model_dir(model_name)

    for fold, (dev_idx, test_idx) in enumerate(
        kf.split(np.arange(len(base_dataset)), labels),
        start=1
    ):
        print(f"\n🔹 Training {model_name} Fold {fold}/{KFOLDS}")
        f_dir = fold_dir(model_name, fold)
        saved_checkpoint = False

        # -------------------
        # Inner split: train vs internal validation (from dev_idx only)
        # -------------------
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=INNER_VAL_FRACTION,
            random_state=SEED + fold
        )
        y_dev = labels[dev_idx]
        inner_train_rel, val_rel = next(sss.split(np.zeros(len(dev_idx)), y_dev))
        train_idx = np.array(dev_idx)[inner_train_rel]
        val_idx = np.array(dev_idx)[val_rel]

        # Subsets with correct transforms
        train_subset = Subset(train_dataset_full, train_idx)   # augmented
        val_subset   = Subset(eval_dataset_full,  val_idx)     # clean
        test_subset  = Subset(eval_dataset_full,  test_idx)    # clean

        # DataLoaders (IMPORTANT: pass num_workers; your previous script forgot this)
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if PERSISTENT_WORKERS else None,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if PERSISTENT_WORKERS else None,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR if PERSISTENT_WORKERS else None,
        )

        # -------------------
        # Model + Optim
        # -------------------
        model = model_builder().to(device)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        best_model_path = os.path.join(f_dir, "best.pt")
        best_val_loss = float("inf")
        epochs_no_improve = 0

        train_losses, val_losses = [], []
        history = []

        # -------------------
        # Epoch loop
        # -------------------
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
            for inputs, targets in pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            avg_train_loss = running_loss / max(1, len(train_loader))
            train_acc = correct / max(1, total)

            val_loss, val_acc, _, _, _, _ = evaluate_model(model, val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": float(optimizer.param_groups[0]["lr"]),
            })

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Train: loss={avg_train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
            )

            # Checkpoint on best internal validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                ensure_dir(f_dir)
                torch.save(model.state_dict(), best_model_path)
                saved_checkpoint = True
                print(f"  ✓ Saved best checkpoint → {best_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print("  ⏹ Early stopping triggered.")
                    break

            scheduler.step(val_loss)

        # -------------------
        # Load best and evaluate on test fold
        # -------------------
        if not saved_checkpoint or not os.path.exists(best_model_path):
            print(f"⚠️ No checkpoint saved for {model_name} fold {fold}. Skipping this fold artifacts.")
            continue

        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # (Optional sanity) internal validation performance
        val_loss_best, val_acc_best, _, _, _, _ = evaluate_model(model, val_loader)
        test_loss, test_acc, per_class, y_true, y_pred, y_prob = evaluate_model(model, test_loader)

        # Save history
        ensure_dir(f_dir)
        pd.DataFrame(history).to_csv(os.path.join(f_dir, "history.csv"), index=False)

        # Plots: learning curve uses train/val; confusion/ROC/PR uses TEST
        plot_learning_curve(train_losses, val_losses, model_name, fold, f_dir)
        plot_confusion_matrix(y_true, y_pred, model_name, fold, f_dir)
        plot_roc_curve(y_true, y_prob, model_name, fold, f_dir)
        plot_precision_recall_curve(y_true, y_prob, model_name, fold, f_dir)

        # Save per-class metrics (TEST fold)
        perclass_rows = []
        for i in range(NUM_CLASSES):
            perclass_rows.append({
                "Class": i + 1,
                "ClassName": DISPLAY_NAMES[i] if i < len(DISPLAY_NAMES) else str(i + 1),
                "Precision": per_class[i]["precision"],
                "Recall": per_class[i]["recall"],
                "F1": per_class[i]["f1"],
            })
        pd.DataFrame(perclass_rows).to_csv(os.path.join(f_dir, "perclass_metrics.csv"), index=False)

        # Record fold summary (TEST is the one you report)
        all_fold_metrics.append({"accuracy": float(test_acc), "per_class_metrics": per_class})
        fold_rows.append({
            "fold": fold,
            "val_loss": float(val_loss_best),
            "val_acc": float(val_acc_best),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        })

        print(f"✅ Fold {fold} done — Val Acc: {val_acc_best:.4f} | Test Acc: {test_acc:.4f}")

    # ============================================================
    # Cross-fold aggregation (per model)
    # ============================================================
    ensure_dir(m_dir)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(m_dir, "fold_acc.csv"), index=False)

    test_acc_mean = float(fold_df["test_acc"].mean())
    test_acc_std  = float(fold_df["test_acc"].std(ddof=0))
    val_acc_mean  = float(fold_df["val_acc"].mean())
    val_acc_std   = float(fold_df["val_acc"].std(ddof=0))

    # Per-class metrics aggregation
    class_metrics = {i: {"precision": [], "recall": [], "f1": []} for i in range(NUM_CLASSES)}
    for fm in all_fold_metrics:
        pcm = fm["per_class_metrics"]
        for i in range(NUM_CLASSES):
            class_metrics[i]["precision"].append(pcm[i]["precision"])
            class_metrics[i]["recall"].append(pcm[i]["recall"])
            class_metrics[i]["f1"].append(pcm[i]["f1"])

    summary_rows = []
    for i in range(NUM_CLASSES):
        summary_rows.append({
            "Class": DISPLAY_NAMES[i] if i < len(DISPLAY_NAMES) else str(i + 1),
            "Precision_Mean": float(np.mean(class_metrics[i]["precision"])),
            "Precision_STD":  float(np.std(class_metrics[i]["precision"])),
            "Recall_Mean":    float(np.mean(class_metrics[i]["recall"])),
            "Recall_STD":     float(np.std(class_metrics[i]["recall"])),
            "F1_Mean":        float(np.mean(class_metrics[i]["f1"])),
            "F1_STD":         float(np.std(class_metrics[i]["f1"])),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(m_dir, "PerClass_Metrics.csv"), index=False)

    # Model summary JSON
    model_summary = {
        "model": model_name,
        "num_folds": KFOLDS,
        "test_acc_mean": test_acc_mean,
        "test_acc_std": test_acc_std,
        "val_acc_mean": val_acc_mean,
        "val_acc_std": val_acc_std,
        "label_smoothing": LABEL_SMOOTHING,
        "inner_val_fraction": INNER_VAL_FRACTION,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "device": str(device),
    }
    with open(os.path.join(m_dir, "model_summary.json"), "w") as f:
        json.dump(model_summary, f, indent=2)

    print(f"📌 {model_name} summary: test_acc_mean={test_acc_mean:.4f} ± {test_acc_std:.4f}")
    return model_summary

# ============================================================
# Benchmarks
# ============================================================
def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def benchmark_model(model: nn.Module, loader: DataLoader, dev: torch.device,
                    warmup: int = 5, include_transfer: bool = True):
    model.eval().to(dev)
    latencies = []
    total_items = 0

    # Warmup
    it = iter(loader)
    for _ in range(warmup):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        x = x.to(dev, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            _ = model(x)
    _sync(dev)

    # Timed
    for x, _ in loader:
        if include_transfer:
            start = time.perf_counter()
            x = x.to(dev, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _ = model(x)
            _sync(dev)
            latencies.append(time.perf_counter() - start)
        else:
            x = x.to(dev, non_blocking=True)
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _ = model(x)
            _sync(dev)
            latencies.append(time.perf_counter() - start)

        total_items += x.size(0)

    if not latencies:
        return {"median_batch_s": None, "items_per_s": None, "per_sample_ms": None}

    ips = total_items / sum(latencies)
    median_batch_s = float(median(latencies))
    mean_batch_s = float(sum(latencies) / len(latencies))
    mean_items_per_batch = float(total_items / len(latencies))
    per_sample_ms = (mean_batch_s / mean_items_per_batch) * 1000.0

    return {"median_batch_s": median_batch_s, "items_per_s": float(ips), "per_sample_ms": float(per_sample_ms)}

def run_benchmarks(model_name: str, build_fn, ckpt_path: str, dataset, batch_sizes=(32, 1)):
    rows = []
    for bs in batch_sizes:
        loader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        model = build_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        e2e = benchmark_model(model, loader, device, include_transfer=True)
        mo  = benchmark_model(model, loader, device, include_transfer=False)

        rows.append({"Model": model_name, "Batch": bs, "Mode": "end-to-end",  **e2e})
        rows.append({"Model": model_name, "Batch": bs, "Mode": "model-only", **mo})
    return rows

# ============================================================
# Main
# ============================================================
def main():
    set_publication_style()

    # Train all models
    model_summaries = []
    for model_name, builder in resnet_models.items():
        print(f"\n==============================")
        print(f"Training {model_name}...")
        print(f"==============================")
        summary = train_model(builder, model_name)
        model_summaries.append(summary)

    # Aggregate CV summary across models
    if model_summaries:
        cv_df = pd.DataFrame([{
            "Model": s["model"],
            "NumFolds": s["num_folds"],
            "TestAccMean": s["test_acc_mean"],
            "TestAccStd": s["test_acc_std"],
            "ValAccMean": s["val_acc_mean"],
            "ValAccStd": s["val_acc_std"],
            "LabelSmoothing": s["label_smoothing"],
        } for s in model_summaries])

        agg = aggregate_dir()
        ensure_dir(agg)
        cv_csv = os.path.join(agg, "cv_summary.csv")
        cv_df.to_csv(cv_csv, index=False)
        print(f"\nSaved cross-model CV summary → {cv_csv}")
        print(cv_df.to_string(index=False))

    # Benchmarks using Fold 1 best checkpoint for each model (consistent with your workflow)
    eval_full = datasets.ImageFolder(root=DATA_PATH, transform=eval_transform)
    all_bench = []
    for name, build in resnet_models.items():
        ckpt = os.path.join(fold_dir(name, 1), "best.pt")
        if not os.path.exists(ckpt):
            print(f"Missing checkpoint for {name}: {ckpt} (skipping benchmark)")
            continue
        rows = run_benchmarks(name, build, ckpt, eval_full, batch_sizes=(32, 1))
        if rows:
            ensure_dir(model_dir(name))
            pd.DataFrame(rows).to_csv(os.path.join(model_dir(name), "benchmarks.csv"), index=False)
            all_bench.extend(rows)

    if all_bench:
        agg = aggregate_dir()
        ensure_dir(agg)
        agg_bench_csv = os.path.join(agg, "benchmarks_all_models.csv")
        pd.DataFrame(all_bench).to_csv(agg_bench_csv, index=False)
        print(f"\nSaved benchmarks for all models → {agg_bench_csv}")

    # Run manifest (repro)
    run_manifest = {
        "run_id": RUN_ID,
        "run_root": RUN_ROOT,
        "data_path": DATA_PATH,
        "save_dir": SAVE_DIR,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": NUM_CLASSES,
            "kfolds": KFOLDS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "label_smoothing": LABEL_SMOOTHING,
            "inner_val_fraction": INNER_VAL_FRACTION,
            "use_amp": USE_AMP,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
        },
        "models_trained": list(resnet_models.keys()),
        "seed": SEED,
    }
    manifest_path = os.path.join(RUN_ROOT, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)

    remove_empty_dirs(RUN_ROOT)

    print(f"\nRun manifest saved → {manifest_path}")
    print(f"All artifacts are under: {RUN_ROOT}")

if __name__ == "__main__":
    main()
