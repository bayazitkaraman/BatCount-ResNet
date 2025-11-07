import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
import random
import time
from statistics import median

# ==============================
# Hyperparameters & Config
# ==============================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 12
KFOLDS = 5
EARLY_STOP_PATIENCE = 7

DATA_PATH = "Data/Final Testing Dataset/"
SAVE_DIR = "SavedModels"

from datetime import datetime

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_ROOT = os.path.join(SAVE_DIR, RUN_ID)
os.makedirs(RUN_ROOT, exist_ok=True)

def model_dir(model_name):
    d = os.path.join(RUN_ROOT, model_name)
    os.makedirs(d, exist_ok=True)
    return d

def fold_dir(model_name, fold):
    d = os.path.join(model_dir(model_name), f"fold{fold:02d}")
    os.makedirs(d, exist_ok=True)
    return d

NUM_WORKERS = max(4, (os.cpu_count() or 20))
PIN_MEMORY = torch.cuda.is_available()
USE_AMP = torch.cuda.is_available()  # mixed precision if CUDA
PREFETCH_FACTOR = None
PERSISTENT_WORKERS = False

SEED = 42
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True
set_seed()

# --- Publication style (Matplotlib) ---
def set_publication_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        # fonts
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # lines & layout
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
    })

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==============================
# Transforms (train vs test/val)
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(15, scale=(0.8, 1.2), shear=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Make a base dataset for indexing only (no transform to avoid double-applying)
base_dataset = datasets.ImageFolder(root=DATA_PATH, transform=None)
# Two mirror datasets that use the same file order but different transforms
train_dataset_full = datasets.ImageFolder(root=DATA_PATH, transform=train_transform)
eval_dataset_full  = datasets.ImageFolder(root=DATA_PATH, transform=test_transform)

labels = base_dataset.targets  # or [y for _, y in base_dataset.imgs]
kf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SEED)

# ==============================
# Robust torchvision weights helper
# ==============================
def get_imagenet_weights_enum(model_name: str):
    try:
        mapping = {
            "resnet18": tv_models.ResNet18_Weights.IMAGENET1K_V1,
            "resnet34": tv_models.ResNet34_Weights.IMAGENET1K_V1,
            "resnet50": tv_models.ResNet50_Weights.IMAGENET1K_V1,
            "resnet101": tv_models.ResNet101_Weights.IMAGENET1K_V1,
        }
        return mapping[model_name]
    except Exception:
        return "IMAGENET1K_V1"  # fallback for older torchvision

# ==============================
# Custom ResNet (single-model wrapper)
# ==============================
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

    def forward(self, x):
        return self.model(x)

resnet_models = {
    "ResNet18":  lambda: CustomResNet("resnet18"),
    "ResNet34":  lambda: CustomResNet("resnet34"),
    "ResNet50":  lambda: CustomResNet("resnet50"),
    "ResNet101": lambda: CustomResNet("resnet101"),
}

# ==============================
# Evaluation
# ==============================
def evaluate_model(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    val_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            predicted = probs.argmax(dim=1)

            #use the batch targets, not the global "labels"
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    avg_val_loss = val_loss / len(loader)
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_metrics = {}
    for class_idx in range(NUM_CLASSES):
        key = str(class_idx)
        if key in report:
            per_class_metrics[class_idx] = {
                'precision': report[key]['precision'],
                'recall': report[key]['recall'],
                'f1': report[key]['f1-score']
            }
        else:
            per_class_metrics[class_idx] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return avg_val_loss, accuracy, per_class_metrics, np.array(y_true), np.array(y_pred), np.array(y_prob)

# ==============================
# Visualization (figures)
# ==============================
def plot_learning_curve(train_losses, val_losses, model_name, fold, out_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{model_name} — Fold {fold} Learning Curve')
    plt.legend(frameon=False)
    plt.tight_layout()
    out_base = os.path.join(out_dir, 'learning_curve')
    plt.savefig(out_base + '.png'); plt.savefig(out_base + '.pdf')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, fold, out_dir):
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    class_labels = [str(i + 1) for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap='Blues', colorbar=False, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{model_name} — Fold {fold} Confusion Matrix')
    plt.tight_layout()
    out_base = os.path.join(out_dir, 'confusion_matrix')
    plt.savefig(out_base + '.png'); plt.savefig(out_base + '.pdf')
    plt.close()

def plot_roc_curve(y_true, y_prob, model_name, fold, out_dir):
    plt.figure(figsize=(6, 5))
    plotted = False
    for i in range(NUM_CLASSES):
        if (y_true == i).sum() == 0:
            continue
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i+1} (AUC={roc_auc:.3f})')
        plotted = True
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.0)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} — Fold {fold} ROC')
    if plotted: plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    out_base = os.path.join(out_dir, 'roc')
    plt.savefig(out_base + '.png'); plt.savefig(out_base + '.pdf')
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, model_name, fold, out_dir):
    plt.figure(figsize=(6, 5))
    plotted = False
    for i in range(NUM_CLASSES):
        if (y_true == i).sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_prob[:, i])
        plt.plot(recall, precision, label=f'Class {i+1}')
        plotted = True
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'{model_name} — Fold {fold} Precision–Recall')
    if plotted: plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    out_base = os.path.join(out_dir, 'precision_recall')
    plt.savefig(out_base + '.png'); plt.savefig(out_base + '.pdf')
    plt.close()

# ==============================
# Training (K-Fold with correct transforms)
# ==============================
def train_model(model_class, model_name):
    all_metrics = []
    fold_accs = []

    # one parent dir per model for this RUN_ID
    m_dir = model_dir(model_name)

    for fold, (train_idx, val_idx) in enumerate(
            kf.split(np.arange(len(base_dataset)), labels), start=1):
        print(f"\n🔹 Training {model_name} Fold {fold}/{KFOLDS}")

        # per-fold directory (plots, history, best.pt, per-class CSV)
        f_dir = fold_dir(model_name, fold)
        hist_rows = []

        # data
        train_subset = Subset(train_dataset_full, train_idx)
        val_subset   = Subset(eval_dataset_full,  val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=BATCH_SIZE, shuffle=True,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR
        )
        val_loader = DataLoader(
            val_subset, batch_size=BATCH_SIZE, shuffle=False,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR
        )

        # model & optimization
        model = model_class().to(device)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        best_model_path = os.path.join(f_dir, "best.pt")
        best_val_loss = float('inf')
        epochs_no_improve = 0

        train_losses, val_losses = [], []
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

        # ---- train epochs ----
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            correct, total = 0, 0

            for inputs, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
                inputs, labels_batch = inputs.to(device, non_blocking=True), labels_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                correct += (outputs.argmax(1) == labels_batch).sum().item()
                total += labels_batch.size(0)

            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total

            # validate
            val_loss, val_accuracy, per_class_metrics, y_true, y_pred, y_prob = evaluate_model(model, val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)

            hist_rows.append({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_acc": train_accuracy,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
            })

            print(f"Epoch {epoch + 1}/{EPOCHS} - "
                  f"Train Acc: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # early stopping on val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved: {best_model_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print("Early stopping triggered!")
                    break

            scheduler.step(val_loss)

        # --- evaluate best & save artifacts for this fold ---
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        val_loss, val_accuracy, per_class_metrics, y_true, y_pred, y_prob = evaluate_model(model, val_loader)

        # per-epoch history
        pd.DataFrame(hist_rows).to_csv(os.path.join(f_dir, "history.csv"), index=False)

        # plots into this fold dir
        plot_learning_curve(train_losses, val_losses, model_name, fold, f_dir)
        plot_confusion_matrix(y_true, y_pred, model_name, fold, f_dir)
        plot_roc_curve(y_true, y_prob, model_name, fold, f_dir)
        plot_precision_recall_curve(y_true, y_prob, model_name, fold, f_dir)

        # per-class metrics for this fold (labels 1..NUM_CLASSES)
        pd.DataFrame([
            {
                "Class": i + 1,
                "Precision": per_class_metrics[i]["precision"],
                "Recall":    per_class_metrics[i]["recall"],
                "F1":        per_class_metrics[i]["f1"]
            }
            for i in range(NUM_CLASSES)
        ]).to_csv(os.path.join(f_dir, "perclass_metrics.csv"), index=False)

        all_metrics.append({'accuracy': val_accuracy, 'per_class_metrics': per_class_metrics})
        fold_accs.append({"fold": fold, "val_acc": val_accuracy})

    # ---- cross-fold aggregation for this model ----
    mean_accuracy = np.mean([m['accuracy'] for m in all_metrics])

    class_metrics_summary = {i: {'precision': [], 'recall': [], 'f1': []} for i in range(NUM_CLASSES)}
    for fold_metrics in all_metrics:
        pcm = fold_metrics['per_class_metrics']
        for i in range(NUM_CLASSES):
            class_metrics_summary[i]['precision'].append(pcm[i]['precision'])
            class_metrics_summary[i]['recall'].append(pcm[i]['recall'])
            class_metrics_summary[i]['f1'].append(pcm[i]['f1'])

    summary_rows = []
    for i in range(NUM_CLASSES):
        summary_rows.append({
            'Class': i + 1,  # 1..NUM_CLASSES
            'Precision_Mean': float(np.mean(class_metrics_summary[i]['precision'])),
            'Precision_STD':  float(np.std (class_metrics_summary[i]['precision'])),
            'Recall_Mean':    float(np.mean(class_metrics_summary[i]['recall'])),
            'Recall_STD':     float(np.std (class_metrics_summary[i]['recall'])),
            'F1-Score_Mean':  float(np.mean(class_metrics_summary[i]['f1'])),
            'F1-Score_STD':   float(np.std (class_metrics_summary[i]['f1'])),
        })

    # cross-fold per-class summary (saved under the model folder)
    pd.DataFrame(summary_rows).to_csv(os.path.join(m_dir, "PerClass_Metrics.csv"), index=False)
    print(f'Per-class metrics saved to: {os.path.join(m_dir, "PerClass_Metrics.csv")}')

    # per-model fold accuracy table
    pd.DataFrame(fold_accs).to_csv(os.path.join(m_dir, "fold_acc.csv"), index=False)

    # per-model summary JSON (mean/std)
    summary = {
        "model": model_name,
        "num_folds": KFOLDS,
        "val_acc_mean": float(np.mean([r["val_acc"] for r in fold_accs])),
        "val_acc_std":  float(np.std ([r["val_acc"] for r in fold_accs])),
    }
    with open(os.path.join(m_dir, "model_summary.json"), "w") as f:
        import json; json.dump(summary, f, indent=2)

    return {"accuracy": mean_accuracy}

# ==============================
# Benchmarks
# ==============================
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def benchmark_model(model, loader, device, warmup=5, include_transfer=True):
    model.eval().to(device)
    latencies, total_items = [], 0

    # warmup
    it = iter(loader)
    for _ in range(warmup):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        if include_transfer:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _ = model(x)
        else:
            # move BEFORE timing so warmup matches timed path
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _ = model(x)
    _sync(device)

    # timed
    for x, _ in loader:
        if include_transfer:
            start = time.perf_counter()
            x = x.to(device, non_blocking=True)           # H→D measured
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _ = model(x)
            _sync(device)
            latencies.append(time.perf_counter() - start)
        else:
            x = x.to(device, non_blocking=True)           # H→D NOT measured
            start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _ = model(x)
            _sync(device)
            latencies.append(time.perf_counter() - start)

        total_items += x.size(0)

    mean_batch = sum(latencies) / len(latencies)
    ips = total_items / sum(latencies)
    per_sample_ms = (mean_batch / (total_items / len(latencies))) * 1000.0
    return {"median_batch_s": median(latencies), "items_per_s": ips, "per_sample_ms": per_sample_ms}

def run_benchmarks(model_name, build_fn, ckpt_path, dataset, batch_sizes=(32, 1)):
    rows = []
    for bs in batch_sizes:
        loader = DataLoader(
            dataset, batch_size=bs, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
        )
        model = build_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # end-to-end (transfer + forward)
        e2e = benchmark_model(model, loader, device, include_transfer=True)
        # model-only (forward only; host→device included inside call but we don't time it)
        mo  = benchmark_model(model, loader, device, include_transfer=False)

        rows.append({"Model": model_name, "Batch": bs, "Mode": "end-to-end",  **e2e})
        rows.append({"Model": model_name, "Batch": bs, "Mode": "model-only", **mo})
    return rows

# ==============================
# Main
# ==============================
def main():
    set_publication_style()

    # ---- Train all models ----
    results = {}
    for model_name, model_fn in resnet_models.items():
        print(f"\nTraining {model_name}...")
        results[model_name] = train_model(model_fn, model_name)

    # ---- Print results to console ----
    print("\nFinal Cross-Validation Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")

    # ---- Build a cross-model CV summary table at RUN_ROOT ----
    # (reads each model's model_summary.json that train_model() already wrote)
    cv_rows = []
    for model_name in resnet_models.keys():
        m_dir = model_dir(model_name)
        summary_json = os.path.join(m_dir, "model_summary.json")
        if os.path.exists(summary_json):
            import json
            with open(summary_json, "r") as f:
                s = json.load(f)
            cv_rows.append({
                "Model": s.get("model", model_name),
                "NumFolds": s.get("num_folds", None),
                "ValAccMean": s.get("val_acc_mean", None),
                "ValAccStd":  s.get("val_acc_std", None),
            })
        else:
            print(f"Missing {summary_json}; skipping CV summary row for {model_name}")

    if cv_rows:
        cv_df = pd.DataFrame(cv_rows)
        cv_path = os.path.join(RUN_ROOT, "_aggregate"); os.makedirs(cv_path, exist_ok=True)
        cv_csv = os.path.join(cv_path, "cv_summary.csv")
        cv_df.to_csv(cv_csv, index=False)
        print(f"\nSaved cross-model CV summary → {cv_csv}")
        print(cv_df.to_string(index=False))

    # ---- Benchmarks using Fold 1 best ----
    eval_full = datasets.ImageFolder(root=DATA_PATH, transform=test_transform)
    all_bench = []

    for name, build in resnet_models.items():
        ckpt = os.path.join(fold_dir(name, 1), "best.pt")
        if not os.path.exists(ckpt):
            print(f"Missing checkpoint for {name}: {ckpt} (skipping benchmark)")
            continue
        rows = run_benchmarks(name, build, ckpt, eval_full, batch_sizes=(32, 1))
        pd.DataFrame(rows).to_csv(os.path.join(model_dir(name), "benchmarks.csv"), index=False)
        all_bench.extend(rows)

    # ---- Global benchmarks table ----
    if all_bench:
        agg_dir = os.path.join(RUN_ROOT, "_aggregate"); os.makedirs(agg_dir, exist_ok=True)
        agg_bench_csv = os.path.join(agg_dir, "benchmarks_all_models.csv")
        pd.DataFrame(all_bench).to_csv(agg_bench_csv, index=False)
        print(f"\nSaved benchmarks for all models → {agg_bench_csv}")

    # ---- Run manifest for reproducibility ----
    run_manifest = {
        "run_id": RUN_ID,
        "run_root": RUN_ROOT,
        "data_path": DATA_PATH,
        "save_dir": SAVE_DIR,
        "device": str(device),
        "hyperparams": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": NUM_CLASSES,
            "kfolds": KFOLDS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "use_amp": USE_AMP,
        },
        "models_trained": list(resnet_models.keys())
    }
    manifest_path = os.path.join(RUN_ROOT, "run_manifest.json")
    import json
    with open(manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"\nRun manifest saved → {manifest_path}")
    print(f"All artifacts are under: {RUN_ROOT}")

if __name__ == "__main__":
    main()
