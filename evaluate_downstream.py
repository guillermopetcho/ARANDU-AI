import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from models.moco import ModelBase

def get_val_transforms():
    """Transformaciones estándar para validación."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def evaluate():
    parser = argparse.ArgumentParser(description="Análisis Profesional del Encoder Entrenado")
    parser.add_argument("--config", type=str, default="config/moco.yaml", help="Ruta al archivo yaml")
    args = parser.parse_args()

    # 1. Cargar Configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Evaluando en dispositivo: {device}")

    # 2. Cargar Dataset de Validación
    val_dir = config['paths']['eval_val_root']
    if not os.path.exists(val_dir):
        # Fallback Kaggle
        val_dir = "/kaggle/input/datasets/joaquinignaciopetcho/base-soja-encoder-full/BASE-SOJA-ENCODER-FULL/val"
        
    print(f"[*] Cargando dataset desde: {val_dir}")
    val_ds = datasets.ImageFolder(val_dir, transform=get_val_transforms())
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    class_names = val_ds.classes
    num_classes = len(class_names)
    print(f"[*] Clases detectadas ({num_classes}): {class_names}")

    # 3. Reconstruir el Modelo
    encoder_path = config['paths']['encoder_export_path']
    head_path = encoder_path.replace('.pth', '_head.pth')
    
    print(f"[*] Cargando Encoder: {encoder_path}")
    encoder = ModelBase(
        dim=config['moco']['dim'],
        predictor_hidden_dim=config['moco'].get('predictor_hidden_dim', 4096)
    )
    # Cargar pesos con weights_only=True por seguridad
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu', weights_only=True))
    encoder = encoder.to(device)
    encoder.eval()

    print(f"[*] Cargando Sonda Lineal (Clasificador): {head_path}")
    # Inferir dimensión proyectada pasando un tensor dummy
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        proj_dim = encoder(dummy, use_predictor=False).shape[-1]
    
    classifier = nn.Sequential(
        nn.LayerNorm(proj_dim),
        nn.Linear(proj_dim, num_classes)
    )
    classifier.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
    classifier = classifier.to(device)
    classifier.eval()

    # 4. Inferencia
    all_preds = []
    all_labels = []
    all_probs = []

    print("[*] Iniciando Inferencia sobre el conjunto de validación...")
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            # Extraer features normalizados (igual que en linear_probe.py)
            feats = F.normalize(encoder(x, use_predictor=False), dim=1)
            logits = classifier(feats)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 5. Calcular Métricas Profesionales
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*50)
    print("📊 REPORTE DE RENDIMIENTO (WEIGHTED)")
    print("="*50)
    print(f"Accuracy : {acc:.4f}  (Porcentaje total de aciertos)")
    print(f"Precision: {precision:.4f}  (Calidad de los positivos detectados)")
    print(f"Recall   : {recall:.4f}  (Capacidad de no omitir enfermos)")
    print(f"F1-Score : {f1:.4f}  (Media armónica Prec-Recall)")
    print("="*50)

    print("\n🔍 REPORTE POR CLASE:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # 6. Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - AranduSSL (Linear Probe)')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    
    cm_path = "/kaggle/working/confusion_matrix.png" if "kaggle" in val_dir else "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    print(f"\n🖼️ Matriz de Confusión guardada en: {cm_path}")
    
    # 7. Análisis de Errores (Opcional pero muy útil)
    errores = np.sum(all_labels != all_preds)
    total = len(all_labels)
    print(f"\n💡 Análisis Rápido: El modelo se equivocó en {errores} de {total} imágenes (Tasa de error: {(errores/total)*100:.2f}%).")
    print("\n✅ Análisis Finalizado con Éxito.")

if __name__ == "__main__":
    evaluate()
