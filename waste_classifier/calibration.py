import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

def get_logits_and_labels(model, dataloader, device):
    """Estrae i logit e le etichette reali da un DataLoader usando il modello."""
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            
    return torch.cat(all_logits), torch.cat(all_labels)

def fit_temperature(model, val_loader, device):
    """
    Applica il Temperature Scaling (Guo et al., 2017) calcolando 
    la Temperatura (T) ottimale che minimizza la Negative Log Likelihood (NLL).
    """
    logits, labels = get_logits_and_labels(model, val_loader, device)
    
    # Obiettivo: minimizzare la NLL (CrossEntropy) rispetto a T
    nll_criterion = torch.nn.CrossEntropyLoss()
    
    def eval_t(t_val):
        t_tensor = torch.tensor(t_val, dtype=torch.float32)
        scaled_logits = logits / t_tensor
        return nll_criterion(scaled_logits, labels).item()

    # T ottimale tipicamente tra 0.1 e 10
    res = minimize(eval_t, x0=1.0, bounds=[(0.1, 10.0)], method='L-BFGS-B')
    best_t = res.x[0]
    
    print(f"[Calibration] Temperature trovata: {best_t:.4f}")
    return best_t

def apply_reject_routing(logits, T, tau, undiff_idx, carveout_indices=None):
    """
    Applica la regola di routing OSR:
    Se la probabilità massima è < tau e le top-2 classi non rientrano nel carveout,
    instrada la predizione verso la classe `undiff_idx`.
    """
    probs = F.softmax(logits / T, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    
    if carveout_indices is None:
        carveout_indices = []
        
    final_preds = preds.clone()
    
    # Identifica quali campioni sono incerti (sotto la soglia)
    uncertain_mask = max_probs < tau
    
    # Identifica il carve-out (Difesa 3: se le top 2 classi appartengono al triangolo di ambiguità)
    if len(carveout_indices) >= 2:
        top2_probs, top2_indices = torch.topk(probs, 2, dim=1)
        # Converte gli indici in un tensore sullo stesso device
        carveout_tensor = torch.tensor(carveout_indices, device=probs.device)
        
        # Verifica quali elementi dei top-2 appartengono al carveout [Batch, 2]
        isin_carveout = torch.isin(top2_indices, carveout_tensor)
        
        # Entrambi i top-2 devono essere presenti (.all lungo la dimensione delle classi)
        in_carveout_mask = isin_carveout.all(dim=1)
    else:
        in_carveout_mask = torch.zeros(len(probs), dtype=torch.bool, device=probs.device)
        
    # Applica il routing solo a chi è incerto E NON rientra nel carve-out
    routing_mask = uncertain_mask & (~in_carveout_mask)
    final_preds[routing_mask] = undiff_idx
    
    return final_preds

def find_optimal_threshold(model, val_loader, device, T, undiff_idx, carveout_indices, n_classes=8):
    """
    Scansiona una serie di soglie tau sul validation set e trova quella che
    massimizza la Balanced Accuracy globale (a 8 classi), applicando la regola di reject.
    """
    logits, labels = get_logits_and_labels(model, val_loader, device)
    labels = labels.numpy()
    
    best_tau = 0.0
    best_bal_acc = 0.0
    
    # Scansioniamo la soglia da 0.1 a 0.99
    taus = np.linspace(0.1, 0.99, 90)
    
    for tau in taus:
        routed_preds = apply_reject_routing(logits, T, tau, undiff_idx, carveout_indices).numpy()
        bal_acc = balanced_accuracy_score(labels, routed_preds)
        
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_tau = tau
            
    print(f"[Calibration] Soglia ottima trovata: tau* = {best_tau:.4f} (Val Balanced Acc: {best_bal_acc*100:.2f}%)")
    return best_tau, best_bal_acc

