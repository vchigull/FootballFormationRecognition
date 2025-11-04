import argparse, os, torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from datasets import FolderDataset
from transforms import make_train_transform, make_eval_transform
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in tqdm(loader, desc="train"):
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    all_y, all_p = [], []
    for x,y in tqdm(loader, desc="eval"):
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
        all_y.append(y.cpu().numpy()); all_p.append(pred.cpu().numpy())
    if len(all_y):
        all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
        try:
            print(classification_report(all_y, all_p, target_names=class_names, digits=3))
        except Exception:
            pass
    return loss_sum/total if total>0 else 0.0, correct/total if total>0 else 0.0

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_train, t_eval = make_train_transform(args.img_size), make_eval_transform(args.img_size)
    ds_full = FolderDataset(args.data_dir, transform=t_train)
    if len(ds_full) < 3:
        print("WARNING: Very small dataset — this is fine for smoke test only.")
    n = len(ds_full)
    n_train = max(1, int(0.8*n)); n_val = max(1, int(0.1*n)); n_test = max(1, n - n_train - n_val)
    ds_train, ds_val, ds_test = random_split(ds_full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))
    ds_val.dataset.transform = t_eval
    ds_test.dataset.transform = t_eval

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(ds_full.classes))
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader   = DataLoader(ds_val,   batch_size=args.bs, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=args.bs, shuffle=False, num_workers=0)

    best_val = -1.0
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, 'best_offense.pt')
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, device, ds_full.classes)
        print(f"Epoch {epoch}: train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")
        if val_acc >= best_val:
            best_val = val_acc
            torch.save({'model': model.state_dict(), 'classes': ds_full.classes}, ckpt_path)
    # final test
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    print("=== TEST ===")
    evaluate(model, test_loader, device, ds_full.classes)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data/offense')
    p.add_argument('--out_dir', type=str, default='checkpoints')
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--bs', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    args = p.parse_args()
    main(args)
