# predict_next_recipient.py
import math, argparse, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------- Data prep ----------
def build_index(s, add_unk=False):
    items = sorted([str(x) for x in s.dropna().unique().tolist()])
    if add_unk: items = ['<UNK>'] + items
    return {t:i for i,t in enumerate(items)}

def load_sequences(csv_path, window=8):
    df = pd.read_csv(csv_path)

    # Keep only passes with known recipient
    p = df[(df['Type'] == 'PASS') & df['To'].notna()].copy()

    # Fill & clip coordinates
    for c in ['Start X','Start Y','End X','End Y']:
        p[c] = p[c].astype(float).fillna(p[c].mean()).clip(0,1)

    # Time + geometry
    p['Start Time [s]'] = p['Start Time [s]'].astype(float)
    p['event_dt'] = p['Start Time [s]'].diff().fillna(0.0)
    dx, dy = p['End X']-p['Start X'], p['End Y']-p['Start Y']
    p['pass_len'] = np.sqrt(dx*dx + dy*dy)
    p['pass_ang'] = np.arctan2(dy, dx)

    # Vocabularies
    iteam   = build_index(p['Team'])
    itype   = build_index(p['Type'], add_unk=True)
    isubt   = build_index(p['Subtype'], add_unk=True)
    ifrom   = build_index(p['From'], add_unk=True)
    ito     = build_index(p['To'])

    # Sort chronologically
    p = p.sort_values(['Period','Start Time [s]','Start Frame']).reset_index(drop=True)

    # Slide a window of N prior events -> predict current 'To'
    cat_seqs, num_seqs, y = [], [], []
    bufC, bufN = [], []
    for _, r in p.iterrows():
        C = (
            iteam[str(r['Team'])],
            itype.get(str(r['Type']), itype['<UNK>']),
            isubt.get(str(r['Subtype']), isubt['<UNK>']),
            ifrom.get(str(r['From']), ifrom['<UNK>']),
        )
        N = np.array([
            r['Start X'], r['Start Y'], r['End X'], r['End Y'],
            r['pass_len'], r['pass_ang'], r['event_dt'], r['Period']
        ], dtype=np.float32)

        bufC.append(C); bufN.append(N)
        if len(bufC) >= window + 1:
            cat_seqs.append(np.array(bufC[-(window+1):-1], dtype=np.int64))
            num_seqs.append(np.stack(bufN[-(window+1):-1]).astype(np.float32))
            y.append(ito[str(r['To'])])

    cat_seqs = np.stack(cat_seqs)           # [num_ex, W, 4]
    num_seqs = np.stack(num_seqs)           # [num_ex, W, 8]
    y        = np.array(y, dtype=np.int64)  # [num_ex]

    meta = dict(
        n_team=len(iteam), n_type=len(itype), n_subtype=len(isubt),
        n_from=len(ifrom), n_to=len(ito), idx_to=ito
    )
    return cat_seqs, num_seqs, y, meta

def split_ordered(n, tr=0.7, va=0.15):
    a = int(n*tr); b = int(n*(tr+va))
    idx = np.arange(n)
    return idx[:a], idx[a:b], idx[b:]

# ---------- Models ----------
class EventEncoder(nn.Module):
    def __init__(self, nteam, ntype, nsubt, nfrom, num_dim, emb=16, hid=32, proj=64):
        super().__init__()
        self.eT = nn.Embedding(nteam, emb)
        self.eY = nn.Embedding(ntype, emb)
        self.eS = nn.Embedding(nsubt, emb)
        self.eF = nn.Embedding(nfrom, emb)
        self.mlp = nn.Sequential(
            nn.Linear(num_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        self.proj = nn.Linear(4*emb + hid, proj)

    def forward(self, C, N):
        t,y,s,f = C[...,0], C[...,1], C[...,2], C[...,3]
        E = torch.cat([self.eT(t), self.eY(y), self.eS(s), self.eF(f)], dim=-1)
        Z = self.mlp(N)
        return self.proj(torch.cat([E, Z], dim=-1))  # [B, W, proj]

class LSTMHead(nn.Module):
    def __init__(self, encoder, n_classes, proj=64, hidden=128, dropout=0.1):
        super().__init__()
        self.enc = encoder
        self.rnn = nn.LSTM(input_size=proj, hidden_size=hidden, batch_first=True)
        self.do  = nn.Dropout(dropout)
        self.fc  = nn.Linear(hidden, n_classes)

    def forward(self, C, N):
        x, _ = self.rnn(self.enc(C, N))     # [B, W, H]
        h = self.do(x[:, -1, :])            # last step
        return self.fc(h)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerHead(nn.Module):
    def __init__(self, encoder, n_classes, d_model=64, nhead=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.enc = encoder
        self.pos = PositionalEncoding(d_model)
        block = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.tr  = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc   = nn.Linear(d_model, n_classes)

    def forward(self, C, N):
        x = self.pos(self.enc(C, N))
        z = self.tr(x)                  # [B, W, D]
        h = self.norm(z[:, -1, :])
        return self.fc(h)

# ---------- Train / Evaluate ----------
def topk_acc(logits, y, k=5):
    k = min(k, logits.size(1))
    tk = logits.topk(k, dim=-1).indices
    return (tk == y.view(-1,1)).any(dim=1).float().mean().item()

def make_loader(C, N, y, bs=128):
    ds = TensorDataset(torch.tensor(C, dtype=torch.long),
                       torch.tensor(N, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=bs, shuffle=False)

def train_and_eval(C, N, y, meta, model_type='lstm', epochs=8, lr=1e-3, bs=128):
    tr, va, te = split_ordered(len(y))
    trL = make_loader(C[tr], N[tr], y[tr], bs)
    vaL = make_loader(C[va], N[va], y[va], bs)
    teL = make_loader(C[te], N[te], y[te], bs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = EventEncoder(meta['n_team'], meta['n_type'], meta['n_subtype'], meta['n_from'], num_dim=N.shape[-1])

    model = (LSTMHead(enc, meta['n_to'])
             if model_type=='lstm'
             else TransformerHead(EventEncoder(meta['n_team'], meta['n_type'], meta['n_subtype'], meta['n_from'], num_dim=N.shape[-1]),
                                  meta['n_to']))

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    def eval_loader(loader):
        model.eval()
        with torch.no_grad():
            all_logits, all_y = [], []
            for Cb, Nb, yb in loader:
                Cb, Nb, yb = Cb.to(device), Nb.to(device), yb.to(device)
                all_logits.append(model(Cb, Nb).cpu())
                all_y.append(yb.cpu())
        logits = torch.cat(all_logits, 0)
        ytrue  = torch.cat(all_y, 0)
        return dict(
            acc1 = (logits.argmax(-1) == ytrue).float().mean().item(),
            acc3 = topk_acc(logits, ytrue, 3),
            acc5 = topk_acc(logits, ytrue, 5),
        )

    best, best_state = -1.0, None
    for ep in range(1, epochs+1):
        model.train()
        for Cb, Nb, yb in trL:
            Cb, Nb, yb = Cb.to(device), Nb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(Cb, Nb), yb)
            loss.backward()
            opt.step()
        val = eval_loader(vaL)
        print(f"Epoch {ep}: val@1={val['acc1']:.3f}  @3={val['acc3']:.3f}  @5={val['acc5']:.3f}")
        if val['acc1'] > best:
            best, best_state = val['acc1'], {k:v.cpu() for k,v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    test = eval_loader(teL)
    return test

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, required=True)
    ap.add_argument('--model', type=str, default='lstm', choices=['lstm','transformer'])
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    C, N, y, meta = load_sequences(args.csv, window=args.window)
    metrics = train_and_eval(C, N, y, meta, model_type=args.model, epochs=args.epochs, lr=args.lr, bs=args.bs)
    print("TEST:", metrics)
