import os
import io
import json
from flask import Flask, request, send_file, render_template, abort
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from models.models import LSTEncoder, PM25Encoder, LandCoverEncoder, FusionAttentionUNet

# ── IMPORTANT: non-interactive backend ─────────────────────────────────────────
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

app = Flask(__name__)

# ── DIRECTORIES ────────────────────────────────────────────────────────────────
PM25_CACHE       = "cache/pm25/pm25_cache"
PM25_MODIFIED    = "cache/pm25/pm25_cache_modified"
LST_CACHE        = "cache/lst/lst_cache"
LST_MODIFIED     = "cache/lst/lst_cache_modified"
ESRI_CACHE       = "cache/landcover/landcover_cache"
ESRI_MODIFIED    = "cache/landcover/landcover_cache_modified"
UHII_CACHE       = "cache/uhii/uhii_cache"

for d in (PM25_CACHE, PM25_MODIFIED, LST_CACHE, LST_MODIFIED, ESRI_CACHE, ESRI_MODIFIED, UHII_CACHE):
    os.makedirs(d, exist_ok=True)

# ── LST COLORMAP & NORMALIZATION ───────────────────────────────────────────────
lst_colors = [
    '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
    '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'
]
LST_CMAP = mcolors.LinearSegmentedColormap.from_list('custom', lst_colors, N=51)
LST_NORM_C = mcolors.BoundaryNorm(np.arange(0, 52, 1), ncolors=51)
LST_NORM_K = mcolors.Normalize(vmin=290, vmax=320)

# ── Landcover COLORMAP & NORMALIZATION ───────────────────────────────────────────────
esri_palette = [
  "#ffff64", "#ffff64", "#ffff00", "#aaf0f0", "#4c7300", "#006400", "#a8c800", "#00a000",
  "#005000", "#003c00", "#286400", "#285000", "#a0b432", "#788200", "#966400", "#964b00",
  "#966400", "#ffb432", "#ffdcd2", "#ffebaf", "#ffd278", "#ffebaf", "#00a884", "#73ffdf",
  "#9ebb3b", "#828282", "#f57ab6", "#66cdab", "#444f89", "#c31400", "#fff5d7", "#dcdcdc",
  "#fff5d7", "#0046c8", "#ffffff", "#ffffff"
]
ESRI_CMAP = ListedColormap(esri_palette)
ESRI_NORM = BoundaryNorm(np.arange(1, 38), ncolors=36)

# ── UHII CONFIGURATION ────────────────────────────────────────────────────────
UHII_CACHE    = "cache/uhii/uhii_cache"        # folder with uhii_YYYY_MM.npy
UHII_VMIN     = 0.1
UHII_VMAX     = 2.5

# ensure cache dir exists
os.makedirs(UHII_CACHE, exist_ok=True)

# ── DATASET HELPERS ────────────────────────────────────────────────────────────
_cache_store = {}

def load_pm25(year:int, month:int):
    key = f"pm25_{year}_{month:02d}"
    if key not in _cache_store:
        fn = os.path.join(PM25_CACHE,    f"pm25_{year:04d}_{month:02d}.npy")
        fn_mod = os.path.join(PM25_MODIFIED, f"pm25_{year:04d}_{month:02d}_edited.npy")
        arr = np.load(fn_mod) if os.path.isfile(fn_mod) else np.load(fn)
        _cache_store[key] = arr
    return _cache_store[key].copy()

def save_pm25(arr, year:int, month:int):
    fn = os.path.join(PM25_MODIFIED, f"pm25_{year:04d}_{month:02d}_edited.npy")
    np.save(fn, arr)

def load_lst(year:int, month:int, use_kelvin:bool):
    unit = 'K' if use_kelvin else 'C'
    key = f"lst_{year}_{month:02d}_{unit}"
    if key not in _cache_store:
        fn = os.path.join(LST_CACHE,    f"lst_{year:04d}_{month:02d}.npy")
        fn_mod = os.path.join(LST_MODIFIED, f"lst_{year:04d}_{month:02d}_{unit}_edited.npy")
        # arr = np.load(fn_mod) if os.path.isfile(fn_mod) else np.load(fn)
        arr = np.load(fn)
        if use_kelvin:
            arr = arr + 273.15
        _cache_store[key] = arr
    return _cache_store[key].copy()

def save_lst(arr, year:int, month:int, use_kelvin:bool):
    unit = 'K' if use_kelvin else 'C'
    fn = os.path.join(LST_MODIFIED, f"lst_{year:04d}_{month:02d}_{unit}_edited.npy")
    np.save(fn, arr)

def load_esri(year: int):
    key = f"esri_{year}"
    if key not in _cache_store:
        fn = os.path.join(ESRI_CACHE,     f"landcover_{year:04d}.npy")
        fnm = os.path.join(ESRI_MODIFIED,      f"landcover_{year:04d}_edited.npy")
        arr = np.load(fnm) if os.path.isfile(fnm) else np.load(fn)
        _cache_store[key] = arr
    return _cache_store[key].copy()

def save_esri(arr, year: int):
    fn = os.path.join(esri_mod_cache, f"landcover_{year:04d}_edited.npy")
    np.save(fn, arr)

# In‐memory store for UHII
uhii_store = {}

def load_uhii(year:int, month:int) -> np.ndarray:
    key = f"{year}_{month:02d}"
    if key not in uhii_store:
        fn = os.path.join(UHII_CACHE, f"uhii_{year:04d}_{month:02d}.npy")
        uhii_store[key] = np.load(fn)
    return uhii_store[key].copy()

# ── MODEL LOADING ──────────────────────────────────────────────────────────────
lst_enc = LSTEncoder().to(torch_device)
pm_enc = PM25Encoder().to(torch_device)
lc_enc  = LandCoverEncoder().to(torch_device)
in_ch  = lst_enc.out_channels + pm_enc.out_channels + lc_enc.out_channels
fusion  = FusionAttentionUNet(in_ch=in_ch, base_ch=64, out_ch=1).to(torch_device)

ckpt = torch.load('checkpoints/best_uhi.pth', map_location=torch_device)
lst_enc.load_state_dict(ckpt['lst_enc'])
pm_enc.load_state_dict(ckpt['pm_enc'])
lc_enc.load_state_dict(ckpt['lc_enc'])
fusion.load_state_dict(ckpt['fusion'])
for m in (lst_enc, pm_enc, lc_enc, fusion): m.eval()

# ── PLOTTING & MODIFICATION ────────────────────────────────────────────────────
def make_plot(dataset, year, month, x0, x1, y0, y1, factor, use_kelvin=False):
    if dataset == 'pm25':
        arr = load_pm25(year, month)
        arr[y0:y1, x0:x1] *= factor
        save_pm25(arr, year, month)

        title = f"PM₂.₅ (µg/m³) – {month:02d}/{year} (×{factor:.1f} in box)"
        cmap = 'viridis'
        norm = None

    elif dataset == 'lst':
        arr = load_lst(year, month, use_kelvin)
        arr[y0:y1, x0:x1] *= factor
        save_lst(arr, year, month, use_kelvin)

        unit = 'K' if use_kelvin else '°C'
        title = f"LST ({unit}) – {month:02d}/{year} (×{factor:.1f} in box)"
        cmap  = LST_CMAP
        norm  = LST_NORM_K if use_kelvin else LST_NORM_C

    else:
        abort(400, f"Unknown dataset: {dataset}")

    # render to PNG buffer
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(arr, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def make_plot_esri(year:int):
    arr = load_esri(year)
    # (no area/factor modification for ESRI per your request)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(arr, cmap=ESRI_CMAP, norm=ESRI_NORM)
    ax.set_title(f"Landcover {year}")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Class Index')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def make_plot_uhii(year:int, month:int):
    arr = load_uhii(year, month)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(arr, cmap='inferno', vmin=UHII_VMIN, vmax=UHII_VMAX)
    ax.set_title(f"Combined Average UHII – {year}-{month:02d} (°C)")
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("UHII Intensity (°C)")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ── UHII PREDICTIONS PLOTTING ──────────────────────────────────────────────────
def make_plot_uhii_cached(year: int, month: int):
    arr = np.load(os.path.join(UHII_CACHE, f"uhii_{year}_{month:02d}.npy"))
    return _plot_array(arr, cmap='inferno', vmin=0.1, vmax=2.5,
                       title=f"UHII Observed – {year}-{month:02d}")

def make_plot_uhii_pred(year: int, month: int):
    pm25_arr = np.load(os.path.join(PM25_CACHE, f"pm25_{year}_{month:02d}.npy")).squeeze()
    lst_arr  = np.load(os.path.join(LST_CACHE, f"lst_{year}_{month:02d}.npy")).squeeze()
    lc_arr   = np.load(os.path.join(ESRI_CACHE, f"landcover_{year}.npy")).squeeze()

    # Ensure inputs are 2D
    pm25_arr = np.squeeze(pm25_arr)
    if pm25_arr.ndim != 2:
        raise ValueError(f"Expected 2D pm25 input, got shape {pm25_arr.shape}")
    
    # Now convert to tensors with shape (1, 1, H, W)
    pm25 = torch.from_numpy(pm25_arr).unsqueeze(0).unsqueeze(0).to(torch_device).float()
    lst  = torch.from_numpy(np.squeeze(lst_arr)).unsqueeze(0).unsqueeze(0).to(torch_device).float()
    lc   = torch.from_numpy(np.squeeze(lc_arr)).unsqueeze(0).unsqueeze(0).to(torch_device).float()

    # Encoding and fusion
    f_pm = pm_enc(pm25)
    f_ls = lst_enc(lst)
    f_lc = lc_enc(lc)
    if f_lc.shape[-2:] != f_pm.shape[-2:]:
        f_lc = F.interpolate(f_lc, size=f_pm.shape[-2:], mode='bilinear', align_corners=False)
    pred_low = fusion(torch.cat([f_pm, f_ls, f_lc], dim=1))
    pred = F.interpolate(pred_low, size=pm25.shape[-2:], mode='bilinear', align_corners=False)

    arr = pred.squeeze().cpu().detach().numpy()
    return _plot_array(arr, cmap='inferno', vmin=0.1, vmax=2.5,
                       title=f"UHII Predicted – {year}-{month:02d}")



def _plot_array(arr, cmap, vmin, vmax, title):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('UHII (°C)')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    payload = request.get_json()
    buf = make_plot(
        dataset   = payload['dataset'],
        year      = int(payload['year']),
        month     = int(payload['month']),
        x0        = int(payload['x0']),
        x1        = int(payload['x1']),
        y0        = int(payload['y0']),
        y1        = int(payload['y1']),
        factor    = float(payload['factor']),
        use_kelvin= bool(payload.get('use_kelvin', False))
    )
    return send_file(buf, mimetype='image/png')

@app.route('/plot_esri', methods=['POST'])
def plot_esri():
    data = request.get_json()
    year = int(data.get('year'))
    buf  = make_plot_esri(year)
    return send_file(buf, mimetype='image/png')

@app.route('/plot_uhii', methods=['POST'])
def plot_uhii():
    data = request.get_json()
    year  = int(data.get('year'))
    month = int(data.get('month'))
    buf   = make_plot_uhii(year, month)
    return send_file(buf, mimetype='image/png')

@app.route('/plot_uhii_pred', methods=['POST'])
def plot_uhii_pred():
    year = int(request.json['year']); month = int(request.json['month'])
    return send_file(make_plot_uhii_pred(year, month), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
