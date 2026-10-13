import os
import subprocess
import xarray as xr

# -----------------------------
# Dossier de destination local
# -----------------------------
DEST = r"C:\Users\fakponi\Desktop\Downscaling\data\ALADIN63\daily"
os.makedirs(DEST, exist_ok=True)

# -----------------------------
# Variables journalières à télécharger
# -----------------------------
VARIABLES = [
    "tasAdjust", "hussAdjust", "hursAdjust",
    "prAdjust", "rsdsAdjust", "sfcWindAdjust",
    "tasmaxAdjust", "tasminAdjust"
]

# -----------------------------
# Chemin base dans le bucket datagouv
# -----------------------------
BASE_BUCKET = "datagouv:meteofrance-drias/SocleOM-Climat-2025/SocleOM-FrReunion-Climat-2025-DBA/CNRM-ESM2/ALADIN63/historical/day"

# -----------------------------
# Fonction pour télécharger une variable avec subprocess
# -----------------------------
def download_variable(var_name):
    source = f"{BASE_BUCKET}/{var_name}"
    dest = os.path.join(DEST, var_name)
    os.makedirs(dest, exist_ok=True)
    print(f"\nTéléchargement de la variable : {var_name}")
    
    # Appel rclone via subprocess
    cmd = ["rclone", "copy", "-P", source, dest]
    subprocess.run(cmd, check=True)

# -----------------------------
# Télécharger toutes les variables
# -----------------------------
for var in VARIABLES:
    download_variable(var)
