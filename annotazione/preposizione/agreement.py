import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
from itertools import combinations
# For Fleiss’ kappa:
from statsmodels.stats.inter_rater import fleiss_kappa

# --- Load & normalize ---
df = pd.read_csv(sys.argv[1], sep=";")  # columns: item_id, G, H1..H5, A

def normalize(s):
    if pd.isna(s): return s
    return s.strip().lower()
for col in ["G","H1","H2","H3","H4","H5","A"]:
    df[col] = df[col].map(normalize)

# --- Pairwise Cohen’s κ helper (drops rows with NA in either column) ---
def pairwise_kappa(s1, s2):
    mask = s1.notna() & s2.notna()
    if mask.sum() == 0: return float("nan")
    return cohen_kappa_score(s1[mask], s2[mask])

# Human–human pairwise κ matrix
humans = ["H1","H2","H3","H4","H5"]
pairs = list(combinations(humans, 2))
hh_kappas = {f"{a}-{b}": pairwise_kappa(df[a], df[b]) for a,b in pairs}
hh_mean_kappa = pd.Series(hh_kappas).mean()

# Human–artificial κ and raw %
ha = {h: pairwise_kappa(df[h], df["A"]) for h in humans}
ha_raw = {h: accuracy_score(df[h].dropna().align(df["A"].dropna(), join="inner")[0],
                            df["A"].dropna().align(df[h].dropna(), join="inner")[0]) for h in humans}
ha_mean_kappa = pd.Series(ha).mean()

# To gold: accuracy and κ
hg_acc = {h: accuracy_score(df["G"].dropna().align(df[h].dropna(), join="inner")[0],
                            df[h].dropna().align(df["G"].dropna(), join="inner")[0]) for h in humans}
hg_kappa = {h: pairwise_kappa(df[h], df["G"]) for h in humans}
ag_acc = accuracy_score(df["G"].dropna().align(df["A"].dropna(), join="inner")[0],
                        df["A"].dropna().align(df["G"].dropna(), join="inner")[0])
ag_kappa = pairwise_kappa(df["A"], df["G"])

# Fleiss’ κ across 5 humans (requires complete 5 ratings per item):
# Build per-item counts over observed categories.
complete = df.dropna(subset=humans)
# Map each unique label among humans to an index
cats = {lab:i for i, lab in enumerate(pd.unique(complete[humans].values.ravel()))}
M = []
for _, row in complete[humans].iterrows():
    counts = [0]*len(cats)
    for lab in row:
        counts[cats[lab]] += 1
    M.append(counts)
fleiss = fleiss_kappa(pd.DataFrame(M), method='fleiss')

print("Human–human mean pairwise κ:", hh_mean_kappa)
print("Human–human pairwise κ:", hh_kappas)
print("Fleiss’ κ (5 humans):", fleiss)
print("Human–A κ:", ha, " (mean:", ha_mean_kappa, ")")
print("Human–A raw %:", ha_raw)
print("Human–G accuracy:", hg_acc)
print("Human–G κ:", hg_kappa)
print("A–G accuracy:", ag_acc, "  A–G κ:", ag_kappa)
