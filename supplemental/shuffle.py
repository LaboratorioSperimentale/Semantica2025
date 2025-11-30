import pandas as pd
import sys

# Prendi il file da terminale
input_file = sys.argv[1]
output_file = sys.argv[2]

# Leggi il file
df = pd.read_excel(input_file)

# Shuffle righe
df_shuffled = df.sample(frac=1, random_state=42)

# Salva il file mescolato
df_shuffled.to_excel(output_file, index=False)
print(f"File salvato come {output_file}")
