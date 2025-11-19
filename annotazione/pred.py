import pandas as pd
from collections import Counter
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Funzioni di utilità
# -------------------------------
def normalize_val(x):
	if pd.isna(x):
		return x
	x = str(x)
	x = unicodedata.normalize("NFKC", x)
	return x.strip().lower()

def normalize_columns(df, cols):
	for c in cols:
		df[c] = df[c].apply(normalize_val)
	return df

def filter_by_gold_and_meaning(df, gold, meaning_keywords):
	pattern = "|".join(meaning_keywords)
	return df[(df["GOLD"] == gold) & df["MEANING"].str.contains(pattern, case=False, regex=True)]

def compute_accuracy_and_errors(df, annotator):
	gold = df["GOLD"]
	preds = df[annotator]
	accuracy = (preds == gold).mean() * 100
	errors = preds[preds != gold]
	top3 = Counter(errors).most_common(3)
	return accuracy, top3

def compute_mean_human_performance(df, human_ann):
	accuracies = []
	all_errors = []
	for ann in human_ann:
		acc, _ = compute_accuracy_and_errors(df, ann)
		accuracies.append(acc)
		errors = df[ann][df[ann] != df["GOLD"]]
		all_errors.extend(errors)
	top3_hum = Counter(all_errors).most_common(3)
	return sum(accuracies)/len(accuracies), top3_hum

# -------------------------------
# MAIN
# -------------------------------
def main():
	path = "annotazione/gold_bert_ann.csv"
	df = pd.read_csv(path, sep=";", engine="python")

	annotators_all = ["GOLD","BERT","MARTYNA","FRANCESCO","FEDERICO","SARA","LARA"]
	annotators_hum = ["MARTYNA","FRANCESCO","FEDERICO","SARA","LARA"]

	df = normalize_columns(df, annotators_all)

	# Controllo NaN tra annotatori
	nan_rows = df[df[annotators_hum].isna().any(axis=1)]
	if len(nan_rows) > 0:
		print("⚠ Righe con almeno un NaN tra annotatori:")
		print(nan_rows)
		for ann in annotators_hum:
			n = nan_rows[ann].isna().sum()
			if n > 0:
				print(f"{ann} ha {n} valori NaN")

	# Gruppi di interesse
	groups = {
		"A / Juxtaposition":  ("a", ["juxtaposition","contact"]),
		"A / Succession":     ("a", ["succession","iteration","distributivity"]),
		"SU / GreaterAccum":  ("su", ["greater accumulation","accumul"]),
		"SU / Succession":    ("su", ["succession"])
	}

	# Tutti gli errori possibili (per heatmap)
	all_errors = set()
	for gname, (gold, patterns) in groups.items():
		subset = filter_by_gold_and_meaning(df, gold, patterns)
		if len(subset) == 0: continue
		all_errors.update(subset["BERT"][subset["BERT"]!=subset["GOLD"]].unique())
		for ann in annotators_hum:
			all_errors.update(subset[ann][subset[ann]!=subset["GOLD"]].unique())
	all_errors = sorted(all_errors)

	# Dati per grafici
	boxplot_data = {gname: [] for gname in groups}
	bert_errors_dict = {}
	human_errors_dict = {}
	stacked_bar_data = {gname: {} for gname in groups}
	bert_acc_list = []
	human_mean_list = []

	for gname, (gold, patterns) in groups.items():
		subset = filter_by_gold_and_meaning(df, gold, patterns)
		if len(subset) == 0: continue

		# Boxplot dati annotatori
		human_accuracies = [(subset[ann]==subset["GOLD"]).mean()*100 for ann in annotators_hum]
		boxplot_data[gname] = human_accuracies
		human_mean = sum(human_accuracies)/len(human_accuracies)
		human_mean_list.append(human_mean)

		# Accuracy BERT
		bert_acc = (subset["BERT"]==subset["GOLD"]).mean()*100
		bert_acc_list.append(bert_acc)

		# Heatmap errori
		bert_counts = Counter(subset["BERT"][subset["BERT"]!=subset["GOLD"]])
		bert_errors_dict[gname] = [bert_counts.get(e,0) for e in all_errors]

		human_counts_total = Counter()
		for ann in annotators_hum:
			hum_errs = subset[ann][subset[ann]!=subset["GOLD"]]
			human_counts_total.update(hum_errs)
		human_errors_dict[gname] = [human_counts_total.get(e,0) for e in all_errors]

		# Stacked bar dati corrette/errate
		for ann in annotators_all:
			correct = (subset[ann]==subset["GOLD"]).sum()
			error = (subset[ann]!=subset["GOLD"]).sum()
			stacked_bar_data[gname][ann] = {"correct": correct, "error": error}

	# -------------------------------
	# Boxplot variabilità annotatori con BERT e media sovrapposti
	# -------------------------------
	plt.figure(figsize=(10,5))
	data_to_plot = [boxplot_data[gname] for gname in groups]
	plt.boxplot(data_to_plot, labels=groups.keys())
	plt.scatter(range(1,len(groups)+1), bert_acc_list, color="red", zorder=5, label="BERT")
	plt.scatter(range(1,len(groups)+1), human_mean_list, color="green", marker='D', s=60, zorder=5, label="Media annotatori")
	plt.ylabel("Accuracy (%)")
	plt.title("Variabilità annotatori con BERT e media annotatori sovrapposti")
	plt.legend()
	plt.show()

	# -------------------------------
	# Stacked bar corrette vs errori
	# -------------------------------
	plt.figure(figsize=(12,5))
	for i, gname in enumerate(groups):
		ann_names = list(stacked_bar_data[gname].keys())
		correct_vals = [stacked_bar_data[gname][ann]["correct"] for ann in ann_names]
		error_vals = [stacked_bar_data[gname][ann]["error"] for ann in ann_names]
		plt.bar([x+i*0.15 for x in range(len(ann_names))], correct_vals, width=0.15, label=f"{gname} corrette")
		plt.bar([x+i*0.15 for x in range(len(ann_names))], error_vals, bottom=correct_vals, width=0.15, label=f"{gname} errori")
	plt.xticks([r + 0.2 for r in range(len(annotators_all))], annotators_all)
	plt.ylabel("Numero esempi")
	plt.title("Predizioni corrette per categoria")
	plt.legend()
	plt.show()

	# -------------------------------
	# Heatmap errori
	# -------------------------------
	df_bert = pd.DataFrame(bert_errors_dict, index=all_errors).T
	df_human = pd.DataFrame(human_errors_dict, index=all_errors).T

	plt.figure(figsize=(15,6))
	plt.subplot(1,2,1)
	sns.heatmap(df_bert, annot=True, fmt="g", cmap="Reds")
	plt.title("Errori BERT")
	plt.ylabel("Categoria")
	plt.xlabel("Errore")
	plt.subplot(1,2,2)
	sns.heatmap(df_human, annot=True, fmt="g", cmap="Blues")
	plt.title("Errori Annotatori")
	plt.ylabel("")
	plt.xlabel("Errore")
	plt.tight_layout()
	plt.show()

	# -------------------------------
	# Bar plot confronto BERT vs media annotatori
		# -------------------------------
	plt.figure(figsize=(8,5))
	x = range(len(groups))
	width = 0.35
	plt.bar([i - width/2 for i in x], bert_acc_list, width=width, color='red', label='BERT')
	plt.bar([i + width/2 for i in x], human_mean_list, width=width, color='green', label='Media Annotatori')
	plt.xticks(x, groups.keys())
	plt.ylabel("Accuracy (%)")
	plt.title("Confronto BERT vs Media annotatori (percentuale)")

	# Etichette percentuali sopra le barre
	for i in x:
		plt.text(i - width/2, bert_acc_list[i]+1, f"{bert_acc_list[i]:.1f}%", ha='center', va='bottom')
		plt.text(i + width/2, human_mean_list[i]+1, f"{human_mean_list[i]:.1f}%", ha='center', va='bottom')

	plt.ylim(0, 110)  # lascia spazio per le etichette
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()
