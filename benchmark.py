import pickle
import anndata
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

sns.set(font='sans-serif', font_scale=1)
sns.set_palette("Paired")
sns.set_style('ticks')


# Find the finetune_cell_type_prediction directory and latest checkpoint automatically
import glob
save_dir = "./c2s_api_testing/csmodel_tutorial_3"

finetune_dirs = sorted(glob.glob(os.path.join(save_dir, "*_finetune_cell_type_prediction")))
assert len(finetune_dirs) > 0, f"No finetune_cell_type_prediction directory found in {save_dir}"
finetune_dir = finetune_dirs[-1]

checkpoint_dirs = sorted(
    glob.glob(os.path.join(finetune_dir, "checkpoint-*")),
    key=lambda x: int(x.split("-")[-1])
)
assert len(checkpoint_dirs) > 0, f"No checkpoint-* directories found in {finetune_dir}"

result_dir = checkpoint_dirs[-1]
print(f"Using checkpoint: {result_dir}")

def f1_dataset(x):
   labels = ['Differentiated','Unipotent','Oligopotent','Multipotent','Pluripotent','Totipotent']
   all1 = precision_recall_fscore_support(x['potency'],x['predicted_potency'], average=None, labels=labels, zero_division=0)
   f1 = pd.Series(all1[2])
   f1.index = labels
   supp = pd.Series(all1[3])
   supp.index = labels
   f1[supp==0] = np.nan
   return f1

def weighted_accuracy(x):
    return x.groupby(['Dataset', 'phenotype']).apply(f1_dataset).reset_index().groupby(['Dataset'])[
      ['Differentiated','Unipotent','Oligopotent','Multipotent','Pluripotent','Totipotent']].mean().mean(0).mean()


# mean multiclass F1 score
def mean_multiclass_f1(x):
    potency_labels = ['Totipotent', 'Pluripotent', 'Multipotent', 'Oligopotent', 'Unipotent', 'Differentiated']
    mc_f1 = f1_score(x['potency'].values,x['predicted_potency'].values,labels=potency_labels,average=None)
    mean = mc_f1.mean()
    return mean


benchmark_df = pd.read_csv("4_folds_F1_cscc.csv")
benchmark_df.columns = ['Metric', "Logistic reg.", "XGBoost", "Linear SVM", "Radial SVM", "Multinomial logistic reg.", "CytoTRACE 2", "SingleCellNet", "scmap", "scPred", "Fold"]

benchmark_df = benchmark_df.loc[benchmark_df['Fold'] == 'OriginalTest']
benchmark_df = benchmark_df.loc[benchmark_df['Metric'] == 'Macro_F1'] 

benchmark_df



# cell2sentence
with open(os.path.join(result_dir, "predicted_cell_types.pkl"), "rb") as f:
    predicted_cell_types = pickle.load(f)

predicted_cell_types = [cell_type.split(".")[0] for cell_type in predicted_cell_types]
test = anndata.read_h5ad("cytotrace2_test.h5ad")
test.obs["predicted_potency"] = predicted_cell_types


f1 = mean_multiclass_f1(test.obs)
print('mean multiclass F1 score: ', f1)

benchmark_df['C2S'] = f1

df = benchmark_df.melt(id_vars=['Metric'], value_vars=['Logistic reg.', 'XGBoost', 'Linear SVM', 'Radial SVM', 'Multinomial logistic reg.', 'CytoTRACE 2', 'SingleCellNet', 'scmap', 'scPred', 'C2S'])
df = df.sort_values(by='value', ascending=False)
# bar plot
plt.figure(figsize=(5, 4))
sns.barplot(x='variable', y='value', data=df)
plt.xlabel('Method')
plt.ylabel('F1 score')
plt.title('F1 score comparison')
# rotate x-axis labels
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "f1_score_comparison.png"))