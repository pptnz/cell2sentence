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

with open(os.path.join(result_dir, "predicted_cell_types.pkl"), "rb") as f:
    predicted_cell_types = pickle.load(f)

predicted_cell_types = [cell_type.split(".")[0] for cell_type in predicted_cell_types]
test = anndata.read_h5ad("cytotrace2_test.h5ad")

# confusion matrix 
test.obs["predicted_potency"] = predicted_cell_types

confusion_matrix = pd.crosstab(test.obs["potency"], test.obs["predicted_potency"])
categories = ['Totipotent', 'Pluripotent', 'Multipotent', 'Oligopotent', 'Unipotent', 'Differentiated']
confusion_matrix = confusion_matrix.loc[categories, categories]

# plot as heatmap
# color: row-normalized
# annot: raw counts
confusion_matrix_normalized = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)
plt.clf()
sns.heatmap(confusion_matrix_normalized, annot=confusion_matrix.values, fmt='d')
plt.xlabel("Predicted potency")
plt.ylabel("Ground truth potency")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "confusion_matrix.png"))


weighted_accuracy = \
test.obs.groupby(['Dataset', 'phenotype']).apply(f1_dataset).reset_index().groupby(['Dataset'])[
    ['Differentiated', 'Unipotent', 'Oligopotent', 'Multipotent', 'Pluripotent', 'Totipotent']].mean().mean(0).mean()
print('weighted accuracy: ', weighted_accuracy)
# 0.54 -> 0.58

f1 = mean_multiclass_f1(test.obs)
print('mean multiclass F1 score: ', f1)

cytotrace2_pred = pd.read_csv("TrainTestTS_240706_MK_split_100_v4k_obs_benchmarking_filtered_5.csv", index_col=0, low_memory=False)
cytotrace2_pred.columns
cytotrace2_pred['predicted_potency'] = np.round(5.5 - 6 * cytotrace2_pred['CytoTRACE.2']).map(dict(zip([0, 1, 2, 3, 4, 5], ['Totipotent', 'Pluripotent', 'Multipotent', 'Oligopotent', 'Unipotent', 'Differentiated'])))

test_cytotrace2 = cytotrace2_pred[cytotrace2_pred['Cohort'] == 'Test 1']

weighted_accuracy = \
test_cytotrace2.groupby(['Dataset', 'phenotype']).apply(f1_dataset).reset_index().groupby(['Dataset'])[
    ['Differentiated', 'Unipotent', 'Oligopotent', 'Multipotent', 'Pluripotent', 'Totipotent']].mean().mean(0).mean()
print('weighted accuracy: ', weighted_accuracy)
# 0.76

f1 = mean_multiclass_f1(test_cytotrace2)
print('mean multiclass F1 score: ', f1)


# read trainer state json and plot loss_curve.png and lr_curve.png
with open(os.path.join(result_dir, "trainer_state.json"), "r") as f:
    trainer_state = json.load(f)

log_history = trainer_state['log_history']

# Extract training loss and learning rate (entries that have 'loss' key)
train_steps = [entry['step'] for entry in log_history if 'loss' in entry]
train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
lr_values = [entry['learning_rate'] for entry in log_history if 'learning_rate' in entry]
lr_steps = [entry['step'] for entry in log_history if 'learning_rate' in entry]

# Extract eval loss
eval_steps = [entry['step'] for entry in log_history if 'eval_loss' in entry]
eval_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]

# Plot training & eval loss curve
plt.figure(figsize=(5, 4))
plt.plot(train_steps, train_loss, label='Train Loss', alpha=0.8, color='tab:blue')
plt.plot(eval_steps, eval_loss, label='Validation Loss', alpha=0.8, color='tab:orange')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "loss_curve.png"))

# Plot learning rate curve
plt.figure(figsize=(5, 4))
plt.plot(lr_steps, lr_values, label='Learning Rate', color='tab:green')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "lr_curve.png"))