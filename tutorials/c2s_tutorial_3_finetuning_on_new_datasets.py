#!/usr/bin/env python
# coding: utf-8

# # Tutorial 3: Finetuning on a New Single-Cell Dataset
# 
# In this tutorial, we will demonstrate how to fine-tune an existing Cell2Sentence (C2S) model on a new single-cell RNA sequencing dataset. Fine-tuning is a crucial step in adapting a pretrained model to perform well on a specific task or dataset, improving its accuracy and generalization. This tutorial will guide you through the process of fine-tuning a C2S model to perform cell type prediction on a new dataset.
# 
# In this tutorial, you will:
# 1. Load an immune tissue single-cell dataset from Domínguez Conde et al. (preprocessed in tutorial notebook 0, two sample donors)
#     - Citation: Domínguez Conde, C., et al. "Cross-tissue immune cell analysis reveals tissue-specific features in humans." Science 376.6594 (2022): eabl5197.
# 2. Format the dataset using a Prompt Formatter object, which prepares the data for the fine-tuning process.
# 3. Load a pretrained C2S model.
# 4. Fine-tune the C2S model to improve its performance on cell type prediction.

# We will begin by importing the necessary libraries. These include Python's built-in libraries, third-party libraries for handling numerical computations, progress tracking, and specific libraries for single-cell RNA sequencing data and C2S operations.

# In[1]:


# Python built-in libraries
import os
from datetime import datetime
import random
from collections import Counter

# Third-party libraries
import numpy as np
from tqdm import tqdm
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_from_disk

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs


# In[2]:


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)


# # Load Data
# 
# Next, we will load the preprocessed dataset from the tutorial 0. This dataset has already been filtered and normalized, so it it ready for transformation into cell sentences.
# 
# <font color='red'>Please make sure you have completed the preprocessing steps in Tutorial 0 before running the following code, if you are using your own dataset.</font>. Ensure that the file path is correctly set in <font color='gold'>DATA_PATH</font> to where your preprocessed data was saved from tutorial 0.

# In[ ]:


DATA_PATH = "/oak/stanford/groups/amnewman/mkang9/0_Code/cytotrace2_training_pipeline/cytotrace2_training_gdrive.h5ad"


# In[ ]:


adata = anndata.read_h5ad(DATA_PATH)
adata


# In[ ]:


# fill na with 0
adata.X[np.isnan(adata.X)] = 0
# Count normalization
sc.pp.normalize_total(adata)
# Lop1p transformation with base 10 - base 10 is important for C2S transformation!
sc.pp.log1p(adata, base=10)  
print(adata.X.max())


# In[ ]:


sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)


# In[ ]:


adata.obs = adata.obs[['Species', 'Tissue', 'Dataset', 'DatasetDetail', 'phenotype', 'potency']]
adata.obs['organism'] = adata.obs['Species'].map({'human': 'Homo sapiens', 'mouse': 'Mus musculus'})
adata.obs['sex'] = 'NA'
adata.obs['tissue'] = adata.obs['Tissue']
adata.obs['batch_condition'] = adata.obs['Dataset']
adata.obs['cell_type'] = adata.obs['potency']


# In[6]:


adata.obs.head()


# In[7]:


adata.var.head()


# In[ ]:


sc.pl.umap(
    adata,
    color="phenotype",
    size=8,
    title="Human Immune Tissue UMAP",
)


# In[9]:


adata.X.max()


# We are expecting log10 base 10 transformed data, with a maximum value somewhere around 3 or 4. Make sure to start with processed and normalized data when doing the cell sentence conversion!

# # Cell2Sentence Conversion + CSData Creation
# 
# In this section, we will transform our AnnData object containing our single-cell dataset into a Cell2Sentence (C2S) dataset by calling the functions of the CSData class in the C2S code base. Full documentation for the functions of the CSData class can be found in the documentation page of C2S.

# In[ ]:


adata_obs_cols_to_keep = ['cell_type', 'tissue', 'batch_condition', 'organism', 'sex']


# In[11]:


# Create CSData object
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=SEED, 
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)


# In[12]:


sample_idx = 0
arrow_ds[sample_idx]


# In[ ]:


c2s_save_dir = "./c2s_api_testing"  # C2S dataset will be saved into this directory
c2s_save_name = "ct2_potency_prediction"  # This will be the name of our C2S dataset on disk


# In[14]:


csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=arrow_ds, 
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)


# In[15]:


print(csdata)


# # Load C2S Model
# 
# Now, we will load a C2S model which will finetune on a new dataset. This model can be a LLM pretrained on natural language, or it can be a trained C2S model which will undergo further finetuning on a new dataset of interest. Typically, starting from a pretrained C2S model benefits performance, since C2S models were initialized from natural language-pretrained LLMs and trained on many single-cell datasets on different tasks.
# 
# For this tutorial, we will start finetuning from the C2S-Pythia-410M cell type prediction model, which was trained to do cell type prediction on many datasets from CellxGene and Human Cell Atlas. We will finetune it for cell type prediction on our immune tissue dataset which we have loaded, which will help align the model with the cell type annotations present in this dataset as well as the expression profiles of the cells in our two donor samples. More details about the C2S-Pythia-410M cell type prediction model can be found in the Model Zoo section of the ReadME in the GitHub repo, or in the Huggingface model card.
# 
# We can define our CSModel object with our pretrained cell type prediction model as follows:

# In[ ]:


# Define CSModel object
cell_type_prediction_model_path = "vandijklab/C2S-Pythia-410m-cell-type-prediction"
save_dir = "./c2s_api_testing/csmodel_tutorial_3"
save_name = "cell_type_pred_pythia_410M_2"
csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)


# Note that the `model_name_or_path` parameter can be a name of a Huggingface model, for example 'EleutherAI/pythia-410m' for a 410 million parameter Pythia model pretrained on natural language (see https://huggingface.co/EleutherAI/pythia-410m), or it can be the path to a pretrained model saved on disk, as in the case in the cell above.

# In[17]:


print(csmodel)


# # Finetune on new dataset
# 
# Now, we will finetune our loaded C2S model on our immune tissue dataset. For training, we will need to define training arguments for finetuning our C2S model on our new dataset. Huggingface's Trainer class is used to do training, so we can utilize different training techniques (e.g. mixed precision training, gradient accumulation, gradient checkpointing, etc.) by specifying the corresponding option in the TrainingArguments object. This gives us a vast array of possible options for training, and will allow us to specify important parameters such as batch size, learning rate, and learning rate schedulers. See the full documentation for training arguments at:
# - https://huggingface.co/docs/transformers/en/main_classes/trainer

# First, we define our training task, which in our case will be cell type prediction. Possible values for the training task parameter can be found in the `prompt_formatter.py` file in the source code, under `SUPPORTED_TASKS`.

# In[ ]:


training_task = "cell_type_prediction"


# We will create a datetimestamp to mark our training session:

# In[19]:


datetimestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
output_dir = os.path.join(csmodel.save_dir, datetimestamp + f"_finetune_{training_task}")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
print(output_dir)


# And here, we define our training arguments. For this tutorial, we will use a batch size of 8 with 4 gradient accumulation steps, yielding an effective batch size of 32. We will use a learning rate of 1e-5 with a cosine annealing scheduler, and we will train for 5 epochs total. Some other important parameters specified here are:
# - bf16: Uses mixed-precision training with bfloat16 dtype
# - logging_steps: controls how often we log training loss
# - eval_steps: controls how often we run the eval loop
# - warmup_ratio: percentage of training in which learning rate warms up to the base learning rate specified
# 
# Full explanations of all possible training arguments can be found in the Huggingface Trainer documentation: 
# 
# https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/trainer#transformers.TrainingArguments

# In[ ]:


train_args = TrainingArguments(
    bf16=True,
    fp16=False,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=False,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    logging_steps=50,
    logging_strategy="steps",
    lr_scheduler_type="cosine",
    num_train_epochs=20, 
    eval_steps=50,
    evaluation_strategy="steps",
    save_steps=100,
    save_strategy="steps",
    save_total_limit=3,
    warmup_ratio=0.05,
    output_dir=output_dir
)


# # Balanced Sampling to Handle Class Imbalance
# 
# The 6 potency categories have severe class imbalance (e.g. Differentiated ~58K vs
# Totipotent ~45). We create a balanced dataset by oversampling minority classes and
# undersampling majority classes, targeting a fixed number of samples per class.

# In[ ]:


# Load arrow dataset to access cell_type (potency) labels
hf_ds = load_from_disk(csdata.data_path)
labels = hf_ds["cell_type"]

# Print original class distribution
label_counts = Counter(labels)
print("Original class distribution:")
for cls, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count}")

# Target samples per class: balance via oversample (minority) + undersample (majority)
# Cap oversample ratio to avoid overfitting on extremely rare classes (e.g. Totipotent=43)
TARGET_PER_CLASS = 4000
MAX_OVERSAMPLE_RATIO = 20  # no class will be oversampled more than 20x its original size

balanced_indices = []
for label in label_counts:
    class_indices = [i for i, l in enumerate(labels) if l == label]
    n = len(class_indices)
    # Effective target: min of TARGET_PER_CLASS and MAX_OVERSAMPLE_RATIO * original count
    effective_target = min(TARGET_PER_CLASS, max(n, n * MAX_OVERSAMPLE_RATIO))
    if n >= effective_target:
        # Undersample: randomly pick effective_target without replacement
        sampled = np.random.choice(class_indices, size=effective_target, replace=False).tolist()
    else:
        # Oversample: sample with replacement up to effective_target
        sampled = np.random.choice(class_indices, size=effective_target, replace=True).tolist()
    balanced_indices.extend(sampled)

np.random.shuffle(balanced_indices)

# Stratified train/val split to ensure each split preserves class balance
balanced_labels = [labels[i] for i in balanced_indices]
train_idx, val_idx = train_test_split(
    balanced_indices, test_size=0.1, stratify=balanced_labels, random_state=SEED
)
train_idx.sort()
val_idx.sort()

data_split_indices_dict = {"train": train_idx, "val": val_idx}

print(f"\nBalanced dataset: {len(balanced_indices)} total "
      f"({len(train_idx)} train, {len(val_idx)} val)")
print("Balanced class distribution:")
balanced_label_counts = Counter(balanced_labels)
for cls, count in sorted(balanced_label_counts.items(), key=lambda x: -x[1]):
    print(f"  {cls}: {count}")


# In[21]:


csmodel.fine_tune(
    csdata=csdata,
    task=training_task,
    train_args=train_args,
    loss_on_response_only=False,
    top_k_genes=200,
    max_eval_samples=500,
    data_split_indices_dict=data_split_indices_dict,
)


# Our trained models are now saved in the output directory we specified in the training arguments. Huggingface will save the latest checkpoints of the training session, and will also keep the checkpoint which has the lowest validation loss.
# 
# In the next tutorial notebook (tutorial 4), we will see how to run cell type prediction inference with our trained model.
