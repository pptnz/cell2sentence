#!/usr/bin/env python
# coding: utf-8

# # Tutorial notebook 4: Cell Type Prediction
# 
# In this tutorial, we will demonstrate how to use a pretrained Cell2Sentence (C2S) model to perform cell type prediction on single-cell RNA sequencing datasets. Cell type prediction is a crucial step in single-cell analysis, allowing researchers to identify and classify different cell populations within a dataset. By leveraging the power of C2S models, we can make accurate predictions based on the information encoded in cell sentences.
# 
# In this tutorial, you will:
# 1. Load an immune tissue single-cell dataset from Domínguez Conde et al. (preprocessed in tutorial notebook 0, two sample donors)
#     - Citation: Domínguez Conde, C., et al. "Cross-tissue immune cell analysis reveals tissue-specific features in humans." Science 376.6594 (2022): eabl5197.
# 2. Load a pretrained C2S model that is capable of making cell type predictions.
# 3. Use the model to predict cell types based on the cell sentences derived from the dataset.

# We will begin by importing the necessary libraries. These include Python's built-in libraries, third-party libraries for handling numerical computations, progress tracking, and specific libraries for single-cell RNA sequencing data and C2S operations.

# In[1]:


# Python built-in libraries
import argparse
import os
import pickle
import random
from collections import Counter

# Third-party libraries
import numpy as np
from tqdm import tqdm

# Single-cell libraries
import anndata
import scanpy as sc

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument("--cell_type_prediction_model_path", type=str, default=None)
args, _ = parser.parse_known_args()

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)


# # Load Data
# 
# Next, we will load the preprocessed dataset from the tutorial 0. This dataset has already been filtered and normalized, so it it ready for transformation into cell sentences.
# 
# <font color='red'>Please make sure you have completed the preprocessing steps in Tutorial 0 before running the following code, if you are using your own dataset.</font>. Ensure that the file path is correctly set in <font color='gold'>DATA_PATH</font> to where your preprocessed data was saved from tutorial 0.

# In[ ]:


# DATA_PATH = "/oak/stanford/groups/amnewman/mkang9/0_Code/cytotrace2_training_pipeline/cytotrace2_test_gdrive.h5ad"
DATA_PATH = "./cytotrace2_test.h5ad"


# In[ ]:


adata = anndata.read_h5ad(DATA_PATH)

adata.X = adata.X.toarray()
adata.X[np.isnan(adata.X)] = 0
print(adata)


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


# In[5]:


adata.obs = adata.obs[["cell_type", "tissue", "batch_condition", "organism", "sex"]]


# In[6]:


adata.obs.head()


# In[7]:


adata.var.head()


# In[8]:


sc.pl.umap(
    adata,
    color="cell_type",
    size=8,
    title="Human Immune Tissue UMAP",
)


# In[9]:


adata.X.max()


# We are expecting log10 base 10 transformed data, with a maximum value somewhere around 3 or 4. Make sure to start with processed and normalized data when doing the cell sentence conversion!

# # Cell2Sentence Conversion
# 
# In this section, we will transform our AnnData object containing our single-cell dataset into a Cell2Sentence (C2S) dataset by calling the functions of the CSData class in the C2S code base. Full documentation for the functions of the CSData class can be found in the documentation page of C2S.

# In[10]:


adata_obs_cols_to_keep = ["cell_type", "tissue", "batch_condition", "organism", "sex"]


# In[11]:


# Create CSData object
arrow_ds, vocabulary = cs.CSData.adata_to_arrow(
    adata=adata, 
    random_state=SEED, 
    sentence_delimiter=' ',
    label_col_names=adata_obs_cols_to_keep
)


# In[12]:


arrow_ds


# In[13]:


sample_idx = 0
arrow_ds[sample_idx]


# This time, we will leave off creating our CSData object until after we load our C2S model. This is because along with the model checkpoint, we saved the indices of train, val, and test set cells, which will allow us to select out test set cells for inference.

# # Load C2S Model
# 
# Now, we will load a C2S model with which we will do cell type annotation. For this tutorial, this model will be the last checkpoint of the training session from <font color="red">tutorial notebook 3</font>, where we finetuned our cell type prediction model to do cell type prediction specifically on our immune tissue dataset. We will load the last checkpoint saved from training, and specify the same save_dir as we used before during training.
# - <font color="red">Note:</font> If you are using your own data for this tutorial, make sure to switch out to the model checkpoint which you saved in tutorial notebook 3.
# - If you want to annotate cell types without finetuning your own C2S model, then tutorial notebook 6 demonstrates how to load the C2S-Pythia-410M cell type prediction foundation model and use it to predict cell types without any finetuning.
# 
# We can define our CSModel object with our pretrained cell type prediction model as follows, specifying the same save_dir as we used in tutorial 3:

# In[ ]:


# Define CSModel object
import glob
save_dir = "./c2s_api_testing/csmodel_tutorial_3"

if args.cell_type_prediction_model_path is not None:
    cell_type_prediction_model_path = args.cell_type_prediction_model_path
else:
    # Find the finetune_cell_type_prediction directory and latest checkpoint automatically
    finetune_dirs = sorted(glob.glob(os.path.join(save_dir, "*_finetune_cell_type_prediction")))
    assert len(finetune_dirs) > 0, f"No finetune_cell_type_prediction directory found in {save_dir}"
    finetune_dir = finetune_dirs[-1]

    checkpoint_dirs = sorted(
        glob.glob(os.path.join(finetune_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )
    assert len(checkpoint_dirs) > 0, f"No checkpoint-* directories found in {finetune_dir}"
    cell_type_prediction_model_path = checkpoint_dirs[-1]
print(f"Using checkpoint: {cell_type_prediction_model_path}")

save_name = "cell_type_pred_pythia_410M_inference"
csmodel = cs.CSModel(
    model_name_or_path=cell_type_prediction_model_path,
    save_dir=save_dir,
    save_name=save_name
)


# We will also load the data split indices saved alongside the C2S model checkpoint, so that we know which cells were part of the training and validation set. We will do inference on unseen test set cells, which are 10% of the original data.

# In[15]:


base_path = "/".join(cell_type_prediction_model_path.split("/")[:-1])
print(cell_type_prediction_model_path)
print(base_path)


# In[ ]:


# with open(os.path.join(base_path, 'data_split_indices_dict.pkl'), 'rb') as f:
#     data_split_indices_dict = pickle.load(f)
# data_split_indices_dict.keys()


# In[ ]:


# print(len(data_split_indices_dict["train"]))
# print(len(data_split_indices_dict["val"]))
# print(len(data_split_indices_dict["test"]))


# Select out test set cells from full arrow dataset

# In[18]:


arrow_ds


# In[ ]:


# test_ds = arrow_ds.select(data_split_indices_dict["test"])
# test_ds


# In[ ]:


test_ds = arrow_ds


# Now, we will create our CSData object using only the test set cells:

# In[ ]:


c2s_save_dir = "./c2s_api_testing"  # C2S dataset will be saved into this directory
c2s_save_name = "cytotrace2_test"  # This will be the name of our C2S dataset on disk


# In[21]:


csdata = cs.CSData.csdata_from_arrow(
    arrow_dataset=test_ds, 
    vocabulary=vocabulary,
    save_dir=c2s_save_dir,
    save_name=c2s_save_name,
    dataset_backend="arrow"
)


# In[22]:


print(csdata)


# # Predict cell types
# 
# Now that we have loaded our finetuned cell type prediction model and have our test set, we will do cell type prediction inference using our C2S model. We can use the function predict_cell_types_of_data() from the tasks.py, which will take a CSModel() object and apply it to do cell type prediction on a CSData() object.

# In[23]:


predicted_cell_types = predict_cell_types_of_data(
    csdata=csdata,
    csmodel=csmodel,
    n_genes=200
)


# In[24]:


len(predicted_cell_types)


# In[ ]:


predicted_cell_types[:3]

with open(cell_type_prediction_model_path + "/predicted_cell_types.pkl", "wb") as f:
    pickle.dump(predicted_cell_types, f)


# In[26]:


test_ds


# In[27]:


total_correct = 0.0
for model_pred, gt_label in zip(predicted_cell_types, test_ds["cell_type"]):
    # C2S might predict a period at the end of the cell type, which we remove
    if model_pred[-1] == ".":
        model_pred = model_pred[:-1]
    
    if model_pred == gt_label:
        total_correct += 1

accuracy = total_correct / len(predicted_cell_types)


# In[28]:


print("Accuracy:", accuracy)


# In[32]:


for idx in range(0, 100, 10):
    print("Model pred: {}, GT label: {}".format(predicted_cell_types[idx], test_ds[idx]["cell_type"]))


# We can see that our model achieves high accuracy, correctly predicting the cell type of unseen cells from the immune tissue data 83.4% of the time! The model learned to predict cell type annotations in natural language effectively from a short finetuning period on the new data.
