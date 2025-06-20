# Tahoe100M_practice

**Anticancer Drug Acctivity Prediction by Biological and Chemical Features**

*Research Task (06/02/2025 - 06/09/2025)*

This project explores half-maximal inhibitory concentration (IC50) prediction using drug and gene features by Transformer model.

**Motivation**

During the cancer treatment, precise therapy is getting more and more important since the variability among individuals exist. Hence, IC50, an important measurement for drug sensitivity, is worth re-visiting. Inspired by the work of Carvalho et al. (2022) which predicts IC50 with gene mutation data and drug SMILES using deep neural network, this project aims to explore the possibility of predicting IC50 by chemical structures of drugs and the gene expression profiles of cancer cell lines using Transformer model. The model takes drug SMILES, gene ids, and gene expressions as input and achieves a MSE of 0.227, a Pearson Correlation Coefficient (PCC) of 0.98, and a Spearman Correlation Coefficient (SCC) of 0.98.

**Data**

The primary database used in this study is Tahoe-100M (Zhang et al., 2025). The complete Tahoe database cotains 100M observations of gene expression data, including genes, gene expressions, drug name, and canonical SMILES. Due to the time limit, this study only use a subset of Tahoe-100M gene expression dataset. Details on the subsetting procedure can be found in [downsample](downsample.py).

The prediction target IC50s are obtained from GDSC database. After combining GDSC1 and GDSC2, the dataset contained 575,197 observations. Drug name and cell line name were used as composite keys. For duplicated values of IC50 within a single version of GDSC, aggregated mean is used. For duplicated observations between 2 versions of GDSC, DGSC2 value is considered more updated and is kept in the processed dataset. Please see [pre-processing](preprocessing.ipynb) for more details.

The combined dataset contains 1,652,583 observations after merging downsampled Tahoe dataset and processed GDSC dataset by drug name and cell-line. Due to storage limitations, 20% of the merged dataset was randomly sampled for model training, resulting in a final dataset of 330,517 observations, including 20 unique drugs and 28 unique cell lines. IC50 values (on a natural log scale) ranged from -7 to 5, with a mean of 1.6 and a standard deviation of 2.9. More details of descriptive analysis can be found in [pre-processing](preprocessing.ipynb). The distribution of prediction target is shown below.

![Distribution of log(IC50)](ic50_distn.png)

**Methods**

Due to the limited diversity of SMILES strings in the final dataset, training a SMILES encoder from scratch was impractical. So, this study uses the pretrained ChemBERTa (Chithrananda et al., 2020) to encode drug SMILES. ChemBERTa is trained on a large chemical corpus and captures rich chemical features with a state-of-art performance. Gene features were created by combining gene ID embeddings with gene expression embeddings. And then the drug and gene features are combined for the input of a Transformer encoder. The default setting of model architecture contains an absolute positional encoding layer, 3 encoder layer, and a regression head composed of a two-layer MLP with ReLU activation. The total number of trainable parameters is 11,378,177. Please visit [training log](slurms/0.out) for references.

 - training settings

 The dataset is split to train, valdation, test set  in an 80/10/10 ratio. The pipeline uses Adam optimizer with a learning rate of 1e-4 a weight decay rate of 1e-5 to prevent overfitting. The total number of training epoches is set as 100 with a patience of 3 for early stopping. During the training, model is evaluated by MSE loss. The final model performance was evaluated on the testing set using MSE, PCC, and SCC.

 - computational resources

 The training process will take around 7 hours on GPU RTX Ada 5000 with 100GB memory. Please see [train slurm](slurms/train_array.sh) for references.

**Results & Discussions**

On the testing set, the best model achieved an MSE of 0.227, a PCC of 0.98, and an SCC of 0.98. Training and validation loss curves shown below are aggregated result from 3 runs with different random seeds to aviod randomness. The Transformer model is good at capturing global context. As genes do not act independently to influence the drug sensitivity, instead, they may interact with each other within the network, the self-attention mechanism in the Transformer enables the model to learn these interactions, and the positional encoding helps preserve structural information in the input.

![Training and Validation Loss Curve](train_val_loss.png)

The results suggest that gene id, gene expression and chemical structure are all important predictors for drug sensitivity. Gene expression reflects the biological state of the cell line, while chemical structure captures critical properties of the drug. In application, if we have the gene expression profile from patient tumor samples and the SMILES of compound to be tested, we are able to predict the sensitivity of a certain drug to a certain patient. This could provide the insights for drug-resistances and personalized treatments.


**Limitations & Future work**

 - This study used a subset of the Tahoe-100M dataset. Future work could explore larger or more informative subsets if the computational power allows.
 - The 2 versions of GDSC datasets (GDSC1 and GDSC2) are simply concatenated in this study to get more samples for merging. Although the distribution of IC50 are quite similar between the 2 versions, simply concatenate the 2 dataset could lead to several concerns since they are from different assays and the results might not be directly comparable. Future work should account for experimental differences between versions or only focus on one version of GDSC.
 - The merge was based solely on drug and cell line names. Matching by ids might be better to prevent the situation that a drug can have multiple names. In the future work, a third database is necessary to be the external link to connect the Tahoe and GDSC database using common identifiers.
 - The hyperparameter search of model is not complete. Due to the time limit, I only explored limited parameters for the transformer model by trying different number of layers, different sizs of hidden state and feed forward dimension. Future work should involve a more exhaustive search. Comparing absolute and rotary positional encoding could also be a good point.


**Reproduce**

The required packages to run this pipeline are exported in the yaml file: [environment](env.yaml).
To run this pipeline:
```
cd Tahoe100M_practice
python main.py
```

**References**
1. Yang, W., Soares, J., Greninger, P., Edelman, E. J., Lightfoot, H., Forbes, S., Bindal, N., Beare, D., Smith, J. A., Thompson, I. R., Ramaswamy, S., Futreal, P. A., Haber, D. A., Stratton, M. R., Benes, C., McDermott, U., & Garnett, M. J. (2013). Genomics of Drug Sensitivity in Cancer (GDSC): A resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Research, 41(Database issue), D955–D961. https://doi.org/10.1093/nar/gks1111.
2. Zhang, J., Ubas, A. A., de Borja, R., Svensson, V., Thomas, N., Thakar, N., Lai, I., et al. (2025). Tahoe-100M: A giga-scale single-cell perturbation atlas for context-dependent gene function and cellular modeling. *bioRxiv*. https://doi.org/10.1101/2025.02.20.639398.
3. Carvalho, F. G., Abbasi, M., Ribeiro, B., & Arrais, J. P. (2022). Deep model for anticancer drug response through genomic profiles and compound structures. In *2022 IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS)* (pp. 1–6). IEEE. https://doi.org/10.1109/CBMS55023.2022.00050.
4. Ahmad, W., Simon, E., Chithrananda, S., Grand, G., & Ramsundar, B. (2022). ChemBERTa-2: Towards chemical foundation models. *arXiv preprint arXiv:2209.01712*. https://arxiv.org/abs/2209.01712.