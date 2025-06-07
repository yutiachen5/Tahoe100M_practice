# Tahoe100M_practice

**Anticancer Drug Acctivity Prediction by Biological and Chemical Features**

*Research Task (06/02/2025 - 06/09/2025)*

This project explores half-maximal inhibitory concentration (IC50) prediction using drug and gene features with Transformer model.

**Motivation**

During the cancer treatment, precise therapy is getting more and more important since the variability among individuals exist. Hence, IC50, an important measurement for drug sensitivity, is worth re-visiting. Inspired by the work of Maryam et. al (2024) and a side project over my undergrad study, this project aims to explore the possibility of predicting IC50 in cancer cell lines using the chemical feature from drug and biological features from gene profiles by computational methods. The model takes drug smiles, gene ids, and gene expressions as input and achieves a MSE of 3, PCC of 0.5, and SCC of 0.6.

**Data**

The main database used for this study is Tahoe-100M from Zhang et.al (2025). The complete database cotains 100M observations of gene expression data, including genes, gene expressions, drug name, and canonical smiles. Due to the time limit and limitation of computational resources, this study only use first 10% of Tahoe-100M gene expression dataset. Please see [downsample](downsample.py) for more details.

The prediction target, IC50, is obtained from GDSC. The complete dataset of GDSC1 and GDSC2 conatins 575,197 observations. Composite of durg name and cell-line name is considered as the unique key. For duplicated values of IC50 within a single version of GDSC, aggregated mean is used. For duplicated observations between 2 versions of GDSC, DGSC2 value is considered more updated and is kept in the processed dataset. Please see [pre-processing](preprocessing.ipynb) for more details.

The final dataset used for training pipeline contains 6000 observations after merging downsampled Tahoe dataset and processed GDSC dataset by drug name and cell-line, including 3 unique drugs and 82 unique cell-lines. The IC50s on natural log scale distribute in the range of [-7, 5] with a mean of -1 and standard deviation of 2.7. More details of descriptive analysis can be found in [visulizations](viz.ipynb).

**Methods**

Due to the lack of diversity of drugs smiles in the final dataset, training a smiles encoder from scratch can be hard. So, this study uses the pretrained ChemBERTa from Chithrananda et.al (2020) to encode drug smiles. ChemBERTa is trained on huge chemical dataset to capture diversified chemical features with a state-of-art performance. The gene features are obtained by adding the embeddings of gene id and expression. And then the drug and gene features are combined for the input of a encoder-only transformer model. The default setting of model architecture contains an absolute positional encoding layer, 3 encoder layer, and a regression head for output. The regression head is a MLP model, which contains 2 linear projection layers with ReLU activation function. 
 - training settings
 The dataset is split to train, valdation, test set with proportion of 0.8, 0.1, and 0.1. The pipeline uses Adam optimizer with a learning rate of 1e-4 a weight decay rate of 1e-5 to prevent overfitting. The total number of training epoches is set as 10 without any early stopping. During the training, model is evaluated by validation loss. After the training, the model is evaluated by test MSE, PCC, and SCC.
 - computational resources
 The training process takes 20 minutes on GPU ATX5000 with 50GB memory. Please see [train slurm](train.sh) for references.

**Results**

The model achieves a MSE of 3, PCC of 0.5, and SCC of 0.6, which are poor. The results are collected and tracked by weights&biases.

**Limitations & Future work**

 - The downsampled dataset is the first 10% of Tahoe, which can be biased since I didn't select the sample randomly. In the future work, biased down-sampling can be considered to prefer more informative samples.
 - The 2 versions of GDSC datasets (GDSC1 and GDSC2) are simply concatenated in this study to get more samples gor merging. However, they are from different assays and experiments so the results might not be directly comparable. 
 - The merge of 2 data sources only consider drug name and cell name as merging condition using exact match. Matching by ids might be better to prevent the situation that a drug can have multiple names. In the future work, a thid database is necessary to be the external link to connect the ids of Tahoe and GDSC database.
 - The hyperparameter search of model is not complete. Due to the time limit, I only explored parameters for the transformer model by trying different number of layers, different sizs of hidden state and feed forward dimension. I also plan to compare absolute positional encoding layer with rotary positional encoding layer if given more time.

**Reproduce**

The required packages to run this pipeline are exported in the yaml file: [environment](env.yaml).
To run this pipeline:
```
python main.py
```

**References**
1. Yang W, Soares J, Greninger P, Edelman EJ, Lightfoot H, Forbes S, Bindal N, Beare D, Smith JA, Thompson IR, Ramaswamy S, Futreal PA, Haber DA, Stratton MR, Benes C, McDermott U, Garnett MJ. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. Nucleic Acids Res. 2013 Jan;41(Database issue):D955-61. doi: 10.1093/nar/gks1111. Epub 2012 Nov 23. PMID: 23180760; PMCID: PMC3531057.
2. Zhang, Jesse, Airol A. Ubas, Richard de Borja, Valentine Svensson, Nicole Thomas, Neha Thakar, Ian Lai, et al. 2025. “Tahoe-100M: A Giga-Scale Single-Cell Perturbation Atlas for Context-Dependent Gene Function and Cellular Modeling.” bioRxiv. https://doi.org/10.1101/2025.02.20.639398.
3. F. G. Carvalho, M. Abbasi, B. Ribeiro and J. P. Arrais, "Deep Model for Anticancer Drug Response through Genomic Profiles and Compound Structures," 2022 IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS), Shenzen, China, 2022, pp. 1-6, doi: 10.1109/CBMS55023.2022.00050.