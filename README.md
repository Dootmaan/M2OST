# M2OST: Many-to-one Regression for Predicting Spatial Transcriptomics from Digital Pathology Images

This is the official repository for AAAI 2025 paper *M2OST: Many-to-one Regression for Predicting Spatial Transcriptomics from Digital Pathology Images*

The code of the core M2OST network will be released shortly after in a torch.nn.Module style, while the complete training/validation code will be updated after a comprehensive sweep. Thank you for your interest in our work.

## Update 2025/04/16

Sorry for the late release. As I have finished my diseertatioon today, all the related codes are going to be released very soon. Right now a roughly updated version have been pushed, but please note that the pretrained weights are not verified yet and they may be from my previous project (see [M2ORT](https://github.com/Dootmaan/M2ORT)). I will soon check all the weights and only keep the ones that can fit M2OST.

To run the code, please follow the instructions below:

1. Clone the code

Clone the code by running:

```
git clone git@github.com/Dootmaan/M2OST.git
```

then cd into this directory.

2. Prepare your datasets

Download the datasets from their official site.

* HBC: https://data.mendeley.com/datasets/29ntw7sh4r/5.
  * Make sure you have also downloaded [this file](https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym&col=gd_app_name&col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit) before using the HBC dataset.
* HER2+: https://zenodo.org/records/3957257#.Y4LB-rLMIfg.
  * The files are encrypted by 7z. To decrypt these files, use the following passwords:
    * count matrices and images: zNLXkYk3Q9znUseS
    * meta data and spot selection: yUx44SzG6NdB32gY
* cSCC: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240

3. Train or validate the model

Start training the M2ORT model using the following command:

```
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -u train_m2ost_example.py >train_m2ost_example.log 2>&1
```

Please note that the train_m2ost_example.py is for HBC dataset. You will have to modify the code accordingly when testing on the HER2+, cSCC or other datasets. You can also refer to the train_m2ort.py file in the [M2ORT repo](https://github.com/Dootmaan/M2ORT).
