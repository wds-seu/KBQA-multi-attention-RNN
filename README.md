# KBQA-multi-attention-RNN
Source code and datasets of paper: Complex Question Answering over Knowledge Base with Multi-Attention RNN.

## Requirements  
jieba                     0.39  
python                    3.6  
keras                     2.2.4   
nltk                      3.4.5    
numpy                     1.16.5  
scikit-learn              0.21.2  
tensorflow                1.14.0  
tqdm                      4.32.1 

## Datasets
[LC_Quad](https://figshare.com/projects/LC-QuAD/21812)

## Run and Test
* entity_linking  
``` 
python build_data.py      
python train.py  
python evaluate.py
```
* relation_detection
```
python preprocess.py
python model_twoatt_con.py
python eval_twoatt_con.py
```

