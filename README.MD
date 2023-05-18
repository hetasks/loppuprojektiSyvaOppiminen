## Rintasyövän selviytymisen ennustaminen binääriluokittelulla

**Tekijä:** Heta Strömberg, heta.stromberg@gmail.com

**Tiivistelmä**: World Health Organizationin nettisivujen mukaan rintasyöpä on yksi yleisimmistä syövistä, joita esiintyy (https://www.who.int/news-room/fact-sheets/detail/cancer). Kagglen sivuilta oli löydettävissä dataa, jossa on tutkittu nimenomaan rintasyövässä läsnäolevia tekijöitä (https://www.kaggle.com/c/histopathologic-cancer-detection). Tutkimustiedon perusteella on ennustettu kuolleisuutta binääriluokittelun avulla.

## Testattu seuraavilla kirjastoilla

- Python 3.9
- [Tensorflow 2.7.0](https://www.tensorflow.org/overview/?hl=fi)
- [Pandas 1.3.5](https://pandas.pydata.org/)
- [NumPy 1.21.2](https://numpy.org/)
- [sklearn 1.0.2](https://scikit-learn.org/)
- [matplotlib 3.5.0](https://matplotlib.org/)

## Työn rakenne

```
loppuprojekti
│   README.md
│   report.md    
│   param_grid.txt
│   cfg_list.txt
│   1_data_preprocessing
│   2_grid_search.py
│   3_model_diagnostics_and_predictions.ipynb
│
└───data
└───gs_dnn_ensemble_20220311T0958
└───gs_dnn_ensemble_20220321T0821
└───resources
```

ks. [raportti](./report.md).

## Tulokset

Valitun mallin (ks. [1D-CNN-2](./gs_dnn_ensemble_20220321T0821/model_infos/1D-CNN-2.png) ja [summary](./gs_dnn_ensemble_20220321T0821/model_infos/1D-CNN-2_summary.txt)) suorituskyky evaluointidataa vastaan:

| TP | TN | FN | FP | ACC | PREC | REC | FSCORE
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 21 | 21 | 0 | 0 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |

Valitun mallin suorituskyky testidataa vastaan (@ th:0.75):

| TP | TN | FN | FP | ACC | PREC | REC | FSCORE
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 50 | 32 | 4 | 18 | 0.77885 | 0.71831 | 0.94444 | 0.81600 |