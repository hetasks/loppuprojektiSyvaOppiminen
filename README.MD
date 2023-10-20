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
└───gs_dnn_20231019T0938
└───resources
```

ks. [raportti](./report.md).

## Tulokset

Valitun mallin (./gs_dnn_20231019T0938) suorituskyky evaluointidataa vastaan:

| TP  | TN  |  FN  | FP  |  ACC    |  PREC   |  REC    |  FSCORE |
| :-: | :-: |  :-: | :-: |  :-:    |  :-:    |  :-:    |  :-:    |
| 14  | 338 |  48  |  3  | 0.89345 | 0.22581 | 0.35443 | 1.00000 |

Valitun mallin suorituskyky testidataa vastaan:

| TP  | TN  |  FN |  FP |   ACC    |  PREC   |  REC    |  FSCORE |
| :-: | :-: | :-: | :-: |   :-:    |  :-:    |  :-:    |  :-:    |
| 14  | 338 |  48 |  3  | 0.87345  | 0.22581 | 0.35443 | 0.35443 |