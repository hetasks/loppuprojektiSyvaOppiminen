{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d4f9b720-4a5a-469b-ad8a-7ed9879a8b18",
   "metadata": {},
   "source": [
    "Optimaalinen arkkitehtuuri täytyy löytää sovellus- ja datakohtaisesti. Tavoitteena on, että saadaan malli, joka on suorituskykyinen ennestään tuntemattomien mallien kanssa - ei opetusdatan kanssa. Oppimisprosessiin voidaan vaikuttaa hyperparametreilla, esimerkiksi oppimisnopeudella, batch-koolla, epochien ja kerrosten lukumäärällä. Grid search on yksi tekniikka, jolla voidaan hakea sopivat hyperparametrit mallin optimoimiseen."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f1886225-8a10-4ed2-9e87-9c0fa82cf451",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import shutil\n",
    "\n",
    "from tensorflow.keras import Model\n",
    "from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise\n",
    "from tensorflow.keras.regularizers import l2\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib.backends.backend_pdf import PdfPages\n",
    "from datetime import datetime"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "47dd16aa-54a2-4c65-bbf5-c1178f4bbb5d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 4024 entries, 0 to 4023\n",
      "Data columns (total 16 columns):\n",
      " #   Column                  Non-Null Count  Dtype\n",
      "---  ------                  --------------  -----\n",
      " 0   Age                     4024 non-null   int64\n",
      " 1   Race                    4024 non-null   int32\n",
      " 2   Marital Status          4024 non-null   int32\n",
      " 3   T Stage                 4024 non-null   int32\n",
      " 4   N Stage                 4024 non-null   int32\n",
      " 5   6th Stage               4024 non-null   int32\n",
      " 6   differentiate           4024 non-null   int32\n",
      " 7   Grade                   4024 non-null   int32\n",
      " 8   A Stage                 4024 non-null   int32\n",
      " 9   Tumor Size              4024 non-null   int64\n",
      " 10  Estrogen Status         4024 non-null   int32\n",
      " 11  Progesterone Status     4024 non-null   int32\n",
      " 12  Regional Node Examined  4024 non-null   int64\n",
      " 13  Reginol Node Positive   4024 non-null   int64\n",
      " 14  Survival Months         4024 non-null   int64\n",
      " 15  Status;                 4024 non-null   int32\n",
      "dtypes: int32(11), int64(5)\n",
      "memory usage: 330.2 KB\n"
     ]
    }
   ],
   "source": [
    "df_breastCancer = pd.read_csv(\"https://raw.githubusercontent.com/hetasks/loppuprojektiSyvaOppiminen/main/Breast_Cancer.csv\", sep=\",\")\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "encoder = LabelEncoder()\n",
    "def encodeColumn(parameter):\n",
    "    df_breastCancer[parameter] = encoder.fit_transform(df_breastCancer[parameter])\n",
    "\n",
    "# Run Function\n",
    "listColumn = ['Race','Marital Status','T Stage ','N Stage','6th Stage','differentiate','Grade','A Stage','Estrogen Status','Progesterone Status','Status;']\n",
    "for i in listColumn:\n",
    "    encodeColumn(i)\n",
    "df_breastCancer.info()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aed81edb-4985-47ea-91a8-ac59f298e8a1",
   "metadata": {},
   "source": [
    "Otetaan käyttöön oppimiskäyrän piirtofunktio:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "711eae6c-efcc-4b19-88f9-3149cf385c26",
   "metadata": {},
   "outputs": [],
   "source": [
    "def learning_curves(history, i, accuracy, pdf):\n",
    "    fig = plt.figure(figsize=(10, 4))\n",
    "    hist_dict = history.history\n",
    "    epochs = [x+1 for x in history.epoch]\n",
    "    plt.subplot(1,2,1)\n",
    "    plt.plot(epochs, hist_dict['loss'], label=\"Opetusvirhe\")\n",
    "    plt.plot(epochs, hist_dict['val_loss'], label=\"Validointivirhe\")\n",
    "    plt.title('Opetus- ja validointivirhe')\n",
    "    plt.xlabel('Epochs')\n",
    "    plt.ylabel('Loss')\n",
    "    plt.legend()\n",
    "    plt.subplot(1,2,2)\n",
    "    plt.plot(epochs, hist_dict['accuracy'], label=\"Opetustarkkuus\")\n",
    "    plt.plot(epochs, hist_dict['val_accuracy'], label=\"Validointitarkkuus\")\n",
    "    plt.title('Opetus- ja validointitarkkuus')\n",
    "    plt.xlabel('Epochs')\n",
    "    plt.ylabel('Accuracy')\n",
    "    plt.legend()\n",
    "    plt.suptitle(f\"cfg nro. {i}: >>> {accuracy*100:.5f}\")\n",
    "    plt.tight_layout()\n",
    "    pdf.savefig(fig)\n",
    "    plt.close(fig)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3bc34f1-32e9-4e5d-80a0-aa30ce63c88b",
   "metadata": {},
   "source": [
    "Tehdään sanakirja. Avaimiksi tulee hyperparametreja ja muita, joihin halutaan vaikuttaa. Seuraavissa tulostetaan kirjasto, jotta saadaan listana tulostettua tämä sanakirja listana ja kaikki sen eri kombinaaatiot."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0e26406a-b308-4e06-ab56-d11667c63879",
   "metadata": {},
   "outputs": [],
   "source": [
    "param_grid = {\n",
    "    'layers': [[15, 11], [30, 10]],\n",
    "    'reg': [None, 0.001],\n",
    "    'dout': [[0.0, 0.0], [0.1, 0.0]],\n",
    "    'act': ['relu', 'elu'],\n",
    "    'init': ['he_uniform'],\n",
    "    'noise': [0.0],\n",
    "    'test_pct': [0.20],\n",
    "    'lr': [0.001, 0.0001],\n",
    "    'epochs': [150],\n",
    "    'b_size': [12],\n",
    "    'num_classes': [1]\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3ad86d28-4e0d-458f-b098-277ebaa80abe",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import ParameterGrid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "fd43450d-c445-407c-bcba-50878166bc78",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'relu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.0, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [15, 11],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': None,\n",
       "  'test_pct': 0.2},\n",
       " {'act': 'elu',\n",
       "  'b_size': 12,\n",
       "  'dout': [0.1, 0.0],\n",
       "  'epochs': 150,\n",
       "  'init': 'he_uniform',\n",
       "  'layers': [30, 10],\n",
       "  'lr': 0.0001,\n",
       "  'noise': 0.0,\n",
       "  'num_classes': 1,\n",
       "  'reg': 0.001,\n",
       "  'test_pct': 0.2}]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(ParameterGrid(param_grid))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "47afb01a-ba51-4b0f-ac9b-cca7c9d26d48",
   "metadata": {},
   "source": [
    "Nähdään myös listan pituus"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "4b82d6d9-fc95-4f7d-8a3f-24ab5cc3be5c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "32"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(list(ParameterGrid(param_grid)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e3544793-d3fb-4653-b7c6-ad6e9b46d3e9",
   "metadata": {},
   "source": [
    "Tehdään konfiguraatiot -tietorakenne."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "08f12ea4-928d-4a7b-9839-0ae6d601e0a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "2 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "3 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "4 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "5 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "6 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "7 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "8 {'act': 'relu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "9 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "10 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "11 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "12 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "13 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "14 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "15 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "16 {'act': 'relu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "17 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "18 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "19 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "20 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "21 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "22 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "23 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "24 {'act': 'elu', 'b_size': 12, 'dout': [0.0, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "25 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "26 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "27 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "28 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [15, 11], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "29 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "30 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n",
      "31 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': None, 'test_pct': 0.2}\n",
      "32 {'act': 'elu', 'b_size': 12, 'dout': [0.1, 0.0], 'epochs': 150, 'init': 'he_uniform', 'layers': [30, 10], 'lr': 0.0001, 'noise': 0.0, 'num_classes': 1, 'reg': 0.001, 'test_pct': 0.2}\n"
     ]
    }
   ],
   "source": [
    "cfgs = {i: cfg for i, cfg in enumerate(list(ParameterGrid(param_grid)), 1)}\n",
    "for key, value in cfgs.items():\n",
    "    print(key, value)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c7820ce-7466-44bd-a137-4fb4e2d6520a",
   "metadata": {},
   "source": [
    "Otetaan käyttöön timestamp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3c21f15c-2750-458d-ab4d-ad2cd4206003",
   "metadata": {},
   "outputs": [],
   "source": [
    "time = datetime.now().strftime('%Y%m%dT%H%M')\n",
    "models_path = f\"gs_dnn_{time}\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5610a3e6-66f9-467b-a675-45757d60d835",
   "metadata": {},
   "source": [
    "Alustetaan polku ja samoin pdf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "3e084408-6c3c-43a8-9ee9-465c682b55d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "if not os.path.exists(models_path):\n",
    "    os.mkdir(models_path)\n",
    "    \n",
    "pdf = PdfPages(os.path.join(models_path, \"learning_curves.pdf\"))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71f9a44f-967d-476b-aa09-e606b67e0472",
   "metadata": {},
   "source": [
    "Jaetaan data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f12dedff-131e-4b21-899d-04e255b82ce5",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_X_train_full, df_X_test_full, df_y_train_full, df_y_test_full = train_test_split(df_breastCancer.iloc[:,0:], df_breastCancer.iloc[:,15], test_size=0.20, random_state=42)\n",
    "df_X_test_eval, df_X_test_unseen, df_y_test_eval, df_y_test_unseen = train_test_split(df_X_test_full, df_y_test_full, test_size=0.80, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b89f7da-c802-44c5-bba2-0bec5bf448f7",
   "metadata": {},
   "source": [
    "Tehdään funktioita ennen grid_search - funktiota."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "955baa69-8c53-4362-986f-caf5e3dfef5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_model(df_X_train_full, df_y_train_full, cfg, models_path):\n",
    "    X_train, X_valid, y_train, y_valid = split_data(df_X_train_full, df_y_train_full, cfg['test_pct'])\n",
    "    model = build_model(X_train.shape[1], cfg)\n",
    "    sgd = Adam(amsgrad=True, learning_rate=cfg['lr'])\n",
    "    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])\n",
    "    checkpoint_dir = os.path.sep.join([models_path, 'tmp'])\n",
    "    checkpoint_filepath = os.path.sep.join([checkpoint_dir, 'checkpoint'])\n",
    "    cbs_list = [\n",
    "        ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', mode='max', save_weights_only=True, verbose=0),\n",
    "        CSVLogger(os.path.sep.join([models_path, 'training.log']), append=True)\n",
    "        ]\n",
    "    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=cfg['epochs'], batch_size=cfg['b_size'], callbacks=cbs_list, verbose=0)\n",
    "    open(os.path.sep.join([models_path, 'training.log']), 'a').write(f\"model.load_weights(checkpoint_filepath), shutil.rmtree(checkpoint_dir) \\n\\n\")\n",
    "    return model, history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "3ade6f96-a292-47a5-88c1-7e2793e3e982",
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_data(df_X, df_y, test_pct):\n",
    "    df_X_train, df_X_valid, df_y_train, df_y_valid = train_test_split(df_X, df_y, test_size=test_pct)\n",
    "    return df_X_train.to_numpy(), df_X_valid.to_numpy(), df_y_train.to_numpy(), df_y_valid.to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "da0dabd2-a48d-40f7-82f1-e98b72473f6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_model(input_shape, cfg):\n",
    "    reg = l2(cfg['reg']) if cfg['reg'] is not None else None\n",
    "    input_data = Input(shape=(input_shape,))\n",
    "    x = GaussianNoise(cfg['noise'])(input_data)\n",
    "    for layer_size in cfg['layers']:\n",
    "        x = Dropout(cfg['dout'][0])(x)\n",
    "        x = Dense(layer_size, activation=cfg['act'], kernel_initializer=cfg['init'], kernel_regularizer=reg)(x)\n",
    "        x = Dropout(cfg['dout'][1])(x)\n",
    "        output = Dense(cfg['num_classes'], activation=\"sigmoid\")(x)\n",
    "        model = Model(input_data, output)\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c5831eb5-ccd2-4a1b-b590-678ca298b7e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate_model(model, df_X_test_eval, df_y_test_eval):\n",
    "    _, accuracy = model.evaluate(df_X_test_eval.to_numpy(), df_y_test_eval.to_numpy(), verbose=0)\n",
    "    print(f\"Evaluation accuracy: {accuracy*100:.3f}\")\n",
    "    return accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5f881ac0-d286-46b8-86c9-042790a444bc",
   "metadata": {},
   "source": [
    "Otetaan muuttujaan nimeltä \"scores\" talteen opetusdata, testidata, konfiguraatiot ja polku sekä pdf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4869474d-c39c-4086-b528-6fd6216734a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def grid_search(df_X_train_full, df_y_train_full, df_X_test_eval, df_y_test_eval, cfgs, models_path, pdf):\n",
    "    scores = []\n",
    "    print(f\"Yhteensä {len(cfgs)} konfiguraatiota, aloitetaan... {models_path}\")\n",
    "    for i, cfg in cfgs.items():\n",
    "        open(os.path.sep.join([models_path, 'training.log']), 'a').write(f\"Nykyinen konfiguraatio {i}: {cfg} \\n\\n\")\n",
    "        model, history = train_model(df_X_train_full, df_y_train_full, cfg, models_path)\n",
    "        accuracy = evaluate_model(model, df_X_test_eval, df_y_test_eval)\n",
    "        learning_curves(history, i, accuracy, pdf)\n",
    "        scores.append((i, accuracy, model))\n",
    "        open(os.path.sep.join([models_path, \"all_configs.txt\"]), 'a').write(f\"{i}: {cfg} \\n\\n\")\n",
    "        if len(scores) > 5:\n",
    "            scores.sort(key=lambda tup: tup[1], reverse=True)\n",
    "        del scores[-1]\n",
    "    pdf.close()\n",
    "    return scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "ef113d09-3a2b-4ff2-8b2c-29fc716f7b3b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Yhteensä 32 konfiguraatiota, aloitetaan... gs_dnn_20230517T2014\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n",
      "Evaluation accuracy: 87.578\n"
     ]
    }
   ],
   "source": [
    "scores = grid_search(df_X_train_full, df_y_train_full, df_X_test_eval, df_y_test_eval, cfgs, models_path, pdf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "3430c474-67a1-4fe4-a1e2-b89d7ef228f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['all_configs.txt', 'learning_curves.pdf', 'tmp', 'training.log']"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.listdir(models_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "e789d192-7974-4041-a1d4-14881d46aa5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i, accuracy, model in scores:\n",
    "    print(i, accuracy, model)\n",
    "    filepath = os.path.sep.join([models_path, f'{i}_acc_{accuracy*100:.5f}.hdf5'])\n",
    "    model.save(filepath)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "db600378-4c75-42cd-8059-48101ffbf948",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['all_configs.txt', 'learning_curves.pdf', 'tmp', 'training.log']"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.listdir(models_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "6cf8e220-0c16-47e2-be83-9bda4a8bb6b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "for file in os.listdir(models_path):\n",
    "    if file.endswith('.hdf5'):\n",
    "        filename = os.path.sep.join([models_path, file])\n",
    "        model = load_model(filename)\n",
    "        preds = (model.predict(df_X_test_unseen.to_numpy()) > 0.5).astype(int)\n",
    "        print(f\"{accuracy_score(df_y_test_unseen.to_numpy(), preds)*100:.3f}\", file)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
