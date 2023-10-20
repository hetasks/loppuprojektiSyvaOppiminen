## Rintasyövän selviytymisen ennustaminen binääriluokittelulla

**Tekijä:** Heta Strömberg, heta.stromberg@gmail.com

**Tiivistelmä**: World Health Organizationin nettisivujen mukaan rintasyöpä on yksi yleisimmistä syövistä, joita esiintyy (https://www.who.int/news-room/fact-sheets/detail/cancer). Kagglen sivuilta oli löydettävissä dataa, jossa on tutkittu nimenomaan rintasyövässä läsnäolevia tekijöitä (https://www.kaggle.com/c/histopathologic-cancer-detection). Näiden perusteella oli mielenkiintoista lähteä tutkimaan selviytymisen ennustetta tutkimusajan sisällä. 

## Testattu seuraavilla kirjastoilla

- Python 3.9
- [Tensorflow 2.7.0](https://www.tensorflow.org/overview/?hl=fi)
- [Pandas 1.3.5](https://pandas.pydata.org/)
- [NumPy 1.21.2](https://numpy.org/)
- [sklearn 1.0.2](https://scikit-learn.org/)
- [matplotlib 3.5.0](https://matplotlib.org/)


## Johdanto

World Health Organizationin nettisivujen mukaan rintasyöpä on yksi yleisimmistä syövistä, joita esiintyy (https://www.who.int/news-room/fact-sheets/detail/cancer). Kagglen sivuilta oli löydettävissä dataa, jossa on tutkittu nimenomaan rintasyövässä läsnäolevia tekijöitä (https://www.kaggle.com/c/histopathologic-cancer-detection). Datassa on huomioitu erilaisia muuttuvia tekijöitä, kuten sairastuneen ikää, kasvaimen kokoa ja laatua, hormoneille aktiivista rintasyöpämuotoa sekä selviytymistä. Selviytymisen suhteen on erikseen määritelty selviytymiskuukausien määrä sekä kuolleisuus: onko henkilö ollut elossa vai kuollut tutkimusajan sisällä. 

Maailmanlaajuisesti rintasyöpä todettiin yli 2,2 miljoonalla naisella vuonna 2020. Sen yleisyys on ohittanut USA:ssa ihosyövän ja keuhkosyövän esiintyvyyden. Kuolinsyynä rintasyöpä on viidenneksi yleisin. Kuitenkin selviytyminen viiden vuoden ajanjaksolla tarkasteltuna kertoo, että ei-metastoivasta rintasyövästä selviää 91 %. 10 vuoden seurantajaksolla vastaava luku on 85 %. (https://www.cancer.net/cancer-types/breast-cancer/statistics)

Työssä tarkastellaan kuolleisuutta tarkkailuajanjaksolla. Tarkoitus on ottaa huomioon kaikki muuttujat annetun aineiston mukaan ja ennustaa kuolleisuutta näiden perusteella. Ongelmaa on lähdetty käsittelemään binääriluokitteluna ja tuloksissa nähdään kuinka hyvin aineiston muut parametrit pystyvät ennustamaan kuolleisuuttta.



## Tehtävänanto

Työssä tarkastellaan binääriluokittelun avulla naisten selviytymistä tutkittavalla ajanjaksolla. Kaikki muuttujat otetaan tarkastelussa huomioon ja arvioidaan muuttujien avulla mikä on ennuste selviytymisen suhteen. 


## Data ja testiympäristö

Data rintasyöpäpotilaista on saatu NCI:lta. NCI (National Cancer Institute) tarjoaa tietoa väestöpohjaisista syöpätilastoista. Aineisto sisälsi naispotilaita, joilla oli infiltroiva kanavasyöpä ja lobulaarinen karsinoomarintasyöpä, jotka diagnosoitiin vuosina 2006–2010. Osa potilaista jätettiin aineistosta pois: muun muassa potilaat, joilla oli tuntematon kasvaimen koko tai joiden eloonjäämiskuukaudet olivat alle 1 kuukauden. Näin ollen 4024 potilasta sisällytettiin lopulta mukaan. (https://www.kaggle.com/c/histopathologic-cancer-detection)

Muuttujia ovat ikä, rotu, siviilisääty, kasvaimen koko (T Stage, Tumor Size), onko  kasvain levinnyt imusolmukkeisiin (N Stage), syövän luokittelu asteina (6th Stage, Grage), kasvaimen erilaistuminen (differentiate), kasvaimen paikallisuus (A Stage), estrogeenille ja progesteronille reagoiva syöpä, kasvaimen tutkiminen, selviämiskuukaudet ja status tällä hetkellä (elossa vai kuollut). (https://www.pennmedicine.org/cancer/types-of-cancer/breast-cancer/breast-cancer-staging)


## Metodologia

Edellä kuvattua datasettiä voidaan lähes suoraan käyttää binääriluokitteluongelman muodostamiseen eloonjäämisen suhteen. Datassa piti silti tehdä hieman siistimistä, koska alkuun tulokset näyttivät hyvin oudoilta. Dataa tarkemmin tarkasteltuna kävi ilmi, että data on syötetty kahdella eri tavalla sarakkeseen 'Status'. Esimerkiksi elossa-status oli kirjoitettu muodossa 'Alive' ja 'Alive;', jolloin ylimääräinen kaksoispiste piti käsitellä datasetistä pois. Tämän jälkeen arvot 'Alive' ja 'Dead' muutettiin vielä numeerisiksi arvoiksi niin, että arvo 0 vastaa arvoa 'elossa' ja arvo 1 vastaa arvoa 'kuollut'. Lisäksi testidataa jaettaessa evaluointi- ja unseen-dataseteetihin oli otettava mukaan "stratify"-parametri. Kyseinen parametri tekee jaon niin, että esimerkkidata jaetaan suhteessa saman arvon eri setteihin. Tämä tekee lopullisesta analyysista luotettavamman.


Optimaalinen arkkitehtuuri täytyy löytää sovellus- ja datakohtaisesti. Tavoitteena on, että saadaan malli, joka on suorituskykyinen ennestään tuntemattomien mallien kanssa - ei opetusdatan kanssa. Oppimisprosessiin voidaan vaikuttaa hyperparametreilla, esimerkiksi oppimisnopeudella, batch-koolla, epochien ja kerrosten lukumäärällä. Grid search on yksi tekniikka, jolla voidaan hakea sopivat hyperparametrit mallin optimoimiseen. Tässä työssä on käytetty grid_searchia. Grid_searchin lopputuloksena saatiin konfiguraatioille tarkkuus, joka vaihteli välillä 85.075 ja 90.547. Opetus- ja validointivirheitä voidaan tarkastella kuvaajilla (learning_curves). Malliin tallennettiin viisi parasta mallia ja näistä esimerkiksi cfg nro 13 on tarkkkuuden osalta onnistunut hyvin. Opetustrendi lähtee kyseisessä kuvaajassa nousemaan alussa melko jyrkästi ja lopulta se tasoittuu aika pienellä intervallilla. Kohinaa ei tällä datalla olllut juurikaan näissä viidessä parhaassa, mutta muiden kuvaajien joukossa sitäkin on nähtävissä. ([learning_curves](./gs_dnn_20231019T0938/learning_curves.pdf)))
    
Testauksessa saatiin lopputuloksena 32 konfiguraatiota, joiden ajaminen kesti 1 tunti ja 50 minuuttia. Virheitä lopputulokseen voi tuoda se, että malleja ajetaan vain 50 epochia ja dataa on suhteellisen vähän.


## Virheanalyysi

Mallikandidaatteja arvioidaan ja diagnosoidaan erikseen sekaannusmatriisin, sen johdannaismetriikkojen ja F1-mitan avulla. Lisäksi virheanalyysissa hyödynnetään tilastollisia summamuuttujia ja manuaalista tarkastelua. Diagnosoinnin ja virheanalyysin avulla tehdään valinta lopullisesta mallista.

Valitun mallin suorituskyky evaluointidataa vastaan:

| TP  | TN  |  FN  | FP  |   ACC   |  PREC   |  REC     | FSCORE  |
| :-: | :-: |  :-: | :-: |   :-:   |   :-:   |  :-:     |  :-:    |
| 28  | 336 |  33  |  5  | 0.90547 | 0.84848 | 0.45902  | 0.59574 |

Laatikko-janakuvioon (_boxplot_) tiivistyy paljon informaatiota datan ankkurikohtaisesta jakautuneisuudesta ja sen avulla saa yleiskuvan siitä, mitä arvoja yleisesti ottaen oikein ja väärin tunnistetut havainnot saavat ja minkälainen niiden hajonta on. Laatikon korkeus havainnollistaa interkvartiilialuetta eli mille välille 50 prosenttia arvoista keskittyvät. Laatikossa oleva oranssi viiva havainnollistaa mediaaniarvoa. Laatikon ala- ja yläviiksien välinen alue kuvastaa ankkurille mahdollisia arvoja ja niiden ulkopuolelle jäävät pallot mahdollisia poikkeamia.


## Tulokset, virheanalyysi ja jatkotoimenpiteet

Mallin suorituskyky testidataa vastaan (@ th:0.75):

|  TP  |  TN  |  FN  |  FP  |  ACC    |  PREC   |  REC    |  FSCORE |
|  :-: |  :-: |  :-: |  :-: |  :-:    |  :-:    |  :-:    |  :-:    |
|  14  |  338 |  48  |  3   | 0.87345 | 0.82353 | 0.22581 | 0.35443 |

Testidataa vasten tehty mallin suorituskyky heikkeni verrattuna evaluointidataa vastaan. Tähän vaikuttanee muun muassa datan vähäinen määrä sekä se, että välttämättä kuolleisuutta ei luotettavasti pystytä ennustamaan sen perusteella mitkä tekijät on datan keruussa huomioitu. Jatkossa olisi mielenkiintoista saada selville päästäisiinkö parempaan ennustettavuuteen mikäli dataa olisi enemmän saatavilla. Toisaalta voidaan myös miettiä, onko kaikki tekijät täysin oleellisia huomioida ennustuksessa. Kuten esimerkiksi siviilisääty tai ihonväri. Joka tapauksessa tästäkin datasta saisi erilaisella datan käsittelyllä varmasti mielenkiintoisia tuloksia aikaiseksi. 