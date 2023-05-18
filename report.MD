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



## Tehtävänanto

Työssä tarkastellaan binääriluokittelun avulla naisten selviytymistä tutkittavalla ajanjaksolla. Kaikki muuttujat otetaan tarkastelussa huomioon ja arvioidaan muuttujien avulla mikä on ennuste selviytymisen suhteen. 


## Data ja testiympäristö

Data rintasyöpäpotilaista on saatu NCI:lta. NCI (National Cancer Institute) tarjoaa tietoa väestöpohjaisista syöpätilastoista. Aineisto sisälsi naispotilaita, joilla oli infiltroiva kanavasyöpä ja lobulaarinen karsinoomarintasyöpä, jotka diagnosoitiin vuosina 2006–2010. Osa potilaista jätettiin aineistosta pois: muun muassa potilaat, joilla oli tuntematon kasvaimen koko tai joiden eloonjäämiskuukaudet olivat alle 1 kuukauden. Näin ollen 4024 potilasta sisällytettiin lopulta mukaan. (https://www.kaggle.com/c/histopathologic-cancer-detection)

Muuttujia ovat ikä, rotu, siviilisääty, kasvaimen koko (T Stage, Tumor Size), onko  kasvain levinnyt imusolmukkeisiin (N Stage), syövän luokittelu asteina (6th Stage, Grage), kasvaimen erilaistuminen (differentiate), kasvaimen paikallisuus (A Stage), estrogeenille ja progesteronille reagoiva syöpä, kasvaimen tutkiminen, selviämiskuukaudet ja status tällä hetkellä (elossa vai kuollut). (https://www.pennmedicine.org/cancer/types-of-cancer/breast-cancer/breast-cancer-staging)


## Metodologia

Edellä kuvattua datasettiä voidaan lähes suoraan käyttää binääriluokitteluongelman muodostamiseen eloonjäämisen suhteen. Datassa piti silti tehdä hieman siistimistä, koska alkuun tulokset näyttivtä hyvin oudoilta. Dataa tarkemmin tarkasteltuna kävi ilmi, että data on syötetty kahdella eri tavalla sarakkeseen 'Status'. Esimerkiksi elossa-status oli kirjoitettu muodossa 'Alive' ja 'Alive;', jolloin ylimääräinen kaksoispiste piti käsitellä datasetistä pois. Tämän jälkeen arvot 'Alive' ja 'Dead' muutettiin vielä numeerisiksi arvoiksi niin, että arvo 0 vastaa arvoa 'elossa' ja arvo 1 vastaa arvoa 'kuollut'. 


Optimaalinen arkkitehtuuri täytyy löytää sovellus- ja datakohtaisesti. Tavoitteena on, että saadaan malli, joka on suorituskykyinen ennestään tuntemattomien mallien kanssa - ei opetusdatan kanssa. Oppimisprosessiin voidaan vaikuttaa hyperparametreilla, esimerkiksi oppimisnopeudella, batch-koolla, epochien ja kerrosten lukumäärällä. Grid search on yksi tekniikka, jolla voidaan hakea sopivat hyperparametrit mallin optimoimiseen. Tässä työssä on käytetty grid_searchia. Grid_searchin lopputuloksena saatiin konfiguraatioille kaikille sama tarkkuus 87.578. Opetus- ja validointivirheissä oli pientä heittoa kuvaajien perusteella, mutta samansuuntaisilta nekin näyttivät keskenään. Kohinaa ei tällä datalla olllut juurikaan, mutta opetus- ja vallidointitarkkuuden suhteen kuvaajissa on enemmän heittelyä. Tämän perusteella valikoitui viisi konfiguraatiota, joissa on parhaat tulokset ([learning_curves](./gs_dnn_20230517T2014/learning_curves.pdf)))
    
Testauksessa saatiin lopputuloksena 32 konfiguraatiota, joiden ajaminen kesti 1 tunti ja 32 minuuttia.


## Virheanalyysi

Mallikandidaatteja arvioidaan ja diagnosoidaan erikseen sekaannusmatriisin avulla. 


## Tulokset, virheanalyysi ja jatkotoimenpiteet

