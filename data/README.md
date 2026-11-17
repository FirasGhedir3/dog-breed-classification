# Données

Les images ne sont **pas** stockées dans ce dépôt.

1. Télécharger le jeu Stanford Dogs : http://vision.stanford.edu/aditya86/ImageNetDogs/
   (`images.tar` et `annotation.tar`).
2. Garder uniquement ces 6 races et les ranger ainsi :

```
data/SmallDB/
    Images/<dossier de race>/*.jpg        images
    Annotation/<dossier de race>/<nom>    boîtes englobantes (mêmes noms, sans extension)
```

Les dossiers de races gardent leurs noms d'origine dans Stanford Dogs (par exemple `n02085620-Chihuahua`).

Races : Chihuahua, basset, Kerry blue terrier, groenendael, malinois, chow (1002 images au total).

3. Lancer `python scripts/prepare_data.py`. Il écrit `data/processed/smalldb_64.npz`, que tous
   les autres scripts lisent.
