# LinearRegression

Voici une implémentation de la régression linéaire avec descente de gradient pour minimiser la fonction de coût RMSE (Root Mean Square Error) en C#. Cette implémentation supporte la version stochastique, mini-batch, ou la version batch complète, en fonction de la taille du batch spécifiée par l'utilisateur via le paramètre batch_size dans la méthode Fit.

## Explication

### Descente de gradient : La descente de gradient est utilisée pour ajuster les poids du modèle afin de minimiser l'erreur (RMSE ici)

* Stochastique : Si batchSize == 1, cela correspond à une descente de gradient stochastique (un seul échantillon à la fois).

* Mini-batch : Si 1 < batch_size < nombre d'échantillons, cela correspond à une descente de gradient mini-batch (plusieurs échantillons sont utilisés à la fois).

* Batch complète : Si batch_size == nombre d'échantillons, cela correspond à la descente de gradient par batch complet (l'ensemble des données est utilisé à chaque itération).
