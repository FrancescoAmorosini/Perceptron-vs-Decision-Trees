# Perceptron vs Decision Tree

Questo progetto ha l'obiettivo di confrontare le curve di apprendimento di due semplici classificatori molto utilizzati nel campo dell'apprendimento supervisionato: Perceptron (single-layer) e Decision Tree.

## Dataset

I dati utilizzati per realizzare questo confronto sono reperibili tramite [questo link](https://github.com/zalandoresearch/fashion-mnist). Si tratta di un dataset di immagini di articoli di moda 28x28 in scala di grigio messo a disposizione da Zalando. Per il funzionamento degli esperimenti **non** è richiesto all'utente di scaricare l'intera repository contenente il dataset: durante la prima esecuzione del codice i file e gli script necessari saranno scaricati automaticamente dal link indicato sopra, e inseriti in un'apposita cartella `./data`.

## Prerequisiti

Per la corretta esecuzione del codice è necessario che l'utente scarichi le librerie e i moduli che sono stati adoperati per la realizzazione del progetto. Di seguito abbiamo una lista di tutti i package utilizzati con il relativo utilizzo che ne è stato fatto:

- [NumPy](http://www.numpy.org): funzioni di utility generiche.
- [Matplotlib](https://matplotlib.org): tracciamento delle curve di apprendimento.
- [Sci-Kit Learn](https://scikit-learn.org/stable/index.html#): libreria contenente i modelli studiati.
- [GitPython](https://gitpython.readthedocs.io/en/stable/): modulo necessario per scaricare temporaneamente il dataset durante la prima esecuzione.
- [os](https://docs.python.org/3/library/os.html#module-os), [shutil](https://docs.python.org/3/library/shutil.html#module-shutil), [tempfile](https://docs.python.org/3/library/tempfile.html#module-tempfile), [collections](https://docs.python.org/3/library/collections.html#module-collections): moduli miscellanei **già inclusi** in Python 3.7, utilizzati per creare la cartella `./data` a partire dal dataset scaricato precedentemente.
- [PyDotPlus](https://pydotplus.readthedocs.io): modulo facoltativo utilizzato esclusivamente per la visualizzazione degli alberi di decisione.

L'utente può scaricare i moduli mancanti semplicemente utilizzando il package manager [pip](https://pip.pypa.io/en/stable/) da riga di comando.

```bash
pip install <modulo mancante>
```

## Esecuzione
L'interfaccia dalla quale è possibile compiere gli esperimenti è costituita dalla classe `main.py`. Da qui è infatti possibile modificare i macroparametri dei classificatori (e.g. il numero di epoche per il perceptron, la massima profondità dell'albero, ...) e il numero di esperimenti da eseguire (tramite il parametro `iterations`). La funzione di testing eseguirà l'esperimento il numero di volte indicato e provvederà a disegnare una curva che rappresenta la media e la deviazione standard dei risultati ottenuti. La funzione di testing è compatibile con qualsiasi stimatore instanziato tramite la libreria Sci-Kit Learn.

```python
from sklearn import tree
import test

[...]

decision_tree = tree.DecisionTreeClassifier(max_depth=13)

#Draw learning curve
test.plot_learning_curve(decision_tree, (train, test), (train_labels, test_labels), iterations = 30)
```

Inoltre è possibile disegnare a piacimento sia l'albero di decisione ottenuto, sia ogni singolo sprite contenuto nel dataset. Queste funzioni non sono fondamentali per la riuscita degli esperimenti, infatti durante i test non vengono mai esplicitamente chiamate, tuttavia per una migliore comprensione dei dati e degli oggetti che si stanno manipolando è possibile invocarle dal file `util.py`.


```python
import util

util.draw_image(x_train[<index>])
util.draw_image(x_test[<index>])
util.draw_tree(decision_tree)
```
Per maggiori dettagli sulla conduzione degli esperimenti è consigliato leggere il documento `relazione.pdf`.

