import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

path = "../data/ZP-River.dat"


def load_data(path):
    diretorio = os.path.join(os.path.dirname(__file__), path)

    nomes = [
        "POSITION",
        "EHS",
        "TOTAL_POT",
        "POT_ODDS",
        "BOARD_SUIT",
        "BOARD_CARDS",
        "BOARD_CONNECT",
        "PREV_ROUND_ACTION",
        "PREVIOUS_ACTION",
        "BET_VILLAIN",
        "AGG",
        "IP_VS",
        "OOP_VS",
        "ACTION_HERO",
    ]

    return pd.read_table(
        filepath_or_buffer=diretorio, header=None, names=nomes, sep=" "
    )


def classificar(i_tr, i_te, o_tr, o_te, metodos):
    metodos[1].fit(i_tr, o_tr)
    predicao = metodos[1].predict(i_te)
    acuracia = accuracy_score(o_te, predicao)
    print(metodos[0], " Acc = ", acuracia)


def benchmark(metodos, proporcao, quant_iter):
    dados = load_data(path)
    input = dados.drop("ACTION_HERO", axis=1)
    output = dados.ACTION_HERO
    input_treino, input_teste, output_treino, output_teste = train_test_split(
        input, output, train_size=proporcao
    )

    for met in metodos:
        classificar(input_treino, input_teste, output_treino, output_teste, met)


if __name__ == "__main__":
    metodos = [
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVC", SVC()),
        ("Arvore", DecisionTreeClassifier()),
        (
            "MLP",
            MLPClassifier(
                solver="lbfgs",
                alpha=1e-5,
                hidden_layer_sizes=(5, 2),
                random_state=1,
                max_iter=1000,
            ),
        ),
        ("Naive Bayes", GaussianNB()),
    ]

    benchmark(metodos, 2 / 3, 3)
