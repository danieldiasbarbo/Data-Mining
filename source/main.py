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


def classificar(i_tr, i_te, o_tr, o_te, clas):
    clas.fit(i_tr, o_tr)
    predicao = clas.predict(i_te)
    acuracia = accuracy_score(output_teste, predicao)
    print(type(clas).__name__, " Acc = ", acuracia)


if __name__ == "__main__":
    dados = load_data(path)
    input = dados.drop("ACTION_HERO", axis=1)
    output = dados.ACTION_HERO

    input_treino, input_teste, output_treino, output_teste = train_test_split(
        input, output, train_size=2 / 3
    )

    classificar(
        input_treino,
        input_teste,
        output_treino,
        output_teste,
        KNeighborsClassifier(n_neighbors=5),
    )

    classificar(input_treino, input_teste, output_treino, output_teste, SVC())

    classificar(
        input_treino, input_teste, output_treino, output_teste, DecisionTreeClassifier()
    )

    classificar(
        input_treino,
        input_teste,
        output_treino,
        output_teste,
        MLPClassifier(
            solver="lbfgs",
            alpha=1e-5,
            hidden_layer_sizes=(50, 20),
            random_state=1,
            max_iter=10,
            verbose=False,
        ),
    )

    classificar(input_treino, input_teste, output_treino, output_teste, GaussianNB())
