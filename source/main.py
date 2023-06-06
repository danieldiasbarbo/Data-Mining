import pandas as pd
import os.path

if __name__ == "__main__":
    diretorio = os.path.join(os.path.dirname(__file__), "../data/ZP-River.dat")

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

    dados = pd.read_table(
        filepath_or_buffer=diretorio, header=None, names=nomes, sep=" "
    )

    print(dados.head(10))
