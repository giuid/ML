import pandas as pd

"""
Metodo per caricare il CUP dataset dalla sotto cartella /dataset/CUP
"""
def load_cup(path):
    header_list = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15",
                   "p16", "p17", "p18", "p19", "p20", "out_x", "out_y"]
    tr = pd.read_csv(path + "/dataset/CUP/ML-CUP19-TR.csv", names=header_list, skiprows=7)
    test = pd.read_csv(path + "/dataset/CUP/ML-CUP19-TS.csv", names=header_list[:20], skiprows=7)
    return tr, test

"""
Metodo per caricare i dataset MONK dalla sotto cartella /dataset/MONK
"""

def load_monk(path_folder, ver):
    assert (3 >= ver > 0), "Le versioni disponibili del MONK sono 1, 2, 3 !"
    header_list = ["class", "a1", "a2", "a3", "a4", "a5", "a6", "Id"]
    tr = pd.read_csv(path_folder + "/dataset/MONK/monks-" + str(ver) + ".train", sep=" ", names=header_list, usecols=range(1, 9))
    ts = pd.read_csv(path_folder + "/dataset/MONK/monks-" + str(ver) + ".test", sep=" ", names=header_list, usecols=range(1, 9))

    return tr, ts
