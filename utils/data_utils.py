import numpy as np
import pandas as pd
import random

def read_data(path: str = "./data/data_1.csv") -> tuple:
    """ Funkcija za ucitavanje podataka """

    # Ucitavanje podataka
    Xy = pd.read_csv(path).to_numpy()

    # Razdvajanje ulaza i izlaza
    X, y = Xy[:,:-1], Xy[:,-1]#.reshape(-1,1)

    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, train_ratio: int = 0.8, random_state: int = 1234) -> tuple:
    """ Funkcija za podelu podataka na obucavajuci i testirajuci skup """
    # Seme generatora slucajnih brojeva
    random.seed(random_state)

    # Indeksi
    ind = [i for i in range(X.shape[0])]

    # Broj podataka za obucavanje
    n_train = int(len(ind) * train_ratio)

    # Odabiranje podataka za obucavanje
    ind_train = random.sample(ind, n_train)
    ind_test = [i for i in ind if i not in ind_train]

    X_train, y_train = X[ind_train], y[ind_train].reshape(-1,1)
    X_test, y_test = X[ind_test], y[ind_test].reshape(-1,1)

    assert (X_train.shape[0] + X_test.shape[0]) == X.shape[0]
    assert (y_train.shape[0] + y_test.shape[0]) == y.shape[0]
    assert set(ind_train + ind_test) == set(ind)

    return X_train, y_train, X_test, y_test


def cv_split(X: np.ndarray, y: np.ndarray, n_folds: int = 4, random_state: int = 1234) -> list:
    """ Podela podataka u strukove """

    # Seme generatora slucajnih brojeva
    np.random.seed(random_state)

    # Pomocna lista strukova
    folds_joint_list = []
    # Konacan lista strukova
    folds = []
    # Lista indeksa
    ind = [i for i in range(X.shape[0])]

    for y_c in [0, 1]:
        # Indeksi trenutne klase

        ind_ = list(filter(lambda x: y[x] == y_c, ind))
        # Izdvajanje podatak trenutne klase
        X_, y_ = X[ind_], y[ind_]
        # Spajanje ulazno-izlaznih parova
        Xy = np.concatenate((X_, y_), axis=1)
        # Podela na strukove
        Xy_folds = np.array_split(Xy, n_folds)
        # Dodavanje u listu
        folds_joint_list.append(Xy_folds)

    for i in range(n_folds):

        # Izdvajanje ulazno izlaznih parova obe klase
        Xy1 = folds_joint_list[0][i]
        Xy2 = folds_joint_list[1][i]
        # Spajanje klasa
        Xy = np.concatenate((Xy1, Xy2), axis=0)
        # Mesanje podataka
        np.random.shuffle(Xy)
        # Razdvajanje ulazno-izlaznih parova
        X, y = Xy[:,:-1], Xy[:,-1].reshape(-1,1)
        folds.append((X, y))

    return folds


def merge_folds(folds):
    """ Funkcija za spajanje strukova """

    X = np.concatenate(list(map(lambda x: x[0], folds)), axis=0)
    y = np.concatenate(list(map(lambda x: x[1], folds)), axis=0)

    return X, y


class StandardScaler():

    def __init__(self):
        """ Inicijalizaca srednje vrednosti i standardne devijacije """
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> None:
        """ Racunanje statistickih parametara matrice prediktora """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Standardizacija ulazne matrice prediktora """

        # Provera dimenzija
        assert self.mean.size == X.shape[1], \
        f"Ocekivani broj prediktora je {self.mean.size}, dobijeno je {X.shape[1]}"

        # Standardizacija
        X_ = (X - self.mean) / self.std

        return X_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """ Ucenje statistickih parametara i transformacija matrice prediktora """

        # Ucenje statistickih parametara
        self.fit(X)

        # Primena naucenih parametara na prosledjeni skup podataka
        X_ = self.transform(X)

        return X_

    def get_mean(self) -> np.ndarray:
        """ Metoda za dohvatanje srednje vrednosti prediktora """
        return self.mean

    def get_std(self) -> np.ndarray:
        """ Metoda za dohvatanje standardne devijacije prediktora """
        return self.mean

    def get_params(self) -> tuple:
        """ Metoda za dohvatanje statistickih parametara """
        return self.get_mean(), self.get_params()