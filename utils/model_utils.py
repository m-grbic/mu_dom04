from unicodedata import name
from numpy.core.numeric import cross
import pandas as pd
import numpy as np
from .data_utils import cv_split, merge_folds, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
from copy import deepcopy



def corr_pred(X, y) -> list:
    """ F-ja za racunanje koeficijenta korelacije
    :param x: np.ndarray
    :param y: np.ndaray
    :returns: (list) sortirani prediktori
    """

    # "ravnanje" nize
    y_ = y.flatten()
    # Korelacije prediktora
    corr_coef = [np.correlate(X[:,ip], y_) for ip in range(X.shape[1])]
    # Indeksi prediktora
    pred_inds = [ip for ip in range(X.shape[1])]

    # Sortiranje
    corr_coef, pred_inds = (list(t) for t in zip(*sorted(zip(corr_coef, pred_inds))))

    # Sortiranje od najboljeg ka najgorem
    corr_coef.reverse()
    pred_inds.reverse()

    return pred_inds


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Funkcija za racunanje tacnosti """
    assert y.size == y_hat.size
    return sum(y.flatten() == y_hat.flatten()) / y.size


def error(y: np.ndarray, y_hat: np.ndarray) -> float:
    """ Funkcija za racunanje tacnosti """
    assert y.size == y_hat.size
    return sum(y.flatten() != y_hat.flatten())


def cross_validate(X: np.ndarray, y: np.ndarray, nfolds: int = 4) -> float:
    """ Funkcija za racunanje unakrsne validacije"""

    # Razbijanje skupa na strukove
    folds = cv_split(X, y, nfolds)
    # Lista tacnosti na obucavajucem i validacionom/testirajucem skipu
    train_acc, test_acc = [], []

    for i, fold in enumerate(folds):

        # Strukove za obucavanje
        train_folds_inds = list(filter(lambda x: x != i, [ind for ind in range(nfolds)]))
        train_folds = list(map(lambda x: folds[x], train_folds_inds))
        X_train, y_train = merge_folds(train_folds)
        # Struk za testiranje
        X_test, y_test = fold
        # "Ravnanje" tacnih izlaza
        y_train, y_test = y_train.flatten(), y_test.flatten()
        
        # Skaliranje podataka
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Kreiranje modela
        model = LogisticRegression(penalty='none', random_state=1234)
        # Obucavanje
        model.fit(X_train, y_train)
        # Predikcija
        y_train_hat, y_test_hat = model.predict(X_train), model.predict(X_test)
        # Racunanje tacnosti
        train_acc_, test_acc_ = accuracy(y_train, y_train_hat), accuracy(y_test, y_test_hat)
        
        # Cuvanje vrednosti na trenutnom struku
        train_acc.append(train_acc_)
        test_acc.append(test_acc_)

    # Konverzija list -> np.ndarray
    train_acc, test_acc = np.array(train_acc), np.array(test_acc)

    # Srednja vrednost greske
    train_mean_acc, test_mean_acc = np.mean(train_acc, axis=0), np.mean(test_acc, axis=0)
    # Standardna devijacija greske
    train_std_acc, test_std_acc = np.std(train_acc, axis=0), np.std(test_acc, axis=0)

    return train_mean_acc, test_mean_acc


def wrapper(X: np.ndarray, y: np.ndarray, nfolds: int = 4):
    """ Implementacija omotac algoritma """

    def add_pred(x: np.ndarray, i):
        """ f-ja za dodavanje prediktora """
        x_ = deepcopy(x)
        x_[i] = True
        return x_

    # Niz iskoriscenih prediktora
    used = np.zeros(X.shape[1], dtype=bool)
    # List sortiranih prediktora
    pred_sorted = []

    while np.sum(used) != used.size:

        # Kreiranje novih prediktora
        feature_combinations = [add_pred(used, i) for i in range(X.shape[1]) if not used[i]]
    
        # Kros validacija sa novim prediktorima
        res = list(map(lambda x: cross_validate(X[:,x], y, nfolds), feature_combinations))

        # Tacnosti na obucavajucem i validacionom skupu
        train_acc, val_acc = list(map(lambda x: x[0], res)), list(map(lambda x: x[1], res))

        # Prediktor sa najvecom tacnoscu na validacionom skupu
        ind_best = np.argmax(val_acc)
        best_combination = feature_combinations[ind_best]
        best_pred = np.argmin(best_combination == used)
        pred_sorted.append(best_pred)

        # Prepisivanje iskoriscenih prediktora
        used = best_combination

    return pred_sorted


def train_subsampled_predictors(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                           y_test: np.ndarray, pred_order: list, disp: bool = True) -> None:
    """ Obucavanje modela logisticke regresije na podskupovima prediktora """

    # Niz trenutno koriscenih prediktora
    predictors = np.zeros(X_train.shape[1], dtype=bool)

    # Skaliranje podataka
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # "Ravnanje" tacnih izlaza
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Lista za cuvanje tacnosti
    train_acc, test_acc = [], []

    for pred in pred_order:
        # Dodavanje prediktora
        predictors[pred] = True

        # Izdvajanje prediktora
        X_train_, X_test_ = X_train[:, predictors], X_test[:, predictors]

        # Kreiranje modela
        model = LogisticRegression(penalty='none', random_state=1234)
        # Obucavanje
        model.fit(X_train_, y_train)
        # Predikcija
        y_train_hat, y_test_hat = model.predict(X_train_), model.predict(X_test_)
        # Racunanje tacnosti
        train_acc_, test_acc_ = accuracy(y_train, y_train_hat), accuracy(y_test, y_test_hat)
        # Cuvanje vrednosti na trenutnom struku
        train_acc.append(train_acc_)
        test_acc.append(test_acc_)

    # Prikaz rezultata
    if disp:
        xaxis = np.arange(X_train.shape[1])
        fig = plt.figure(figsize=(16,6))
        plt.plot(xaxis, train_acc)
        plt.plot(xaxis, test_acc)
        plt.xticks(ticks=xaxis, labels=pred_order)
        plt.legend(['Obučavajući skup','Validacioni skup'])
        plt.title('Zavisnost tačnosti od izabranih prediktora')
        plt.ylabel('Tačnost')
        plt.grid(axis='y')
        plt.show()


def error_wrt_depth(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                    y_test: np.ndarray, depth_list: list, disp_error: bool = False,
                    disp_decision: bool = False, disp_tree: bool = False):

    def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                           y_test: np.ndarray, depth: int) -> tuple:
        # Kreiranje modela
        model = DecisionTreeClassifier(max_depth=depth, random_state=1234)
        # Obucavanje modela
        model.fit(X_train, y_train)
        # Predikcija
        y_train_hat, y_test_hat = model.predict(X_train), model.predict(X_test)
        # Procenat gresaka na obucavajucem i testirajucem/validacionom skupu
        train_err, test_err = error(y_train, y_train_hat), error(y_test, y_test_hat)

        if disp_decision:
            plot_step = 0.02
            samples_colors = 'bg'

            # Prikaz granice odlucivanja
            plt.subplots(figsize=(16,6))

            x_min, x_max = min(X_train[:,0].min(), X_test[:,0].min()) - 1, max(X_train[:,0].max(), X_test[:,0].max()) + 1
            y_min, y_max = min(X_train[:,1].min(), X_test[:,1].min()) - 1, max(X_train[:,1].max(), X_test[:,1].max()) + 1

            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.YlOrRd)

            # Prikaz primera
            for y_val in [0, 1]:
                # Indeksi trenutne klase
                ind_train, ind_test = (y_train == y_val).flatten(), (y_test == y_val).flatten()
                plt.scatter(X_train[ind_train,0], X_train[ind_train,1], c=samples_colors[y_val],
                            marker='o',label=f'Obučavajući skup y={y_val}')
                plt.scatter(X_test[ind_test,0], X_test[ind_test,1], c=samples_colors[y_val],
                            marker='x', label=f'Validacioni skup y={y_val}')
            plt.title(f'Prikaz granice odlučivanja za maksimalnu dubinu {depth}')
            plt.xlabel("$x_{9}$")
            plt.ylabel("$x_{12}$")
            plt.legend(loc='lower right')
            plt.show()

        if disp_tree and (depth == depth_list[-1]):
            plt.figure(figsize=(16,6))
            plot_tree(model, filled=True)
            plt.show()

        return train_err, test_err

    # Obucavanje i evaluacija modela
    res = list(map(lambda depth: train_and_evaluate(X_train, y_train, X_test, y_test, depth), depth_list))

    # Greske na obucavajucem i testirajucem skupu
    train_error = list(map(lambda x: x[0], res))
    test_error = list(map(lambda x: x[1], res))

    # Graficki prikaz zavisnosti
    if disp_error:
        fig = plt.figure(figsize=(16,6))
        plt.plot(depth_list, train_error)
        plt.plot(depth_list, test_error)
        plt.legend(['Obučavajući skup','Validacioni skup'])
        plt.title('Zavisnost pogrešno klasifikovanih primera od dubine stabla')
        plt.ylabel('Broj pogrešno klasifikovanih primera')
        plt.xlabel('Dubina stabla')
        plt.xticks(ticks=depth_list, labels=depth_list)
        plt.grid(axis='y')
        plt.show()


