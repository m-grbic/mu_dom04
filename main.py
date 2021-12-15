from utils.data_utils import *
from utils.model_utils import *


# Čitanje podataka
X, y = read_data('./data/data_2.csv')
# Podala podataka na obučavajući i testirajući/validacioni skup
X_train, y_train, X_test, y_test = train_test_split(X, y)


params_dict = {
    "max_depth": [3, 5, 10],
    "criterion": ['gini','entropy']
}

def get_combinations(d: dict) -> list:
    """ Reformatiranje hiper-parametara"""
    
    rep_elem = [1]
    num_combinations = len(list(d.values())[0])
    for v in reversed(list(d.values())[1:]):
        num_combinations *= len(v)
        rep_elem.append(rep_elem[-1]*len(v))
    rep_elem.reverse()
    # Generisanje kombinacija
    d1 = {item[0]: np.repeat(np.array(item[1]),rep).tolist()*int(num_combinations // (len(item[1] * rep))) for item, rep in zip(d.items(), rep_elem)}
    # Lista imena hiper-parametara
    prams_names = list(d.keys())
    # Izdjavanje jednog skupa hiper-parametara
    get_one = lambda x: {name: d1[name][x] for name in prams_names}
    # Indeksi kombinacija
    inds = [i for i in range(num_combinations)]
    params_list = list(map(lambda x: get_one(x), inds))

    return params_list



params_list = get_combinations(params_dict)

n_estimators_list = [i*10 for i in range(1, 15)]
analyze_ensemble_size((X_train, y_train, X_test, y_test), 'rf', params_list, n_estimators_list)
