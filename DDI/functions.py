import numpy as np
from keras.preprocessing import sequence

def inti_list(length): # initialize label list with all values as 'O'
    result = []
    for i in range(0, length):
        result.append('O')
    return result

def inti_list_zero(length): # initialize label list with all values as 0
    result = []
    for i in range(0, length):
        result.append(0)
    return result

def parselist(str):
    '''
    Parse list from string representation of list
    :param strlist: string
    :return:list
    '''
    return [w[1:-1] for w in str[1:-1].split(', ')]

def parselist_dis(str):
    '''
    Parse list from string representation of list
    :param strlist: string
    :return:list
    '''
    return [int(w) for w in str[1:-1].split(', ')]

def vectorize(listoftoklists,idxdict):
    '''
    Turn each list of tokens or labels in listoftoklists to an equivalent list of indices
    :param listoftoklists: list of lists
    :param idxdict: {tok->int}
    :return: list of np.array
    '''
    res = []
    for toklist in listoftoklists:
        res.append(np.array(map(lambda x: idxdict.get(x, idxdict['<UNK>']), toklist)).astype('int32'))
    return res

def vectorize_y(listoflabels,idxdict):
    '''
    Turn each list of tokens or labels in listoftoklists to an equivalent list of indices
    :param listoftoklists: list of lists
    :param idxdict: {tok->int}
    :return: list of np.array
    '''
    res = []
    for label in listoflabels:
        res.append(np.array(map(lambda label: idxdict.get(label, idxdict['<UNK>']),[label])).astype('int32'))
    return res
