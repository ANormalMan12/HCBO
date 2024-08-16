
def calculate_ECCIS_accuracy(
    oracle_ECCIS,
    fitted_ECCIS):
    def calc_set_accuracy(true_set,pred_set):
        right_num=0
        for i in range(len(true_set)):
            if(pred_set[i] in true_set):
                right_num+=1
        return right_num/len(true_set)
    return [calc_set_accuracy(oracle_ECCIS[i],fitted_ECCIS[i]) for i in range(len(oracle_ECCIS))]
    