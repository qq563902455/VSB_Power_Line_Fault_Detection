import pandas as pd
import math

class StratifiedKFold:
    def __init__(self, id_measurement,
                 n_splits, shuffle=False, random_state=2019):
        self.n_splits = n_splits
        self.random_state = 2019
        self.id_measurement = id_measurement.reset_index(drop=True)

    def split(self, x, y):
        dataXy = pd.DataFrame({
            'id_measurement': self.id_measurement,
            'target': y}
            )
        targetCount = dataXy.groupby(by=['id_measurement']).sum()

        valid_id_listlist = []
        for i in range(self.n_splits):
            valid_id_listlist.append([])

        for i in targetCount.target.unique():
            total_num = (targetCount.target==i).sum()

            all_id = targetCount[targetCount.target==i].sample(
                        frac=1.0, random_state=self.random_state).index

            slice_size = math.ceil(total_num/self.n_splits)

            for j in range(self.n_splits):
                valid_id_listlist[j].extend(
                    all_id[(j*slice_size):((j+1)*slice_size)])

        resultlist = []

        for valid_id_list in valid_id_listlist:

            valid_iloc = self.id_measurement[self.id_measurement.apply(
                                lambda x: x in valid_id_list)].index
            train_iloc = self.id_measurement[self.id_measurement.apply(
                                lambda x: x not in valid_id_list)].index
            resultlist.append((list(train_iloc), list(valid_iloc)))

        return resultlist
