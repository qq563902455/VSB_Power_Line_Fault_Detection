import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


def mcc_metric(y_true, y_pred_proba):

    best_score = 0
    best_threshold = 0

    for threshold in [val/100 for val in range(100)]:

        y_pred = y_pred_proba > threshold

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        fp = np.sum((y_true == 1) & (y_pred == 0))
        fn = np.sum((y_true == 0) & (y_pred == 1))

        if tp == 0 or tn == 0: continue

        score = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        if score > best_score:
            best_score = score
            best_threshold = threshold


    return best_score, best_threshold

class threePhasesModel:
    def __init__(self, modellist, modelParamList, phaseList):

        self.modellist = modellist
        self.modelParamList = modelParamList
        self.phaseList = phaseList

    def fit(self, train, kfold, train_epochs_list, batch_size_list,
            test=None,
            test_epochs=None,
            test_batch_size=None,
            debug_phase=None):

        seed_start = 10086
        seed_step = 500

        self.fittedModelslist = [[], [], []]
        self.thresholddict = {}

        train_total_preds = []
        train_total_target = []

        if debug_phase is not None:
            phaseList = [debug_phase]
        else:
            phaseList = self.phaseList

        for phase in phaseList:
            train_x = train[train.phase==phase].features
            train_x = np.array(train_x.tolist())
            train_y = train[train.phase==phase].target.values

            train_preds = np.zeros((len(train_x)))

            for i, (train_idx, valid_idx) in enumerate(kfold.split(train_x, train_y)):
                x_train_fold = train_x[train_idx]
                y_train_fold = train_y[train_idx, np.newaxis]
                x_val_fold = train_x[valid_idx]
                y_val_fold = train_y[valid_idx, np.newaxis]

                model = self.modellist[phase](seed_start+i*seed_step, **self.modelParamList[phase])
                model = model.cuda()

                train_epochs = train_epochs_list[phase]
                batch_size = batch_size_list[phase]

                if test is not None:

                    test_x = np.array(test[test.phase==phase].features.tolist())
                    test_y = np.array(test[test.phase==phase].target.tolist())

                    model.unsupervisedTraining(test_x, test_y,
                                               test_epochs, test_batch_size,
                                               0.7,
                                               plot_fold='_unsupervision_fold'+str(i))
                model.fit(x_train_fold, y_train_fold, train_epochs, batch_size, x_val_fold, y_val_fold,
                          custom_metric=roc_auc_score, plot_fold='_fold'+str(i))

                train_preds[valid_idx] = model.predict_proba(x_val_fold)
                score, threshold = mcc_metric(train_y[valid_idx], train_preds[valid_idx])
                print('fold', str(i), 'score:\t', score, 'threshold:\t', threshold)

                self.fittedModelslist[phase].append(model)

            score, threshold = mcc_metric(train_y, train_preds)
            self.thresholddict[phase] = threshold

            train_total_preds.append(train_preds)
            train_total_target.append(train_y)
            print('phase', phase, 'score:\t', score, 'threshold:\t', threshold)
            print('--'*20)
        print('=='*20)
        self.train_total_preds = np.concatenate(train_total_preds)
        self.train_total_target = np.concatenate(train_total_target)
        print('finished')

    def predict_proba(self, test, proba=False):
        answerlist = []
        for phase in self.phaseList:
            test_x = test[test.phase==phase].features
            test_x = np.array(test_x.tolist())

            test_id = test[test.phase==phase].signal_id

            test_preds = np.zeros((test_x.shape[0], len(self.fittedModelslist[phase])))
            for i in range(test_preds.shape[1]):
                model = self.fittedModelslist[phase][i]
                test_preds[:, i] = model.predict_proba(test_x)
            print('test corr:')
            print(pd.DataFrame(test_preds).corr().values.mean())
            test_preds = test_preds.mean(axis=1)

            if proba:
                answer = pd.DataFrame({'signal_id':test_id, 'target':test_preds>self.thresholddict[phase]})
            else:
                answer = pd.DataFrame({'signal_id':test_id, 'target':test_preds})
            answerlist.append(answer)
        result = pd.concat(answerlist)
        return result

    def predict(self, test):
        return self.predict_proba(test, proba=True)
