from copy import copy
from random import sample, seed

from base.svm_classifier import SVM
from data import data
import numpy as np
import matplotlib
import os
from libsvm.python.svm import RBF, POLY, SIGMOID

gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        print "Testing matplotlib backend...", gui
        matplotlib.use(gui, warn=False, force=True)
        matplotlib.interactive(False)
        from matplotlib import pyplot as plt

        break
    except Exception as e:
        continue
seed(0)


class ModelRecord:
    def __init__(self, acc, params, model):
        self.sv_count, self.acc_test, self.acc, self.params, self.model = None, None, acc, params, model

    def __repr__(self):
        return "%f, %f, %d, \"%s\"" % (
            self.acc_test, self.acc, self.sv_count, self.params)


class ParamTuner:
    iter_gama = range(-15, 3)
    iter_c = range(-3, 7)
    iter_degree = range(1, 5)
    iter_coef0 = range(-3, 3)

    def __init__(self, X_tr, y_tr, X_va, y_va, X_te, y_te):
        self.X_tr, self.y_tr, self.X_va, self.y_va, self.X_te, self.y_te = X_tr, y_tr, X_va, y_va, X_te, y_te

    def _evaluate_rbf(self, kernel_type, c, gama, degree=None, coef0=None):
        model_kernel = SVM()
        acc_kernel = 0
        if kernel_type == RBF:
            acc_kernel = model_kernel.fit_validate(self.X_tr, self.y_tr, self.X_va, self.y_va, RBF, c,
                                                   extra_options='-g %f' % gama)
        elif kernel_type == POLY:
            acc_kernel = model_kernel.fit_validate(self.X_tr, self.y_tr, self.X_va, self.y_va, POLY, c,
                                                   extra_options='-g %f -r %f -d %d'
                                                                 % (
                                                                     gama, coef0,
                                                                     degree))
        elif kernel_type == SIGMOID:
            acc_kernel = model_kernel.fit_validate(self.X_tr, self.y_tr, self.X_va, self.y_va, POLY, c,
                                                   extra_options='-g %f -r %f'
                                                                 % (gama,
                                                                    coef0))

        params = {'c': c, 'gama': gama}
        if degree is not None:
            params['degree'] = degree
        if coef0 is not None:
            params['coef0'] = coef0
        return ModelRecord(acc_kernel, params, model_kernel)

    def tune(self, kernel_type):
        best_model_record = ModelRecord(0, None, None)
        record_list = []
        if kernel_type == RBF:
            for c in ParamTuner.iter_c:
                for gama in ParamTuner.iter_gama:
                    try:
                        model_record = \
                            self._evaluate_rbf(kernel_type, 2 ** c, 2 ** gama)
                        if model_record.acc >= best_model_record.acc:
                            best_model_record = model_record
                        record_list.append(model_record)
                    except ValueError as e:
                        pass

        elif kernel_type == POLY:
            for c in ParamTuner.iter_c:
                for gama in ParamTuner.iter_gama:
                    for coef0 in ParamTuner.iter_coef0:
                        for degree in ParamTuner.iter_degree:
                            try:
                                model_record = \
                                    self._evaluate_rbf(kernel_type, 2 ** c, 2 ** gama, degree, coef0)
                                if model_record.acc >= best_model_record.acc:
                                    best_model_record = model_record
                                record_list.append(model_record)
                            except ValueError as e:
                                pass
        elif kernel_type == SIGMOID:
            for c in ParamTuner.iter_c:
                for gama in ParamTuner.iter_gama:
                    for coef0 in ParamTuner.iter_coef0:
                        try:
                            model_record = \
                                self._evaluate_rbf(kernel_type, 2 ** c, 2 ** gama, coef0=coef0)
                            if model_record.acc >= best_model_record.acc:
                                best_model_record = model_record
                            record_list.append(model_record)
                        except ValueError as e:
                            pass
        random_records = sample(record_list, 20)
        for record in [best_model_record] + random_records:
            record.acc_test = record.model.predict(self.X_te, self.y_te)
            record.sv_count = len(record.model.model.get_SV())
        return best_model_record, self._fine_tune(kernel_type, best_model_record), random_records

    def _fine_tune(self, kernel_type, coarse_model_record):
        parms = coarse_model_record.params
        iter_c = np.linspace(parms['c'] - 10, parms['c'] + 10, 20)
        iter_gama = np.linspace(parms['gama'] - 10, parms['gama'] + 10, 15)
        best_model_record = copy(coarse_model_record)

        if kernel_type == RBF:
            for c in iter_c:
                for gama in iter_gama:
                    try:
                        model_record = \
                            self._evaluate_rbf(kernel_type, c, gama)
                        if model_record.acc >= best_model_record.acc:
                            best_model_record = model_record
                    except ValueError as e:
                        pass

        elif kernel_type == POLY:
            iter_coef0 = np.linspace(parms['coef0'] - 1, parms['coef0'] + 1, 5)
            iter_degree = np.linspace(parms['degree'] - 1, parms['degree'] + 1, 5)
            for c in iter_c:
                for gama in iter_gama:
                    for coef0 in iter_coef0:
                        for degree in iter_degree:
                            try:
                                model_record = \
                                    self._evaluate_rbf(kernel_type, c, gama, degree, coef0)
                                if model_record.acc >= best_model_record.acc:
                                    best_model_record = model_record
                            except ValueError as e:
                                pass
        elif kernel_type == SIGMOID:
            iter_coef0 = np.linspace(parms['coef0'] - 1, parms['coef0'] + 1, 5)
            for c in iter_c:
                for gama in iter_gama:
                    for coef0 in iter_coef0:
                        try:
                            model_record = \
                                self._evaluate_rbf(kernel_type, c, gama, coef0=coef0)
                            if model_record.acc >= best_model_record.acc:
                                best_model_record = model_record
                        except ValueError as e:
                            pass

        best_model_record.acc_test = best_model_record.model.predict(self.X_te, self.y_te)
        best_model_record.sv_count = len(best_model_record.model.model.get_SV())
        return best_model_record


def write_report(report, fname):
    out = "Model, Acc(Test), Acc(Validation), #SV, parameters\n"
    out += "Best model(Coarse grain), %s\n" % report[0]
    out += "Best model(Fine grain), %s\n" % report[1]
    for i, record in enumerate(report[2]):
        out += "Random model(%d),%s\n" % (i + 1, record)
    with open(fname, 'w') as outf:
        outf.write(out)
        print "Report created in %s" % outf.name
        outf.flush()
        outf.close()


if __name__ == '__main__':
    X_tr, y_tr, X_va, y_va, X_te, y_te = data.load_parkinson()

    # tune parameters
    rbf_report = ParamTuner(X_tr, y_tr, X_va, y_va, X_te, y_te).tune(RBF)
    poly_report = ParamTuner(X_tr, y_tr, X_va, y_va, X_te, y_te).tune(POLY)
    sig_report = ParamTuner(X_tr, y_tr, X_va, y_va, X_te, y_te).tune(SIGMOID)

    # write reports to output
    write_report(rbf_report, "outputs%sparkinson%srbf.csv" % (os.sep, os.sep))
    write_report(poly_report, "outputs%sparkinson%spoly.csv" % (os.sep, os.sep))
    write_report(sig_report, "outputs%sparkinson%ssigmoid.csv" % (os.sep, os.sep))

    # test C extreme values
    X = np.r_[X_tr, X_te, X_va]
    y = np.r_[y_tr, y_te, y_va]
    # use rbf best gama
    best_gama = rbf_report[1].params['gama']
    # test on huge C
    acc_huge_c = SVM().cross_validation(X, y, RBF, 2 ** 26,
                                        extra_options='-g %d' % 2 ** best_gama)
    # test on small C
    acc_small_c = SVM().cross_validation(X, y, RBF, 2 ** -20,
                                         extra_options='-g %d' % 2 ** best_gama)
    print "Cross Validation Acc(RBF) for C:2^26 ==>", acc_huge_c
    print "Cross Validation Acc(RBF) for C:2^-20 ==>", acc_small_c
