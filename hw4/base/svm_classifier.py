import matplotlib

from libsvm.python.svm import PRECOMPUTED
import os
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
print "Using:", matplotlib.get_backend()
from mpl_toolkits.mplot3d import Axes3D
from libsvm.python import svmutil
import numpy as np


class SVM(object):
    def __init__(self, fig_name='dummy'):
        self.m, self.features = None, None
        self.W = None
        self.b = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.options = None
        self.fig_name = fig_name

    def fit_validate(self, X_tr, y_tr, X_va, y_va, kernel_type, c, extra_options=None):

        options = '-q -t %d -c %f ' % (kernel_type, c)
        options += extra_options if extra_options is not None else ""
        self.options = options
        model = svmutil.svm_train(y_tr.tolist(), X_tr.tolist(), options.rstrip())
        self.model = model
        return self.predict(X_va, y_va)

    def cross_validation(self, X, y, kernel_type, c, extra_options=None):
        if kernel_type != PRECOMPUTED:
            self.features = X.shape[1]
            X_list = X.tolist()
        else:
            X_list = X
        options = '-v 5 -q -t %d -c %f ' % (kernel_type, c)
        options += extra_options if extra_options is not None else ""
        self.options = options
        acc = svmutil.svm_train(y.tolist(), X_list, options.rstrip())
        # Line Parameters
        # W = np.matmul(X[np.array(model.get_sv_indices()) - 1].T, model.get_sv_coef())
        # b = -model.rho.contents.value
        # if model.get_labels()[1] == -1:  # No idea here but it should be done :|
        #     W = -W
        # b = -b
        self.X_train = X
        self.y_train = y
        # self.W = W
        # self.b = b
        # self.model = model
        # return
        return acc

    def plot_model(self):
        # Plotting
        fig = plt.figure()
        ax = None
        # for i in self.model.get_sv_indices():
        #     dp, project_3d = self.to_data_point(self.X_train[i - 1])
        #     if not project_3d:
        #         ax = fig.gca()
        #         ax.scatter(dp[:, 0], dp[:, 1], color='yellow', s=80)
        #     else:
        #         fig.add_subplot(111, projection='3d')
        #         ax = Axes3D(fig)
        #         ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], color='yellow', s=80)

        train = self.X_train
        dp, project_3d = self.to_data_point(train)
        if not project_3d:
            ax = fig.gca()
            ax.scatter(dp[:, 0], dp[:, 1], c=self.y_train)
        else:
            fig.add_subplot(111, projection='3d')
            ax = Axes3D(fig)
            ax.scatter(dp[:, 0], dp[:, 1], dp[:, 2], c=self.y_train)
        fig.savefig("outputs%smain1%s%s.png" % (os.sep, os.sep, self.fig_name))

    def predict(self, X, y):
        return svmutil.svm_predict(y.tolist(), X.tolist(), self.model)[1][0]

    def to_data_point(self, dp):
        project_3d = False
        if self.features == 1:
            dp = np.c_[dp, np.zeros(len(dp))]
        elif self.features == 3:
            project_3d = True
        return np.atleast_2d(dp), project_3d
