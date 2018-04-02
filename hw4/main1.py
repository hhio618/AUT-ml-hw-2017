from base.kernels import *
from base.svm_classifier import SVM
from data import data
import numpy as np
import os
import matplotlib

from data.data import normalize
from libsvm.python.svm import PRECOMPUTED

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

if __name__ == '__main__':
    X, y = data.load_data1()
    m = len(X)

    # cross validation list
    list_phi = [PhiIdentity(), Phi1(), Phi2(), Phi3()]
    report = "Phi, Acc(Linear), Acc(Kernel), Function\n"
    for i, phi in enumerate(list_phi):
        # lift data to phi space
        X_prime = normalize(phi(X))
        # train model
        model_linear = SVM(phi.__class__.__name__)
        acc_linear = model_linear.cross_validation(X_prime, y, 0, 10000)
        # plot lifted data and support vectors
        model_linear.plot_model()
        # use kernel matrices instead
        model_kernel = SVM()
        # compute kernel matrix
        kernel_matrix = X_prime.dot(X_prime.T).tolist()
        # save kernel to output
        np.savetxt("outputs%smain1%s%s.csv" % (os.sep, os.sep, phi.__class__.__name__), kernel_matrix, fmt="%.4e")
        # train using kernel matrix
        kernel_matrix = [[i + 1] + item for i, item in enumerate(kernel_matrix)]
        acc_kernel = model_kernel.cross_validation(kernel_matrix, y, PRECOMPUTED, 10000)
        report += "%s, %s, %s, %s\n" % \
                  (phi.__class__.__name__, acc_linear, acc_kernel, phi)

    print report
    with open("outputs%smain1%soutput.csv" % (os.sep, os.sep), 'w') as outf:
        outf.write(report)
        print "Report created in %s" % outf.name
        outf.flush()
        outf.close()
    plt.show()
