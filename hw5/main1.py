import sys

from base.dbscan import DBScan
from base.kmeans import KMeans
from data import data
import os
import matplotlib

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
    X = data.load_data1()
    m = len(X)

    # kmeans clustering
    # cross validation list
    list_k = [2, 3, 4, 5]
    report = "K, DBI\n"
    best_k = None
    dbi_min = sys.maxsize
    for i, k in enumerate(list_k):
        # train model
        model = KMeans(k, 2000)
        model.fit(X)
        # plot lifted data and support vectors
        dbi = model.dbi()
        if dbi < dbi_min:
            best_k = k
            dbi_min = dbi
        report += "%d,%f\n" % \
                  (k, dbi)
    report += "best(k),%d" % best_k
    # test against best k
    model = KMeans(best_k, 4000)
    model.fit(X)
    plt.figure(1)
    model.plot(plt)
    plt.savefig("outputs%smain1%skmeans.png" % (os.sep, os.sep))

    print report
    with open("outputs%smain1%skmeans.csv" % (os.sep, os.sep), 'w') as outf:
        outf.write(report)
        print "Report created in %s" % outf.name
        outf.flush()
        outf.close()

    # dbscan clustering
    # test against best k
    model = DBScan()
    # Find good hyper parameters
    eps, minPTs = DBScan.hyper_tuner(plt, X)

    report = "Epsilon,Min points\n%f,%d" % (eps, minPTs)
    with open("outputs%smain1%sdbscan.csv" % (os.sep, os.sep), 'w') as outf:
        outf.write(report)
        print "Report created in %s" % outf.name
        outf.flush()
        outf.close()

    model.fit(X, eps, minPTs)
    plt.figure()
    model.plot(plt)
    plt.tight_layout()
    plt.savefig("outputs%smain1%sdbscan.png" % (os.sep, os.sep))
    plt.show()
