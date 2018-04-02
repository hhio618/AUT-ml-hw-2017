from base.top_down_kmeans import TDKMeans
from data import data
import os
import matplotlib
import numpy as np

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
    X = data.load_data2()
    m = len(X)
    measure = TDKMeans.SSE()
    model = TDKMeans(10, 3000, measure)
    snapshots = model.fit(X)
    report = "Time,Loss(%s),#Clusters\n" % measure
    measures = list()
    for snap in snapshots:
        snap.save_fig(X, plt)
        report += "%d,%f,%d\n" % (snap.t, snap.measure, len(np.unique(snap.C)))
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        measures.append(snap.measure)

    plt.figure()
    plt.plot(measures)
    plt.title("Elbow figure")
    plt.xlabel("Level")
    plt.ylabel("Cluster goodness")
    plt.savefig("outputs%smain2%s%s%selbow.png" % (os.sep, os.sep, measure, os.sep))
    # write report to output
    with open("outputs%smain2%s%s%stpkmeans.csv" % (os.sep, os.sep, measure, os.sep), 'w') as outf:
        outf.write(report)
        print "Report created in %s" % outf.name
        outf.flush()
        outf.close()
