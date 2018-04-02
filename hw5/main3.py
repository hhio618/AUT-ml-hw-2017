from base.bottom_up import AgglomerativeClustering
from base.utils import *
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
    X = data.load_data2()
    m = len(X)
    # iterate over linkage
    for linkage_measure in [AverageLinkage(), SingleLinkage(), CompleteLinkage()]:
        model = AgglomerativeClustering(10, linkage_measure)
        snapshots = model.fit(X)
        report = "#Clusters,Linkage(%s)\n" % linkage_measure
        for snap in snapshots:
            snap.save_fig(X, plt)
            report += "%d,%f\n" % (snap.num_clusters, snap.linkage_measure)

        # write report to output
        with open("outputs%smain3%s%s%sreport.csv" % (os.sep, os.sep, linkage_measure, os.sep), 'w') as outf:
            outf.write(report)
            print "Report created in %s" % outf.name
            outf.flush()
            outf.close()
