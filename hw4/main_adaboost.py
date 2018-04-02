import os
import numpy as np
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


def plot_stump(S, h, size=None, signed=False):
    plot_colors = "br"
    plot_step = 0.02
    class_names = "+-"
    X = np.array([s.x for s in S])
    y = np.array([s.y for s in S])
    W = np.array([s.w for s in S])
    # Plot the decision boundaries
    x_min, x_max = X[:, 0].min(axis=0) - 1, X[:, 0].max(axis=0) + 1
    y_min, y_max = X[:, 1].min(axis=0) - 1, X[:, 1].max(axis=0) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = h.predict_bulk(np.c_[xx.ravel(), yy.ravel()], signed)
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.BuGn)
    plt.colorbar()
    plt.axis("tight")

    # Plot the training points
    for i, n, c in zip(range(2), class_names, plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1],
                    c=c, cmap=plt.cm.BuGn,
                    s=20,
                    label="Class %s" % n)
    selector = lambda case, x, y: x if case.y == 1 else y
    for example in S:
        if size is None:
            s = int(300 * example.w)
        else:
            s = 35
        plt.scatter(example.x[0], example.x[1], c=selector(example, 'b', 'r'), s=s,
                    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right', fancybox=True, framealpha=0.3, prop={'size': 10})
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(h, fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)


class Example:
    # x is a vector, y in {-1,1}, D is weight
    def __init__(self, x, y, w):
        self.x, self.y, self.w = x, y, w


class DecisionStump:
    # sgn(a'x > b), a is a unit vector, b is a scalar
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __str__(self):
        var = 'x1' if self.a[0] != 0 else 'x2'
        ineq = '>' if sum(self.a) > 0 else '<'
        b = self.b if sum(self.a) > 0 else -self.b
        return 'sgn(%2s %s %3.1f)' % (var, ineq, b)

    # prediction on X
    def predict_bulk(self, X, signed=None):
        Z = list()
        for x in X:
            Z.append(self.predict(x))
        return np.asarray(Z)

    # prediction on x
    def predict(self, x):
        return 1 if np.inner(self.a, x) > self.b else -1

    # err rate of h on D
    def err(self, S):
        return sum([s.w * (s.y * self.predict(s.x) < 0) for s in S])


class Ensemble:
    # sgn(sum(alpha * h))
    def __init__(self):
        self.h, self.alpha = list(), list()

    # add hypothesis h with weight alpha
    def add(self, h, alpha):
        self.h.append(h)
        self.alpha.append(alpha)

    # prediction on X
    def predict_bulk(self, X, signed=False):
        Z = list()
        for x in X:
            Z.append(self.predict(x, signed))
        return np.asarray(Z)

    # prediction on x
    def predict(self, x, signed=False):
        T = len(self.h)
        fx = sum([self.alpha[t] * self.h[t].predict(x) for t in range(T)])
        if signed:
            return 1 if fx > 0 else -1
        return fx

    def __str__(self):
        out = "H(x) = \n"
        for t in range(len(self.h)):
            out += "%.2f.%s +" % (self.alpha[t], self.h[t])
        return out[:-1] + ' \n'

    # err rate of ensemble on S (unweighted)
    def err(self, S):
        return sum([1.0 / len(S) * (s.y * self.predict(s.x) < 0) for s in S])


# enumerate and pick the best hypothesis in C
def weaklearn(S, C):
    errs = [h.err(S) for h in C]
    index = np.argmin(errs)
    return C[index], errs[index]


# main adaboost algorithm
def adaboost(S, C, T):
    report = '%12s,%16s,%12s,%12s,%70s,%12s,%12s' % ('t', 'h', 'eps', 'alpha', 'W', 'Z', 'err(H)')
    H = Ensemble()
    for t in range(T):
        h, eps = weaklearn(S, C)
        alpha = 0.5 * np.log((1 - eps) / eps)
        H.add(h, alpha)
        Z = 2 * np.sqrt(eps * (1 - eps))
        report += '\n%12s,%16s,%12.3f,%12.3f,%70s,%12.3f,%12.3f' % (
            t + 1, h, eps, alpha, ' '.join(['%6.3f' % s.w for s in S]), Z, H.err(S))
        # update weights
        for s in S:
            s.w = s.w / Z * np.exp(- alpha * s.y * h.predict(s.x))
        plt.subplot(221 + t)
        plot_stump(S, h)
    return H, report


if __name__ == '__main__':
    # all decision stumps
    C = list()
    for sign in [-1, 1]:
        for a in [np.array([sign, 0]), np.array([0, sign])]:
            for b in sign * np.arange(-1.5, 1.5, 0.3):  # [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
                C.append(DecisionStump(a, b))
    # data
    m = 8
    S = list()
    S.append(Example(np.array([-1., 0.]), 1, 1.0 / m))
    S.append(Example(np.array([-.5, .5]), 1, 1.0 / m))
    S.append(Example(np.array([0., 1.]), -1, 1.0 / m))
    S.append(Example(np.array([.5, 1.]), -1, 1.0 / m))
    S.append(Example(np.array([1., 0.]), 1, 1.0 / m))
    S.append(Example(np.array([1., -1.]), 1, 1.0 / m))
    S.append(Example(np.array([0., -1.]), -1, 1.0 / m))
    S.append(Example(np.array([0., 0.]), -1, 1.0 / m))

    plt.figure()
    H, report = adaboost(S, C, 3)
    print report
    with open("outputs%sadaboost%soutput.csv" % (os.sep, os.sep), 'w') as outf:
        outf.write(report)
        print "Report created at %s" % outf.name
        outf.flush()
        outf.close()

    # plot decision boundary stump for H
    plt.subplot(224)
    plot_stump(S, H, 1. / m)
    # plot signed decision boundary for H
    plt.savefig("outputs%sadaboost%sstumps.png" % (os.sep, os.sep))
    plt.figure()
    plot_stump(S, H, 1. / m, True)
    plt.savefig("outputs%sadaboost%sensemble.png" % (os.sep, os.sep))
    plt.show()
