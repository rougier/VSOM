import gs
import numpy as np
import matplotlib.pylab as plt


def fancy_plot(y, ax, color='C0', label='', alpha=0.3):
    n = y.shape[0]
    x = np.arange(n)
    xleft = x - 0.5
    xright = x + 0.5
    X = np.array([xleft, xright]).T.flatten()
    Xn = np.zeros(X.shape[0] + 2)
    Xn[1:-1] = X
    Xn[0] = -0.5
    Xn[-1] = n - 0.5
    Y = np.array([y, y]).T.flatten()
    Yn = np.zeros(Y.shape[0] + 2)
    Yn[1:-1] = Y
    ax.bar(x, y, width=1, alpha=alpha, color=color, edgecolor=color)
    ax.plot(Xn, Yn, c=color, label=label, lw=3)


if __name__ == '__main__':
    cp = np.load("./data/experiment-2-bis-regular.npy")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    rltx = gs.rlts(cp, n=100, L_0=32, i_max=10, gamma=1.0/8)
    mrltx = np.mean(rltx, axis=0)
    fancy_plot(mrltx, ax, color='k', label='Regular', alpha=0.3)

    cp = np.load("./data/experiment-2-bis-random.npy")
    rltx = gs.rlts(cp, n=100, L_0=32, i_max=10, gamma=1.0/8)
    mrltx = np.mean(rltx, axis=0)
    fancy_plot(mrltx, ax, color='b', label='Random', alpha=0.3)

    seed = 12345
    np.random.seed(seed)
    n = 500
    X = np.random.uniform(0, 1, n)
    Y = np.random.uniform(0, 1, n)
    holes = 64
    for i in range(holes):
        x, y = np.random.uniform(0.1, 0.9, 2)
        r = 0.1 * np.random.uniform(0, 1)
        Inp = ((X-x)**2 + (Y-y)**2) > r*r
        X, Y = X[Inp], Y[Inp]
    X = np.c_[X, Y]
    rltx = gs.rlts(X, n=100, L_0=32, i_max=10, gamma=1.0/8)
    mrltx = np.mean(rltx, axis=0)
    fancy_plot(mrltx, ax, color='orange', label='Input Space', alpha=0.3)

    ax.legend()
    plt.show()
