import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def eigenvectors_check(eigenvectors, eigenvalues, covMatrix):
    print("Checking Eigenvectors reverse compatibility with covariance matrix...")
    eigenvalues2 = np.diag(eigenvalues)

    final_check = eigenvectors @ eigenvalues2 @ eigenvectors.T

    # This should be zero matrix
    test_substract = np.subtract(final_check.round(10), covMatrix.round(10))

    test_result = True
    for i in test_substract:
        for j in i:
            if j != 0:
                test_result = False

    if test_result:
        print("Eigenvectors: Check")
        print("Eigenvalues: Check")
    else:
        print("Eigenvectors: Error while reverse checking covariance")
        print("Eigenvalues: Error while reverse checking covariance")


def whitening_lambda(d_eigenvalues):
    for i in range(0, len(d_eigenvalues)):
        for j in range(0, len(d_eigenvalues[0])):
            if d_eigenvalues[i][j] != 0:
                d_eigenvalues[i][j] = d_eigenvalues[i][j] ** (-1 / 2)
    return d_eigenvalues


def pca(X, d, whitening=False):
    # Calculate covariance matrix
    cov = (X @ X.T) / len(X[0])

    # Calculate eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Check if the solution is correct
    eigenvectors_check(eigenvectors, eigenvalues, cov)

    # Choose d best results
    d_eigenvalues = eigenvalues[-d:]
    d_eigenvectors = eigenvectors[-d:]

    # Turn vector into diagonal matrix
    d_eigenvalues = np.diag(d_eigenvalues)

    # Without whitening
    Z = d_eigenvectors @ X

    # Whitening
    if whitening:
        d_eigenvalues_white = whitening_lambda(d_eigenvalues)
        Z = d_eigenvalues_white @ Z

    return Z, d_eigenvectors, d_eigenvalues


def scatter_plot(X, Y):
    x1 = X[0]
    y1 = X[1]
    x2 = Y[0]
    y2 = Y[1]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(x1, y1, s=10, c='blue', marker="o", label='non-whitened')
    ax1.scatter(x2, y2, s=10, c='red', marker="o", label='whitened')
    plt.legend(loc='lower left')
    plt.savefig('scatter_plot.jpg')


if __name__ == '__main__':
    iris = load_iris()

    # Data is grouped by values in columns
    # (so every sepal width is in first row and so on)
    iris_data = iris.data.T

    # Non-whitened data
    Y, V, Lambda = pca(iris_data, 2)
    # Whitened data
    # I could use pre-whitening Z, but where would I overheat my computer otherwise?
    Y1, V1, Lambda1 = pca(iris_data, 2, True)

    scatter_plot(Y, Y1)
