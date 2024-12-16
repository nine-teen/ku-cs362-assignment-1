import numpy as np

class KNeighborsClassifier:
  def __init__(self, n_neighbors=5, p=2):
    self.n_neighbors = n_neighbors
    self.p = p

  def __minkowski(self, x, z, p=None):
    if p is None:
      p = self.p

    return np.pow(np.sum(np.pow(np.abs(x - z), p)), 1/p)

  def fit(self, X, y):
    self._fit_X = X.astype(int)
    self.classes_, self._y = np.unique(y.astype(int), return_inverse=True)

    return self

  def kneighbors(self, X, n_neighbors=None, return_distance=True):
    if n_neighbors is None:
      n_neighbors = self.n_neighbors

    p = self.p
    _fit_X = self._fit_X

    distances = np.empty(_fit_X.shape[0], dtype=float)
    for ind, fit_X in enumerate(_fit_X):
      distances[ind] = self.__minkowski(X, fit_X)

    neigh_ind = np.argsort(distances)[:p]

    if return_distance:
      neigh_dist = distances[neigh_ind]

      return neigh_dist, neigh_ind
    return neigh_ind

  def predict(self, X):
    probabilities = self.predict_proba(X)

    return self.classes_[np.argmax(probabilities)]

  def predict_proba(self, X):
    classes_ = self.classes_
    _y = self._y

    neighbor_ind = self.kneighbors(X, return_distance=False)
    neighbor_class = _y[neighbor_ind]

    probabilities = np.empty(classes_.size)
    for k, _ in enumerate(classes_):
      probabilities[k] = len(neighbor_class[neighbor_class == k])

    probabilities /= np.sum(probabilities)

    return probabilities

  def score(self, X, y):
    X = X.astype(int)
    y = y.astype(int)

    score = 0
    for i, _ in enumerate(X):
      y_pred = self.predict(X[i])

      if y_pred == y[i]:
        score += 1

    accuracy = score / X.shape[0]

    return accuracy
