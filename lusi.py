import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.neighbors import NearestNeighbors


# SVM algorithm
# Section 6.2, taking V=I and no predicates.
class SVM():

    def __init__(self, C=1, kernel='rbf', gamma='auto'):
        self.C = C           # regularization param
        self.kernel = kernel # kernel type
        self.gamma = gamma   # for rbf kernel


    def fit(self, X, Y):
        self.X = X          # data (scaled to [0,1])
        self.Y = Y          # labels
        self.l = len(Y)     # number of samples
        self.d = len(X[0])  # number of features

        # set gamma parameter
        if self.gamma == 'auto':
            self.gamma = 1 / self.d # default gamma = 1 / n_features

        # K-matrix (Kernel)
        if self.kernel == 'linear':
            self.kernel = linear_kernel
            K = self.kernel(self.X, self.X)

        elif self.kernel == 'rbf':
            self.kernel = rbf_kernel
            K = self.kernel(self.X, self.X, self.gamma)

        I = np.eye(self.l)    # identity matrix
        inv = np.linalg.inv(K + self.C*I)

        # Formulas (113), (114), (115)
        A_V = inv.dot(Y)
        A_C = inv.dot(np.ones(self.l))

        # Eq. 117
        a = np.ones(self.l).dot(K).dot(A_C) - np.ones(self.l).dot(np.ones(self.l))
        b = np.ones(self.l).dot(K).dot(A_V) - np.ones(self.l).dot(self.Y)

        # solving system of linear eq.: obtaining c and mu
        self.c = b/a

        # constructing matrix A
        self.A = A_V - (self.c*A_C)


    def decision_function(self, X):
        preds = []
        for x in X:
            preds.append(self.A.dot(self.kernel(self.X,[x])) + self.c)
        return np.array(preds).flatten()


    def predict(self, X):
        preds = []
        for x in X:
            pred = self.A.dot(self.kernel(self.X,[x])) + self.c
            preds.append(1 if pred >= 0.5 else 0)
        return np.array(preds)


    def score(self, X, Y):
        pred = self.predict(X)
        return np.sum(pred == Y) / len(Y)


    def plot_boundaries(self, X_train, y_train, X_test, y_test):
        h = .02
        x_min, x_max = X_train[:, 0].min() - .2, X_train[:, 0].max() + .2
        y_min, y_max = X_train[:, 1].min() - .2, X_train[:, 1].max() + .2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        viz=np.c_[xx.ravel(),yy.ravel()]
        Z = self.decision_function(viz)
        Z = Z.reshape(xx.shape)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(figsize=(5,5))
        plt.title("Classification boundaries with 0 invariants")
        plt.contourf(xx, yy, Z, levels=np.linspace(-1.3,2.3,13), cmap=cm, alpha=.8)
        plt.contour(xx, yy, Z, levels=[0.5], linestyles='dashed')
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=cm_bright, edgecolors='k')
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        plt.tight_layout()
        plt.show()

        

# SVM with invariants
# Section 6.2 (taking V=I) following the example in section 6.7.
class SVM_I(SVC):

    def __init__(self, C=1., kernel='rbf', gamma='auto', delta=10e-3, pred_nn=False):
        self.C = C            # regularization param.
        self.kernel = kernel  # kernel type
        self.gamma = gamma    # for rbf kernel
        self.delta = delta    # T threshold (eq.108)
        self.pred_nn = pred_nn


    def predicates_matrix(self, pred_nn=True, max_pred=10):

        # P1) using predicates based on 0th and 1st moments
        if not pred_nn:   
            # 0th order moment
            phi = np.ones(self.l, dtype=int)
            Phi = [phi]

            # 1st order moments (1 for each dimension of the data)
            for k in range(self.d):
                Phi.append(np.array([self.X[i,k] for i in range(self.l)]))


        # P2) predicates based on local structure (See Example 3a)
        else:    
            Phi = []
            for k in range(max_pred):
                neigh = NearestNeighbors(n_neighbors=k+2)
                neigh.fit(self.X)
                nn = neigh.kneighbors(self.X)[1][:,1:]
                # self.nn = nn
                phi_k = np.array([sum(self.Y[k]) for k in nn])
                Phi.append(phi_k)

        # Custom predicates, uncomment the desired ones...
        
        # P3) Features with low p-value
        
        # P3.1) Preserve risk values
        # phi = np.array([(self.X[i,2]>0 and self.X[i,4]>0 and self.X[i,8]>0) for i in range(self.l)]) 
        # Phi.append(phi)  
        
        # P3.2) Preserve normal values
        # phi = np.array([(self.X[i,2]==0 and self.X[i,4]>0 and self.X[i,8]==0) for i in range(self.l)])
        # Phi.append(phi)
        
        # P4) Genetic risk score (GRS)

        # P4.1) Preserve total GRS
        # phi = np.array([(self.X[i,3] + self.X[i,4] + self.X[i,5] + self.X[i,6]) for i in range(self.l)]) 

        # P4.2) Preserve normal values
        # phi = np.array([(self.X[i,3]==0 and self.X[i,4]==0 and self.X[i,5]==0 and self.X[i,6]==0) for i in range(self.l)])
        # Phi.append(phi)

        # predicates matrix
        return np.array(Phi)


    def fit(self, X, Y):

        self.X = X.copy()            # data (scaled to [0,1])
        if not isinstance(Y, np.ndarray):  # labels
            self.Y = Y.to_numpy()
        else:
            self.Y = Y.copy()
        self.l = len(Y)              # number of samples
        self.d = len(X[0])           # number of features
        self.classes_ = np.unique(Y) # number of classes

        # set gamma parameter
        if self.gamma == 'auto':
            self.gamma = 1 / self.d # default gamma = 1 / n_features

        # K-matrix (Kernel)
        if self.kernel == 'linear':
            self.kernel_f = linear_kernel
            self.K = self.kernel_f(self.X, self.X)

        elif self.kernel == 'rbf':
            self.kernel_f = rbf_kernel
            self.K = self.kernel_f(self.X, self.X, self.gamma)

        # create predicates matrix
        predicates = self.predicates_matrix(pred_nn=self.pred_nn)
        self.m = 0   # current number of active predicates

        I = np.eye(self.l)    # identity matrix
        inv = np.linalg.inv(self.K + self.C*I)

        # Initial sol. (SVM)
        A_V = inv.dot(Y)
        A_C = inv.dot(np.ones(self.l))

        a = np.ones(self.l).dot(self.K).dot(A_C) - self.l
        b = np.ones(self.l).dot(self.K).dot(A_V) - np.ones(self.l).dot(self.Y)

        self.c = b/a
        self.A = A_V - self.c*A_C

        # visualize boundaries
        if self.d == 2:
            self.plot_boundaries(X, Y, X, Y)


        # adding invariants iteratively
        Phi = []
        T_max = 1 + self.delta

        # iterate over predicates finding the one with maximal disagreement value T
        while T_max > self.delta and len(predicates) > 0:
            T = []
            for k in range(len(predicates)):
                # calculate T
                num = predicates[k].dot(self.K).dot(self.A) + self.c*(predicates[k].dot(np.ones(self.l))) - predicates[k].dot(Y)
                den = Y.dot(predicates[k])
                T.append(abs(num) / den)

            # self.T = T
            T_max = max(T)

            # add a predicate 
            if T_max > self.delta:
                self.m += 1
                if type(Phi) == list:
                    Phi.append(predicates[np.argmax(T)])
                    Phi = np.array(Phi)
                else:
                    Phi = np.concatenate([Phi, [predicates[np.argmax(T)]]])

                # delete predicate from the list
                predicates = np.delete(predicates, (np.argmax(T)), axis=0)


                # Formulas (113), (114), (115)
                # A_V = inv.dot(Y)
                # A_C = inv.dot(np.ones(self.l))
                A_S = np.array([inv.dot(Phi[s]) for s in range(self.m)])

                # Eq. 117
                a1_c = np.ones(self.l).dot(self.K).dot(A_C) - self.l
                a1_mu = np.array([np.ones(self.l).dot(self.K).dot(A_S[s]) - np.ones(self.l).dot(Phi[s]) for s in range(self.m)])
                b1 = np.ones(self.l).dot(self.K).dot(A_V) - np.ones(self.l).dot(self.Y)

                # Eq. 118
                ak_c = np.array([A_C.dot(self.K).dot(Phi[k]) - np.ones(self.l).dot(Phi[k]) for k in range(self.m)])
                ak_mu = np.array([np.array([A_S[s].dot(self.K).dot(Phi[k]) for s in range(self.m)]) for k in range(self.m)])
                bk = np.array([A_V.dot(self.K).dot(Phi[k]) - Y.dot(Phi[k]) for k in range(self.m)])

                # creating system of linear equations
                a1 = np.concatenate(([a1_c], a1_mu))
                a2 = np.vstack(([ak_c],ak_mu.T)).T
                a = np.concatenate(([a1],a2), axis=0)
                b = np.concatenate(([b1], bk))

                # obtaining c and mu
                sol = np.linalg.solve(a,b)
                self.c = sol[0]
                self.mu = sol[1:]

                # constructing matrix A
                self.A = A_V - self.c*A_C - np.sum(np.array([self.mu[s]*A_S[s] for s in range(self.m)]), axis=0)

                self.Phi = Phi 

                # visualize boundaries
                if self.d == 2:
                    self.plot_boundaries(X, Y, X, Y)

        return self


    def decision_function(self, X):
        preds = []
        for x in X:
            preds.append(self.A.dot(self.kernel_f(self.X,[x])) + self.c)
        return np.array(preds).flatten()


    def predict(self, X):
        preds = []
        for x in X:
            pred = self.A.dot(self.kernel_f(self.X,[x])) + self.c
            preds.append(1 if pred >= 0.5 else 0)
        return np.array(preds)


    def score(self, X, Y):
        preds = self.predict(X)
        return np.sum(preds == Y) / len(Y)


    def plot_boundaries(self, X_train, y_train, X_test, y_test):
        h = .02
        x_min, x_max = X_train[:, 0].min() - .2, X_train[:, 0].max() + .2
        y_min, y_max = X_train[:, 1].min() - .2, X_train[:, 1].max() + .2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        viz=np.c_[xx.ravel(),yy.ravel()]
        Z = self.decision_function(viz)
        Z = Z.reshape(xx.shape)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(figsize=(5,5))
        plt.contourf(xx, yy, Z, levels=np.linspace(-1.3,2.3,13), cmap=cm, alpha=.8)
        plt.contour(xx, yy, Z, levels=[0.5], linestyles='dashed')
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap=cm_bright, edgecolors='k')
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
        plt.tight_layout()
        plt.title(f"Classification boundaries with {self.m} invariants")
        plt.show()
