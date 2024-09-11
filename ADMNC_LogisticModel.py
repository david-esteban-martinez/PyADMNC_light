import time

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


from LogisticModel import LogisticModel
import GMM2


class ADMNC_LogisticModel:
    DEFAULT_SUBSPACE_DIMENSION = 10
    DEFAULT_REGULARIZATION_PARAMETER = 1.0
    DEFAULT_LEARNING_RATE_START = 1.0
    DEFAULT_LEARNING_RATE_SPEED = 0.1
    DEFAULT_FIRST_CONTINUOUS = 2
    DEFAULT_MINIBATCH_SIZE = 100
    DEFAULT_MAX_ITERATIONS = 50
    DEFAULT_GAUSSIAN_COMPONENTS = 4
    DEFAULT_NORMALIZING_R = 10.0
    DEFAULT_LOGISTIC_LAMBDA = 1.0

    def __init__(self, subspace_dimension=DEFAULT_SUBSPACE_DIMENSION,
                 regularization_parameter=DEFAULT_REGULARIZATION_PARAMETER,
                 learning_rate_start=DEFAULT_LEARNING_RATE_START,
                 learning_rate_speed=DEFAULT_LEARNING_RATE_SPEED,
                 gaussian_num=DEFAULT_GAUSSIAN_COMPONENTS,
                 normalizing_radius=DEFAULT_NORMALIZING_R,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 logistic_lambda=DEFAULT_LOGISTIC_LAMBDA,
                 minibatch_size=DEFAULT_MINIBATCH_SIZE,
                 first_continuous=0, threshold=-1.0, anomaly_ratio=0.1, data2=None):


        self.first_continuous = first_continuous
        self.logistic_lambda = logistic_lambda
        self.max_iterations = max_iterations
        self.normalizing_radius = normalizing_radius
        self.gaussian_num = gaussian_num
        self.learning_rate_speed = learning_rate_speed
        self.learning_rate_start = learning_rate_start
        self.regularization_parameter = regularization_parameter
        self.subspace_dimension = subspace_dimension
        self.minibatch_size = minibatch_size
        self.threshold = threshold
        self.anomaly_ratio = anomaly_ratio
        self.data2 = data2


    def fit(self, data,y=None):

        numElems = data.shape[0]
        minibatchFraction = self.minibatch_size / numElems
        if minibatchFraction > 1:
            minibatchFraction = 1

        self.logistic = LogisticModel(data, data.shape[1] - self.first_continuous,
                                      self.subspace_dimension, self.normalizing_radius, self.logistic_lambda,self.data2)
        tries = 0
        while self.logistic.trained is False:
            try:
                self.logistic.trainWithSGD(data, self.max_iterations, minibatchFraction, self.regularization_parameter,
                                           self.learning_rate_start,
                                           self.learning_rate_speed)
            except Exception as e:
                tries += 1
                print("Logistic training failed, {tries} times, with Exception {e}".format(tries=tries, e=e))
                if tries > 20:
                    exit(0)
        #Change between custom GMM2 and sklearn GMM
        self.gmm = GMM2.GaussianMixtureModel(n_components=self.gaussian_num)
        # self.gmm = GaussianMixture(n_components=self.gaussian_num)

        data_cont = data[:, self.first_continuous:]
        self.gmm.fit(data_cont)

        estimators = self.getProbabilityEstimators(data)

        targetSize = int(numElems * self.anomaly_ratio)
        if targetSize <= 0: targetSize = 1

        estimators.sort()
        self.threshold = estimators[numElems - targetSize]
        self.findMinMax(data)
        self.classes_ = np.array([-1, 1])


    def getProbabilityEstimators(self, elements):


        logisticEstimators = self.logistic.getProbabilityEstimators(elements)*-1
        # logisticEstimators = np.log(logisticEstimators)
        #Change between custom GMM2 and sklearn GMM
        # gmmEstimators = self.gmm.score_samples(elements[:, self.first_continuous:])
        gmmEstimators = self.gmm.predict_proba(elements[:, self.first_continuous:])

        return logisticEstimators * gmmEstimators



    def isAnomaly(self, elements):
        return self.getProbabilityEstimators(elements) > self.threshold

    # set_params: a function that sets the parameters of the model
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # get_params: a function that returns the parameters of the model
    def get_params(self, deep=True):
        params = {}
        for key in self.__dict__:
            if isinstance(self.__dict__[key], ADMNC_LogisticModel) and deep:
                params[key] = self.__dict__[key].get_params(deep)
            else:
                params[key] = self.__dict__[key]
        return params

    # get_param_names: a function that returns the names of the parameters of the model
    def get_param_names(self):
        names = []
        for key in self.__dict__:
            if isinstance(self.__dict__[key], ADMNC_LogisticModel):
                names.extend(self.__dict__[key].get_param_names())
            else:
                names.append(key)
        return names

    def predict_proba(self, elements):
        return self.getProbabilityEstimators(elements)
        # results = np.zeros((elements.shape[0], 2))  # TODO
        # for i in range(len(elements)):
        #     result = self.getProbabilityEstimator(elements[i])
        #     interpolation = self.interpolate(result)
        #     results[i] = np.array([interpolation, 1 - interpolation])
        # return results

    def findMinMax(self, data):
        results = self.getProbabilityEstimators(data)
        self.max = max(results)
        self.min = min(results)

    def interpolate(self, value):
        t = (value - self.min) / (self.max - self.min)
        return t

    def predict(self, elements):
        return self.getProbabilityEstimators(elements) > self.threshold

        # for i in range(len(elements)):
        #     # a = self.getProbabilityEstimator(elements[i])
        #     if self.getProbabilityEstimators(elements[i]) < self.threshold:
        #         result[i] = 1
        #     else:
        #         result[i] = 0
        # return result

    def decision_function(self, elements):
        return self.getProbabilityEstimators(elements)






if __name__ == '__main__':
    X, y = load_svmlight_file("reduced_data_movie_0.03_4_0.4_0.3_random.libsvm")

    X = X.toarray()

    results = []

    start_time = time.time()
    for i in range(5):
        admnc = ADMNC_LogisticModel(first_continuous=19, subspace_dimension=2, logistic_lambda=0.1,
                                    regularization_parameter=0.001, learning_rate_start=1,
                                    learning_rate_speed=0.2, gaussian_num=2, normalizing_radius=10, anomaly_ratio=0.2)
        admnc.data2=X
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)
        X_train = X_train[y_train[:] != 1]
        y_train = y_train[y_train[:] != 1]
        # print(admnc.get_params())
        admnc.fit(X_train)

        resultsBools = admnc.isAnomaly(X_test)
        resultsProb = admnc.getProbabilityEstimators(X_test)
        result = roc_auc_score(y_test, resultsProb)
        results.append(result)
        print("AUC: " + str(roc_auc_score(y_test, resultsBools)))
        print("\nAUCProb: " + str(result))

    end_time = time.time()
    duration = end_time - start_time
    print(f"TIME: {duration}")
    arrayResults = np.array(results)
    print(arrayResults)
    print("MEAN AND STD: " + str(arrayResults.mean()) + " | " + str(arrayResults.std()))
