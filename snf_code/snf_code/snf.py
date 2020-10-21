class InterpolatedEnergy(object):
    def __init__(self, energy_model_0, energy_model_1, lambda_):
        """ Single flow layer that does a BD step """
        self.energy_model_0 = energy_model_0
        self.energy_model_1 = energy_model_1
        self.lambda_ = lambda_

    def energy(self, X):
        return (1.0-self.lambda_) * self.energy_model_0.energy(X) + self.lambda_ * self.energy_model_1.energy(X)

    def force(self, X):
        return (1.0-self.lambda_) * self.energy_model_0.force(X) + self.lambda_ * self.energy_model_1.force(X)

