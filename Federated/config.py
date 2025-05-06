import sys
sys.dont_write_bytecode = True

class FederatedConfig:
    def __init__(self):
        print("Federated Learning Configuration")
        print(" "*50)
        self.clientNum = int(input("Enter number of clients: "))
        self.trainingRounds = int(input("Enter number of training rounds: "))
        self.privacy_scheme = input("Choose Privacy Scheme (none, laplace, gaussian): ").lower()
        self.epsilon = float(input("Enter epsilon (privacy budget): "))
        self.sensitivity = float(input("Enter sensitivity: "))
        self.delta = None
        if self.privacy_scheme == 'gaussian':
            self.delta = float(input("Enter delta for Gaussian DP: "))
        print(" "*50)