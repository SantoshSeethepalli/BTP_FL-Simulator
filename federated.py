import sys
sys.dont_write_bytecode = True

import time
from copy import deepcopy
from threading import Thread, Lock
import torch
from dataset import dataset_loader
from model.layers import CNN
from model.train_test import test, train
from Federated.config import FederatedConfig
from Federated.privacy import apply_privacy
from Federated import plotting

clientModels = []
clientModelsLock = Lock()

# Set number of classes according to MedNIST
n_classes = 9  # e.g., PathMNIST has 9 classes

def clientTraining(serverModel, clientDatasets, client, round, epsilon, sensitivity, privacy_scheme, delta=None):
    global clientModels
    start_time = time.time()
    
    clientTrainingSet = clientDatasets[client][round]
    trainLoader = dataset_loader.get_dataloader(clientTrainingSet)
    clientModel = deepcopy(serverModel)
    trainedClientModel = train(clientModel, trainLoader, client_id=client)  # Pass client_id here

    with torch.no_grad():
        for server_param, trained_param in zip(serverModel.parameters(), trainedClientModel.parameters()):
            update = trained_param.data - server_param.data
            private_update = torch.tensor(
                apply_privacy(update.numpy(), scheme=privacy_scheme, epsilon=epsilon, sensitivity=sensitivity, delta=delta)
            )
            trained_param.data = server_param.data + private_update.to(trained_param.dtype)

    training_time = time.time() - start_time

    clientModelsLock.acquire()
    clientModels.append({
        'model': trainedClientModel,
        'time': training_time,
        'client_id': client
    })
    clientModelsLock.release()

def fedAvg(models):
    averagedModel = deepcopy(models[0])
    with torch.no_grad():
        for model in models[1:]:
            for param1, param2 in zip(averagedModel.parameters(), model.parameters()):
                param1.data += param2.data.to(param1.dtype) 
        for param in averagedModel.parameters():
            param.data /= len(models)
            param.data = param.data.float()
    return averagedModel

def federated():
    global clientModels
    config = FederatedConfig()
    
    client_times = [[] for _ in range(config.clientNum)]
    round_accuracies = []
    avg_round_times = []

    # Load MedNIST dataset
    trainSet = dataset_loader.load_dataset(isTrainDataset=True)
    clientDatasets = dataset_loader.split_client_datasets(
        trainSet, config.clientNum, config.trainingRounds
    )

    testSet = dataset_loader.load_dataset(isTrainDataset=False)
    testLoader = dataset_loader.get_dataloader(testSet)
    
    # Initialize CNN (reads n_classes from updated model/layers.py)
    serverModel = CNN()

    for round in range(config.trainingRounds):
        round_start_time = time.time()
        print(f"Round {round+1}/{config.trainingRounds}")

        clientModels.clear()
        clientThreads = []
        for client in range(config.clientNum):
            t = Thread(
                target=clientTraining, args=(serverModel, clientDatasets, client, round, config.epsilon, config.sensitivity, config.privacy_scheme, config.delta)
            )
            t.start()
            clientThreads.append(t)

        for t in clientThreads:
            t.join()

        round_times = {c['client_id']: c['time'] for c in clientModels}
        for client_id, time_taken in round_times.items():
            client_times[client_id].append(time_taken)
            
        models = [c['model'] for c in clientModels]
        serverModel = fedAvg(models)
        
        testAcc = test(serverModel, testLoader)
        round_accuracies.append(testAcc)
        
        avg_round_time = sum(c['time'] for c in clientModels) / len(clientModels)
        avg_round_times.append(avg_round_time)
        print(" " * 50)
        print(f"Round Accuracy: {testAcc:.4f}")

    print(f"Final Model Accuracy: {round_accuracies[-1]:.4f}")

    plotting.plot_accuracy_vs_rounds(round_accuracies, config.trainingRounds)
    plotting.plot_client_training_times(client_times, avg_round_times, config.trainingRounds)

if __name__ == "__main__":
    federated()
