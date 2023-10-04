import numpy as np
import random
from guest import Guest
from model import LogisticRegressionModel
from host import Host
import heart_disease_dataset as hdd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,f1_score,accuracy_score
# Constants
NUM_CLIENTS = 3
BATCH_SIZE = 10
COMM_ROUNDS = 1500

def generate_batch_ids(limit, n_samples, batch_size, seen_samples):
    ids = [e for e in random.sample(range(limit), n_samples) if e not in seen_samples]
    seen_samples.update(ids[:batch_size])
    return ids[:batch_size]

def main():
    x1, x2, x3 = hdd.get_data()
    
    y = hdd.get_labels()
    N = x1.shape[0]
    num_batch = N // BATCH_SIZE
    
    guest = Guest(0.0001, LogisticRegressionModel, data=(x1, y))
    host1 = Host(0.0001, LogisticRegressionModel, data=x2)
    host2 = Host(0.0001, LogisticRegressionModel, data=x3)

    loss_per_epoch = []

    for r in range(COMM_ROUNDS):
        seen_samples = set()
        losses = []
        for _ in range(num_batch):
            ids = generate_batch_ids(N, N, BATCH_SIZE, seen_samples)
            guest.forward(ids)
            host1.forward(ids)
            host2.forward(ids)

            guest.receive(host1.send(), host2.send())
            guest.compute_gradient()

            diff = guest.send()
            host1.receive(diff)
            host2.receive(diff)

            host1.compute_gradient()
            host2.compute_gradient()
            guest.update_model()
            host1.update_model()
            host2.update_model()

            losses.append(guest.loss)
        
        epoch_loss = sum(losses) / len(losses) if losses else 0
        loss_per_epoch.append(epoch_loss)

    x1_test,x2_test,x3_test = hdd.get_testdata()
    
    predictions_local = guest.predict_local(x1_test)
    print("Local Model Accuracy of Guest: ",accuracy_score(hdd.get_testlabels(),predictions_local))
    print("Local Model F1 Score of Guest: ",f1_score(hdd.get_testlabels(),predictions_local))
    print("Local Report",classification_report(hdd.get_testlabels(),predictions_local))

    # The guest receives the contributions and makes the final prediction
    host1_contribution = host1.compute_contribution(x2_test)
    host2_contribution = host2.compute_contribution(x3_test)
    predictions = guest.predict(x1_test, [host1_contribution, host2_contribution])
    print("Federated Model Accuracy of Guest: ",accuracy_score(hdd.get_testlabels(),predictions))
    print("Federated Model F1 Score of Guest: ",f1_score(hdd.get_testlabels(),predictions))
    print("Federated Report",classification_report(hdd.get_testlabels(),predictions))


    plot_loss(loss_per_epoch)


def plot_loss(loss_per_epoch):
    plt.plot(loss_per_epoch)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()

if __name__ == "__main__":
    main()
