#This script is used for running VFL simulating the interaction among the clients 
import numpy as np
import random
from guest import Guest
from LR import model
from host import Host
#import vertical_splitter as vs
from multiprocessing import Pool, Process
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean
#import testing_dataset as td
import heart_disease_dataset as hdd
#number of participants
num_client=3
#batch size
bs = 1000


#Communication Round 
comm_round = 400

x1,x2,x3=hdd.get_data()
y=hdd.get_labels()
#number of samples
N = x1.shape[0]
#Number of batch 
num_batch=int(N/bs)
epochs=[i for i in range(1,comm_round+1)]
loss_per_epoch = []
losses = []
seen_sample=[]
def generate_batch_ids(limit=N,n_samples=N,batch_size=bs):
    ids=[]
    counter=0
    r = random.sample(range(limit), n_samples)
    #print(r)
    for e in r:
        if e not in seen_sample:
            seen_sample.append(e)
            ids.append(e)
            counter=counter+1
            if counter==batch_size:
                return ids
if __name__=="__main__":
    guest = Guest(0.1,model,data=(x1,y))
    host1 = Host(0.1,model,data=x2)
    host2 = Host(0.1,model,data=x3)
    for r in  tqdm(range(comm_round), desc = 'Communication Round'):
        seen_sample=[]
        for batch in range(int(num_batch)):
            ids=generate_batch_ids()

            # Compute the output Z_b for all participants
            #client_fun_pool=[guest.forward,host1.forward,host2.forward]
            #pool = Pool(processes=10)
            '''for c in client_fun_pool:
                pool.map_async(c, ids)'''
            # pool.map_async(host1.forward, ids)
            #p = Process(target=host1.forward,args=(ids,))
            #p.start()
            #p.join()
            # host1.forward(ids)
            #print(host1.send())
            '''guest.receive(host1.send())
            guest.receive(host2.send())
            
            dw,db=guest.compute_gradient() #fix this
            host1.receive((dw,db))
            host2.receive((dw,db))

            #update model
            loss=pool.map_async(guest.update_model)
            pool.map_async(host1.update_model)
            pool.map_async(host2.update_model)'''
            

            guest.forward(ids)
            host1.forward(ids)
            host2.forward(ids)

            guest.receive(host1.send(),host2.send())
            #guest.receive(host2.send())
            
            guest.compute_gradient()

            
            host1.receive(guest.send())
            host2.receive(guest.send())

            host1.compute_gradient()
            host2.compute_gradient()
            guest.update_model()
            
            host1.update_model()
            host2.update_model()
            losses.append(guest.loss)
        loss_per_epoch.append(sum(losses)/len(losses))
        losses.clear()



    x1_test,x2_test,x3_test = hdd.get_testdata()
    y_test = hdd.get_testlabels()
    #print(x2_test.shape)
    pred_guest= guest.model.predict(x1_test)
    print("Model Accuracy for guest is ", guest.model.accuracy(y_test,pred_guest)*100,"%")
    #print(pred_guest[:10])
    pred_host1= host1.model.predict(x2_test)
    print("Model accuracy for host1 is ",host1.model.accuracy(y_test,pred_host1)*100,"%")
    #print(y_test[:10])
    #print(pred_host1[:10])


    pred_host2= host2.model.predict(x3_test)
    print("Model accuracy for host2 is ",host2.model.accuracy(y_test,pred_host2)*100,"%")

    plt.plot(loss_per_epoch)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()
   



    




            
            

