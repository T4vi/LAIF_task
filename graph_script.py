import re
import numpy as np
import matplotlib.pyplot as plt

with open('..//../file.txt') as f:
    text = f.read().strip()

tr_loss = np.asarray(re.findall('(?<=, loss: )\d.\d*', text), dtype=np.float32)
times = np.asarray(re.findall('(?<=epoch time: )\d*.\d*0', text), dtype=np.float32)
val_loss = np.asarray(re.findall('(?<=validation loss: )\d.\d*', text), dtype=np.float32)
val_rse = np.asarray(re.findall('(?<=validation rse: )\d.\d*', text), dtype=np.float32)
val_corr = np.asarray(re.findall('(?<=validation corr: )\d.\d*', text), dtype=np.float32)
test_rse = np.asarray(re.findall('(?<=test rse: )\d.\d*', text), dtype=np.float32)
test_corr = np.asarray(re.findall('(?<=test corr: )\d.\d*', text), dtype=np.float32)

training_stats = {'tr_loss':tr_loss, 
					's_per_epoch':times, 
					'val_loss':val_loss, 
					'val_rse':val_rse,
					'val_corr':val_corr,
					'test_rse':test_rse,
					'test_corr':test_corr}

plt.figure(1)
plt.title('Train(G) /Val(B) loss')
plt.plot(training_stats['tr_loss'], 'g-')
plt.plot(training_stats['val_loss'], 'b-')
plt.ylabel('Training (G) / Validation (B) loss')
plt.xlabel('Epochs')
#plt.show()

plt.figure(2)
plt.title('Val/Test RSE (B) and CORR (R)')
plt.plot(training_stats['val_rse'], 'b-')
plt.plot(len(training_stats['val_rse'])-1, training_stats['test_rse'][:1], 'k+')
plt.plot(training_stats['val_corr'], 'r-')
plt.plot(len(training_stats['val_corr'])-1, training_stats['test_corr'][:1], 'k+')
plt.ylabel('Validation and Test RSE (B) and CORR (R)')
plt.xlabel('Epochs')
plt.show()

#TODO: save data to file 
#TODO: add poss to load and plot from (data only) file