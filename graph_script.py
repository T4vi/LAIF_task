import re
import numpy as np
import matplotlib.pyplot as plt

with open('file.txt') as f:
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
plt.title('Train/Val loss')
plt.plot(training_stats['tr_loss'], 'g-')
plt.plot(training_stats['val_loss'], 'b-')
plt.ylabel('Training (G) / Validation (B) loss')
plt.xlabel('Epochs')
#plt.show()

plt.figure(2)
plt.title('Val/Test RSE and CORR')
plt.plot(training_stats['val_rse'], 'b-')
plt.plot(len(training_stats['val_rse'])-1, training_stats['test_rse'], 'r+')
plt.plot(training_stats['val_corr'], 'b-')
plt.plot(len(training_stats['val_corr'])-1, training_stats['test_corr'], 'r+')
plt.ylabel('Validation (B) and Test (R) RSE and CORR')
plt.xlabel('Epochs')
plt.show()