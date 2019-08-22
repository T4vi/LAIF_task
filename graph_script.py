import re
import numpy as np
import matplotlib.pyplot as plt

with open('results_TF_16_XX_3_128.txt') as f:
    text = f.read().strip()

tr_loss 	= np.asarray(re.findall('(?<=, loss: )\d.\d*', text), dtype=np.float32)
times 		= np.asarray(re.findall('(?<=epoch time: )\d*.\d*0', text), dtype=np.float32)
val_loss 	= np.asarray(re.findall('(?<=validation loss: )\d.\d*', text), dtype=np.float32)
val_rse 	= np.asarray(re.findall('(?<=validation rse: )\d.\d*', text), dtype=np.float32)
val_corr 	= np.asarray(re.findall('(?<=validation corr: )\d.\d*', text), dtype=np.float32)
test_rse 	= np.asarray(re.findall('(?<=test rse: )\d.\d*', text), dtype=np.float32)
test_corr 	= np.asarray(re.findall('(?<=test corr: )\d.\d*', text), dtype=np.float32)
test_p		= re.findall('(?<=precision: )\d.\d*', text)
test_r		= re.findall('(?<=recall: )\d.\d*', text)
test_f1		= re.findall('(?<=F1 score: )\d.\d*', text)

training_stats = {'tr_loss':tr_loss, 
					's_per_epoch':times, 
					'val_loss':val_loss, 
					'val_rse':val_rse,
					'val_corr':val_corr,
					'test_rse':test_rse,
					'test_corr':test_corr,
					'test_p':test_p,
					'test_r':test_r,
					'test_f1':test_f1,}

f = plt.figure(1)
p = f.add_subplot(111)
plt.title('Train(G) /Val(B) loss')
plt.plot(training_stats['tr_loss'], 'g-')
plt.plot(training_stats['val_loss'], 'b-')
plt.ylabel('Training (G) / Validation (B) loss')
plt.xlabel('Epochs')
#plt.show()


if (training_stats['val_rse'].shape[0] != 0):
	plt.figure(2)
	plt.title('Val/Test RSE (B) and CORR (R)')
	plt.plot(training_stats['val_rse'], 'b-')
	if (training_stats['test_rse'].shape[0] != 0):
		plt.plot(len(training_stats['val_rse'])-1, training_stats['test_rse'][-1:], 'k+')
	plt.plot(training_stats['val_corr'], 'r-')
	if (training_stats['test_corr'].shape[0] != 0):
		plt.plot(len(training_stats['val_corr'])-1, training_stats['test_corr'][-1:], 'k+')
	plt.ylabel('Validation and Test RSE (B) and CORR (R)')
	plt.xlabel('Epochs')
elif (training_stats['test_p'][0] != 0):
	plt.text(0.6, 0.95, ['Precision: '] + training_stats['test_p'], transform=p.transAxes)
	plt.text(0.6, 0.9, ['Recall: '] + training_stats['test_r'], transform=p.transAxes)
	plt.text(0.6, 0.85, ['F1 Score: '] + training_stats['test_f1'], transform=p.transAxes)

plt.show()

#TODO: plot P, R, F1
#TODO: save data to file 
#TODO: add poss to load and plot from (data only) file