# cil-workshop
## road segmentation

### Run
	cd model/xxx
	train: python train.py
	evaluation: python eval.py
	test: python test.py

### Todo
Add new branch: feature/xxx to modify codes

#### Problems 
 - Currently the last validation loss in the last epoch in training is different from that of eval
 - To validate the model save / reload
 - To validate this code does equal to the previous one (by maybe training it and see if the score is close to the previous)

#### improvemnets
 - random rotation
 - there should still be some unnecessary codes can be removed
 - network: add attention in res block