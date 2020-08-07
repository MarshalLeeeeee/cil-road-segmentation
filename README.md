# Road Segmentation

A [Computational Intelligence Lab](http://www.da.inf.ethz.ch/teaching/2020/CIL/) (spring, 2020) project.



Team member: Yiqun Liu, Minchao Liu, Zirun Wang, Zehao Wei



major dependencies (version tested):

- python (3.8.3)
- pytorch (1.5.1)
- matplotlib (3.2.2)
- opencv (4.3.0)



We only tested the code in an environment with NVidia RTX 2070 Super GPU available. However, the code should be runnable with CPU.



Note: many scripts assume that the working directory is the root directory of this project.

To run the test of individual module:

```bash
cd /path/to/project/root/
python -m cil.*
# E. g. python -m cil.utilities.train
```



## Directory Structure

Download the official dataset.

move `training/` and `test_images` directories to  `/path/to/project/root/dataset/`



## Usage

### Training

To train a model, you need to modify `cil/utilities/train.py`:

1. set the checkpoint setting as desired (pay attention to the resume option)
2. set appropriate `max_epochs` and `checkpoint_interval`.
3. select the correct model (normally you need only to change one statement in function `train()`)
4. run command `python -m cil.utilities.train`



During training, checkpoints will be saved in forms of `${model_name}-${epoch}` under `trained_model` directory. You might want to interrupt training and evaluate on checkpoints, or resume training later. In this case you need to set this `checkpoint_name` properly.



If training successfully terminated, a savepoint would be generated, with the name `${model_name}` under `trained_model` directory. Savepoint does not contain optimizer's state and is not a good starting point of resumed training.



### Evaluation

To evaluate a model, you need to modify `cil/utilities/eval.py`:

1. set the checkpoint setting as desired

   - If you want to do evaluation on a checkpoint, you need to set `load_from_checkpoint=True` and specify `checkpoint_name`

   - otherwise, the scripts will do evaluation with savepoint, global variable `savepoint_name` must be set

2. select the correct model (normally you need only to change one statement in function `evaluate()`)

3. run command `python -m cil.utilities.eval`

Evaluation results will be stored in directory `evaluation`



You might want to submit the result to the Kaggle platform. If that is your case, you could run: `python submit_scripts/mask_to_submission.py`. Then a standard `csv` file will be generated in `submission/` directory. They are named by timestamps.



### Postprocessing

Conditional Random Fields (CRF) are used for postprocessing. The implementation refers to https://github.com/lucasb-eyer/pydensecrf.

Run command `python -m cil.utilities.crf` to refine the network output images in `evaluation`. The postprocessed images will overwrite these original images in `evaluation`.

If you want to use hard labels as the unary energy, change the parameter of the `dense_crf` function to `True`.



# Reproduce Submitted Result

Currently, all configurations are the same as the submitted model.

To reproduce the submitted result, you simply need to:

1. make sure your working directory are at the root of project (the directory which contains this file)
2. make sure you have copied dataset into root directory as specified above.
3. run `python -m cil.utilities.train`, wait until the job terminates. Model will be saved to `trained_model` directory.
4. run `python -m cil.utilities.eval`, wait until the job terminates. Predictions on test set will be saved to `evaluation` directory.
5. run `python submit_scripts/mask_to_submission.py`, the latest `csv` file generated in `submission` directory is the file you might like to submit



## Reproduce Results in Experiment Table (in the report)

Set variable `train_all=False` in `cil/utilities/train.py`.

Set variable `edge_penalty=False` if boundary loss does not apply.

Set variable `tta_crop=True` in `cil/utilities/eval.py`.

Select the model you would like to test: `model=SomeModel(...)` in `cil/utilities/train.py` 

Change the `checkpoint_name` in `cil/utilities/eval.py`.



All other steps are same as above.