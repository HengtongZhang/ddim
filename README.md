# Denoising Diffusion Implicit Models (DDIM)

DDIM Implementation: [Jiaming Song](http://tsong.me), [Chenlin Meng](http://cs.stanford.edu/~chenlin) and [Stefano Ermon](http://cs.stanford.edu/~ermon), Stanford

*Modified into Conditional DDIM by Hengtong.*

## Running the Experiments
The code has been tested on PyTorch 1.12.

### Train a model
Training is exactly the same as DDPM with the following:
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --ni --cond
```
the new added ```--cond``` is the option for conditional ddim.

### Sampling from the model

#### Sampling from the generalized model for FID evaluation
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --sample --fid --timesteps {STEPS} --eta {ETA} --ni
```
where 
- `ETA` controls the scale of the variance (0 is DDIM, and 1 is one type of DDPM).
- `STEPS` controls how many timesteps used in the process.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.

#### Sampling from the sequence of images that lead to the sample
Use `--sequence` option instead.

The above two cases contain some hard-coded lines specific to producing the image, so modify them according to your needs.


## References and Acknowledgements

This implementation is based on / inspired by:

- [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim) (the DDIM Pytorch repo), 
