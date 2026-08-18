# amc-meta-learning-of-optimal-cate-estimators

Code for the book chapter "Adversarial Monte Carlo Meta-Learning of Conditional Average Treatment Effects" by A. Luedtke and I. Chung [[link](https://alexluedtke.com/papers/Luedtke_Chung_Adversarial_Monte_Carlo_Meta-Learning_of_Conditional_Average_Treatment_Effects.pdf)]

## Environment

The Python scripts were run on GPU machines using Python 3 and PyTorch. The repository also includes `R` scripts, which were run with version 3.2 of the flam package and version 2.1.0 of the grf package.

## Training new estimators

The estimator reported in the chapter was meta-trained under a fused lasso additive model prior with total variation bound 10 and sparsity level 4, at sample size 500 with 10 covariates. It was trained by running the following from a bash shell:

```
./Gam_m10_n500_s4_wdim10.py > Gam_m10_n500_s4_wdim10.out &
```

The scripts `Gam_mM_n500_sS_wdim10.py` similarly train estimators with total variation bound `M` and sparsity level `S`, and the scripts `Linear_n500_sS_wdim10.py` train estimators under linear model priors. The estimators trained by these other scripts are not reported in the chapter.

## Loading the estimators that we trained

The trained estimators are attached to this repository's [release](https://github.com/alexluedtke12/amc-meta-learning-of-optimal-cate-estimators/releases), as several of them exceed GitHub's in-repository file size limit. To use them, download the `.tar` files into a folder named `estimators` at the root of this repository.

For example, the following code, run from the root of this repository, loads the estimator reported in the chapter and uses it to estimate the CATE at new feature values:

```python
import torch
import learn2predict as l2p

# initialize the estimator architecture and load the trained weights
T, _, _ = l2p.initT(lr=0.001, rank_based=True, ablation=0)
checkpoint = torch.load('estimators/Gam_m10_n500_s4_wdim10.tar',
                        map_location=l2p.device, weights_only=False)
T.load_state_dict(checkpoint['T_state_dict'])

# example dataset: x is n x p, a is n x 1, y is n x 1  (here n=500, p=10)
x, a, y, cate = l2p.sim_owl(500, nbatch=1, setting=1)

# CATE estimates at each row of a matrix x0 of test points (returned with shape 1 x 1000)
x0 = 2 * torch.rand(1000, 10, device=l2p.device) - 1
with torch.no_grad():
    cate_hat = T(x0, x[0], a[0], y[0])
```

The estimators trained under linear model priors should instead be initialized with `l2p.initT(lr=0.001, rank_based=False, ablation=0)`.

## Evaluating estimator performance

To evaluate the performance of our estimator in the four simulation scenarios considered in the chapter, we executed `eval_gam.py`. The results are stored in `gam_performance/amc.csv`.

To evaluate the performance of the three existing methods considered in the chapter (a linear-model T-learner, a FLAM T-learner, and causal forests), we ran the `R` script `eval_comparators.R`. The results are stored in `owl_results.Rdata`, and the output of this script can be found in `eval_comparators.out`.

## Citation

If you use our code, please consider citing the following:

Luedtke, Alex, and Incheoul Chung. "Adversarial Monte Carlo Meta-Learning of Conditional Average Treatment Effects." Handbook of Statistical Methods for Precision Medicine. Chapman and Hall/CRC, 2024. 237-248.
