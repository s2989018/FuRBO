Feasibility-Driven Trust Region Bayesian Optimization (FuRBO) is a Bayesian optimization for high dimensional and highly constrained problems.

## General information
FuRBO is a Bayesian optimization for high-dimensional black-box functions under black-box constraints. This algorithm uses trust regions to reduce the search space to only the area where all constraints are fulfilled (i.e., the feasible area). To do so, the algorithm relies on approximating the black-box objective function and constraints with Gaussian process regression to estimate the location of the feasible area with the best objective function. The trust region is placed in the estimated feasible region.

We propose a novel FuRBO variant which addresses a common challenge in trust-region-based BO methods, which struggle to efficiently exploit objective functions that are separable along directions not aligned with the coordinate axes.
We extend this approach in two key ways:
**Multiple trust regions**:
Instead of relying on a single trust region, we use multiple(i.e. 3) trust regions. This allows the algorithm to explore several feasible subspaces in parallel and improves robustness in complex, high-dimensional landscapes.
**PCA-based ellipsoidal trust regions**:
We replace the original axis-aligned trust region with ellipsoidal trust regions using Principal Component Analysis (PCA). PCA captures the local geometry of the explored feasible samples, allowing the trust region to align with the dominant directions of variation in the data. Also, we use an adaptive rescaling strategy based on the length-scales of a local Gaussian process surrogate model with automatic relevance determination. This strategy is used for shrinking/expanding the trust regions. This results in more flexible region shapes compared to the fixed axis-aligned boxes in the original FuRBO. 

#### Workflow
**FuRBO**
1. Generate initial samples
2. Fit Gaussian processes to the objective function and constraints
3. Find the current optimum:
	- if a feasible sample is evaluated: feasible point with the best objective function
	- if no feasible sample is evaluated: point with the smallest violation
4. Sample with a multinormal distribution the Gaussian processes
5. Define trust region around best points according to estimated objective value and constraint violation
6. Draw a Thompson sample in the trust region and estimate the next optimum candidate according to objective value and constraint violation
7. Evaluate the next optimum candidate
8. Update multinormal distribution parameters:
	- if the last n_f optimum candidates are feasible and improve objective -> enlarge multinormal distribution
	- if the last n_f optimum candidates are infeasible or do not improve objective -> shrink multinormal distribution
9. Repeat steps 2 - 8 until the stopping criterion is met

![alt text](https://github.com/paoloascia/FuRBO/blob/main/Figures/workflow/graphical_abstract_furbo.png)

**Novel FuRBO**
## Novel FuRBO Workflow

The novel FuRBO extends the original FuRBO by using **multiple trust regions** and **PCA-based ellipsoidal trust regions**. The workflow proceeds as follows:

1. Generate initial samples using Sobol sequences

2. Fit Gaussian Processes (GPs) to the objective function and constraints using the samples

3. Identify current optima
   - Each trust region tracks its local best feasible point.  
   - If no feasible point exists in a region, the point with the smallest constraint violation is tracked.  
   - The global best is determined from all trust region bests.

4. Maintain multiple trust regions
   - TRs explore different promising subspaces in parallel.  
   - Each TR has its own center, radius, and PCA-based rotation matrix.

5. Define PCA-based ellipsoidal trust regions  
   - Trust regions are ellipsoids aligned with the principal components of feasible samples.  
   - This allows exploration along directions of maximum variation rather than being constrained to axis-aligned boxes.

6. Draw Thompson samples within each trust region  
   For each trust region:
   - Draw random perturbations in a multivariate ellipsoid defined by the TR center, radii, and PCA rotation.  
   - Combine perturbations with the local best point to bias sampling toward promising regions.  
   - Use Constrained Max Posterior Sampling to estimate the next optimum according to objective value and constraint violation

7. Evaluate candidates on the true objective and constraints
   - Update GP models and local/global best points after each batch evaluation.

8. Adapt trust regions using PCA and GP lengthscales**  
   For each TR:  
   - Sample points inside the TR and evaluate their predicted objective and constraint values using the GP models.  
   - Select top-performing points based on feasibility and objective.  
   - Compute PCA on selected points to get principal directions (rotation) and mean (center).  
   - Fit a local GP to extract lengthscales.  
   - Update the TR **center** (`mu`), **rotation** (`R`), and **radii** (scaled PCA eigenvalues × GP lengthscales).


9. **Update trust regions **  
   - TR centers may shift toward **local best points** to bias sampling toward promising regions.

10. **Repeat steps 2–9**  
    - Continue until the **stopping criterion** (maximum evaluations) is met.

11. **Post-processing**  
    - Extract the best objective and constraint values at each iteration.  
    - Generate monotonic convergence curves for visualization and analysis.



## Requirements
To install all required Python packages, you can directly download the `requirements.txt`

```bash
pip install -r requirements.txt
```

## How to run

1) Create a virtual environment
```bash
conda create -n FuRBO python=3.10
conda activate FuRBO
```
2) Extract the repository and install the required packages
```bash
cd FuRBO_repo
pip install -r requirements.txt
```
3) Run the optimization loop for the novel FuRBO and the (baseline) FuRBO

**Novel FuRBO:**
```bash
cd Tests/bbob-constrained-suite/Code/novel_furbo
python 00_main.py
```
**FuRBO** 
```bash
cd Tests/bbob-constrained-suite/Code/FuRBO
python 00_main.py
```
4) Make the convergence plot for 2D and 10D --> after running optimization loop paste the results folder in below mentioned folders respectively
**Novel FuRBO** 
```bash
cd ../Post-processing/novel_furbo
```
**Novel FuRBO** 
```bash
cd ../Post-processing/FuRBO
```
``` bash
cd ../Post-processing/
python FuRBOtenDim.py
python FuRBOtwoDim.py
```

## Reproducibility
To reproduce the results of the poster, please navigate to the folder Tests/bbob-constrained-suite. In here, the folder Post-processing contains the raw data generated by us and the code used to read and generate the plots. The folder Code contains the code to reproduce the results. To generate the results, run `00_main.py`. The raw data will be saved in the Results folder in `.npy` and `.torch` formats. The first type of files contain the monotonic convergence curve (`01_Y_mono.npy`), and the evaluated samples (`02_Y_best.npy`, `02_C_best.npy`). The `.torch` files contain all the information used by the algorithm to define the trust region, to train the surrogates and to estimate the next batch.

## Reference
```
@article{ascia2025feasibility,
  title={Feasibility-Driven Trust Region Bayesian Optimization},
  author={Ascia, Paolo and Raponi, Elena and Bäck, Thomas and Duddeck, Fabian},
  journal={arXiv preprint arXiv:2506.14619},
  year={2025}
  doi={https://doi.org/10.48550/arXiv.2506.14619}
}

The original FuRBO implementation can be found here:  
**https://github.com/paoloascia/FuRBO**



