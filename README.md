# Sobolev Independence Criterion
Pytorch source code for paper
> Mroueh, Sercu, Rigotti, Padhi, dos Santos, "Sobolev Independence Criterion", NeurIPS 2019

### Requirements
* Python 3.6 or above
* PyTorch 1.1.0
* Torchvision 0.3.0
* Scikit-learn 0.21
* Pandas 0.25 (for CCLE dataset)

These can be installed using `pip` by running:

```bash
pip install -r requirements.txt
```

### Usage

We will look at the example of performing feature selection on one of the toy datasets examined in [Zhang et al., arXiv:1606.07892](https://arxiv.org/abs/1606.07892) (see sections 5.1 5.2) that we will call `SinExp`.

* To train an elastic net (one the implemented baseline models) on 250 samples from `SinExp` execute:

  ```bash
  python run_baselines.py --dataset sinexp --numSamples 250 --do-hrt
  ```
  The flag `--do-hrt` tells the script to use the Holdout Randomization Test by [Tansey et al., arXiv:1811.00645](https://arxiv.org/abs/1811.00645) to rank the important features in the data and control False Discovery Rate (FDR).

* To train a multilayer neural network on the prediction problem of regressing the responses `y` on the inputs `X`, subject to gradient penalty (Sobolev penalty), again on 250 samples from `SinExp` execute:
  ```bash
  python run_sic_supervised.py --dataset sinexp --numSamples 250 --do-hrt
  ```

* To train a multilayer discriminator network using the Sobolev Independence Criterion (SIC) between the responses `y` and the inputs `X` on 250 samples from `SinExp` execute:
  ```bash
  python run_sic.py --dataset sinexp --numSamples 250 --do-hrt
  ```

* The results can be plotted using the script `plot_results.py`, which will generate the following figure:
  ![figure](/output/SINEXP_250.png)
  Visualization of the results of executing the previous commands. We plot True Positive Rate (TPR, i.e. Power) and False Discovery Rate (FDR) for the three algorithms, indicating when FDR is controlled with HRT. Higher is better for TPR (blue bars), and lower is better for TPR (red bars)


## Citation
> Youssef Mroueh, Tom Sercu, Mattia Rigotti, Inkit Padhi, Cicero Dos Santos, "Sobolev Independence Criterion", NerIPS, 2019
