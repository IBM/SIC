# Sobolev Independence Criterion
Pytorch source code for paper
> Mroueh, Sercu, Rigotti, Padhi, dos Santos, "Sobolev Independence Criterion", NeurIPS 2019 [[arXiv:1910.14212](https://arxiv.org/abs/1910.14212)] [[NeurIPS 2019 Proceedings]](https://papers.nips.cc/paper/9147-sobolev-independence-criterion)


## Requirements
* Python 3.6 or above
* PyTorch 1.1.0
* Torchvision 0.3.0
* Scikit-learn 0.21
* Pandas 0.25 (for CCLE dataset)

These can be installed using `pip` by running:

```bash
pip install -r requirements.txt
```

## Usage

We will look at the example of performing *feature selection* on one of the toy datasets examined in [Zhang et al., arXiv:1606.07892](https://arxiv.org/abs/1606.07892) (see sections 5.1 5.2) that we will call `SinExp`.

* **Baseline models:**
  * To train an *elastic net* (one of the implemented baseline models) on 250 samples from `SinExp` execute:
    ```bash
    python run_baselines.py --model en --dataset sinexp --numSamples 250 --do-hrt
    ```
  * Analogously, to train a *random forest* on 250 samples from `SinExp` execute:
    ```bash
    python run_baselines.py --model rf --dataset sinexp --numSamples 250 --do-hrt
    ```
    The flag `--do-hrt` tells the script to use the Holdout Randomization Test by [Tansey et al., arXiv:1811.00645](https://arxiv.org/abs/1811.00645) to rank the important features in the data and control False Discovery Rate (FDR).    

* **Multi-layer neural network regression with Sobolev penalty:** To train a multilayer neural network on the prediction problem of regressing the responses `y` on the inputs `X`, subject to gradient penalty (Sobolev penalty), again on 250 samples from `SinExp` execute:
  ```bash
  python run_sic_supervised.py --dataset sinexp --numSamples 250 --do-hrt
  ```

* **Sobolev Independence Criterion:** To train a multilayer discriminator network using the Sobolev Independence Criterion (SIC) between the responses `y` and the inputs `X` on 250 samples from `SinExp` execute:
  ```bash
  python run_sic.py --dataset sinexp --numSamples 250 --do-hrt
  ```

* The results can be plotted using the script `plot_results.py`, which will generate the following figure:

  ![figure](/output/SINEXP_250.png)
  Visualization of the results of executing the previous commands. We plot True Positive Rate (TPR, i.e. Power) and False Discovery Rate (FDR) for the three algorithms, indicating when FDR is controlled with HRT. Higher is better for TPR (blue bars), and lower is better for TPR (red bars). The red horizontal dashed line indicates a TPR of 10%, which is what was used as target FDR for HRT. In this case SIC combined with HRT (bars on the right) has the highest TPR, while maintaining a low FDR.
  

## Citation
> Youssef Mroueh, Tom Sercu, Mattia Rigotti, Inkit Padhi, Cicero Dos Santos, "Sobolev Independence Criterion", NeurIPS, 2019 [[arXiv](https://arxiv.org/abs/1910.14212)] [[NeurIPS Proceedings]](https://papers.nips.cc/paper/9147-sobolev-independence-criterion)

``` 
@incollection{NIPS2019_9147,
title = {Sobolev Independence Criterion},
author = {Mroueh, Youssef and Sercu, Tom and Rigotti, Mattia and Padhi, Inkit and Nogueira dos Santos, Cicero},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {9505--9515},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/9147-sobolev-independence-criterion.pdf}
}
```

