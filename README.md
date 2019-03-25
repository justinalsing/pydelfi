# pydelfi

**NOTE:** the API has changed recently, please check the docs and/or notebook examples to get up-to-date.

**Density Estimation Likelihood-Free Inference** with neural density estimators and adaptive acquisition of simulations. The implemented methods are described in detail in [Alsing, Charnock, Feeney and Wandelt 2019](https://arxiv.org/abs/1903.00007), and are based closely on [Papamakarios, Sterratt and Murray 2018](https://arxiv.org/pdf/1805.07226.pdf), [Lueckmann et al 2018](https://arxiv.org/abs/1805.09294) and [Alsing, Wandelt and Feeney, 2018](https://academic.oup.com/mnras/article-abstract/477/3/2874/4956055?redirectedFrom=fulltext). Please cite these papers if you use this code!

**Installation:**

The code is in python3 and has the following dependencies:<br>
[tensorflow](https://www.tensorflow.org)<br>
[getdist](http://getdist.readthedocs.io/en/latest/)<br>
[emcee](http://dfm.io/emcee/current/)<br>
[tqdm](https://github.com/tqdm/tqdm)<br>
[mpi4py](https://mpi4py.readthedocs.io/en/stable/) (if MPI is required)<br>

You can install the requirements and this package with,
```
pip install git+https://github.com/justinalsing/pydelfi.git
```
or alternatively, pip install the requirements and then clone the repo and run `python setup.py install`

**Documentation and tutorials:** 

Once everything is installed, try out either `cosmic_shear.ipynb` or `jla_sne.ipynb` as example templates for how to use the code; plugging in your own simulator and letting pydelfi do it's thing. 

If you have a set of pre-run simulations you'd like to throw in rather than allowing pydelfi to run simulations on-the-fly, look at `cosmic_shear_prerun_sims.ipynb` as a template for how to do this.

If you are interested in using pydelfi with nuisance hardened data compression to project out nuisances ([Alsing & Wandelt 2019](https://arxiv.org/abs/1903.01473v1)), take a look at `jla_sne_marginalized.ipynb`.

**[Documentation](https://pydelfi.readthedocs.io/en/latest/)** can be found here (work in progress).

If you are interested in applying pydelfi to your problem but need some help getting started, or have an application that requires adaptations of the code, don't hesitate to get in touch with us (at justin.alsing@fysik.su.se) or open an issue - we welcome collaboration!
