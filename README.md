# pydelfi

**[Documentation](https://pydelfi.readthedocs.io/en/latest/)** (work in progress)

**Density Estimation Likelihood-Free Inference** with neural density estimators and adaptive acquisition of simulations. The implemented methods are described in detail in [Alsing, Charnock, Feeney and Wandelt 2019](https://arxiv.org/abs/1903.00007), and are based closely on [Papamakarios, Sterratt and Murray 2018](https://arxiv.org/pdf/1805.07226.pdf), [Lueckmann et al 2018](https://arxiv.org/abs/1805.09294) and [Alsing, Wandelt and Feeney, 2018](https://academic.oup.com/mnras/article-abstract/477/3/2874/4956055?redirectedFrom=fulltext). Please cite these papers if you use this code!

**Dependencies:** [tensorflow](https://www.tensorflow.org), [getdist](http://getdist.readthedocs.io/en/latest/), [emcee](http://dfm.io/emcee/current/), [mpi4py](https://mpi4py.readthedocs.io/en/stable/).

You can install the requirements and this package with:
```
pip install git+https://github.com/justinalsing/pydelfi.git
```

**Usage:** Once everything is installed, try out either `cosmic_shear.ipynb` or `jla_sne.ipynb` as example templates for how to use the code; plugging in your own simulator and letting pydelfi do it's thing. 

If you have a set of pre-run simulations you'd like to throw in rather than allowing pydelfi to run simulations on-the-fly, look at `cosmic_shear_prerun_sims.ipynb` as a template for how to do this.

If you are interested in using pydelfi with nuisance hardened data compression to project out nuisances ([Alsing & Wandelt 2019](https://arxiv.org/abs/1903.01473v1)), take a look at `jla_sne_marginalized.ipynb`.

The code is not documented yet (documentation coming imminently), but if you are interested in applying it to your problem please get in touch with us (at justin.alsing@fysik.su.se) - we welcome collaboration!
