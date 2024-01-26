## Magnification bias estimation

In this repository, you can find our code to estimate magnification bias for a galaxy sample with a complex photometric selection for the example of SDSS BOSS. The code provided is for the example of CMASS (see Magnification_bias_estimate_example_CMASS.ipynb) and also works for the LOWZ, z1 and z3 samples.

This is the underlying code of the publication Wenzl, Chen, Bean 2023 https://arxiv.org/abs/2308.05892


We also provide a template to apply our approach to other surveys. The information needed to apply the approach to other surveys consists of:

 * The galaxy catalog including the magnitudes used for the photometric selection
 * The exact conditions used for the photometric selection
 * An understanding of how the magnitudes used behave under lensing. In our work for SDSS BOSS we characterized this for magnitudes that capture the full light of the galaxy, psf magnitudes and aperture magnitudes. If you need other magnitudes you need to characterize them yourself.

See magnification_bias_template_other_surveys.py to get started.

