
# ngmix4piff

This is a simple pluggin for Piff to add support of Ngmix to measure shape on
the output PSF model and stars (similar to the HSMCatalog)

## Installation

Clone the repo locally and then from the root of the repo:

```bash
conda env update -f environment.yml
pip install .
```

## How to use

Here is how to use this pluggin in Piff from the config file:

```yaml
modules:
    - ngmix4piff

[your piff config]

output:
    stats:
        -
            type: NgmixCatalog
            file_name: "ngmix_cat.fits"
            seed: 42
            fitters:
                - 
                    model: gauss
                    ntry: 3
                    ngauss: 1
                    seed: 24
                - 
                    model: gauss
                    ntry: 3
                    ngauss: 5
                    em: true
                - 
                    model: wmom
                    weight:
                        fwhm: 0.5
                - 
                    model: am
                    ntry: 3
```

## Available fitters

At the moment the availalble fitters are:

- `wmom`: weighted moment for which one can set the size of the window function
- `am`: adaptive moments
- `gauss`: mixture of gaussian fitting. By default the fitter will use LMSimple minimization. Thought for `ngauss > 1` it is recommended to set `em: true` to use the EMFitter instead.

## Examples

The example directory is taken from `Piff` with the updated config file to include the new `NgmixCatalog` stat.
