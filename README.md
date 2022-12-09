# wbc-micrograph-generation
Generation of artificial white blood cell micrographs using Pyhon

This is a project I did for school. It adapts the work of Scalbert et al. to generating white blood cells (WBCs). Use `python3 <script> --help` to view expected parameters for each script. Details can be found in `report.Rmd`, which can be turned into a PDF using RStudio. A pretrained mask model is provided in `models`, as well as some example masks and images in `images`.

Unfortunately, the texture transfer routine used in this project uses up a huge amount of memory, severely limiting the size of the images that can be generated. We may be able to try something like [`@chuanli11/CNNMRF`](https://github.com/chuanli11/CNNMRF), where we would operate on channel-wise batches of patches, and potentially convert the normalized cross correlation to a convolution with style patches, followed by normalization. Unlike that project, however, we need to consider the semantic layers, and I don't think that we can just concatenate them with each batch.

# References
A list of references used in this project can be found in `report.Rmd`.
