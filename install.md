# Packages
- (set CUDA_HOME env to compile deform_attn and GroundingDINO packages NOT in CPU only mode)
- mish-cuda
  - install as descripted in repository.
  - -> "mish-cuda" folder

- conda install pycocotools tqdm cython scipy (for **deform_attn**)
- deformable_attention
  - doesn't install like descripted, needed to copy "**deform_attn**" from docker container to host machine and install that.
  - -> "deform_attn" folder

- GroundingDINO
  - install all packages in "requirements.txt" (pip install -e .)
  - install supervison=0.21.0 (with pip3!)