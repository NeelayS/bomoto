Matrices downloaded from https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=model_transfer.zip and converted to sparse COO format.

When converting between two models with different topology, it is required to use a deformation matrix that maps between the two topologies.

Currently, the supported models are in

- SMPL topology (6890 vertices)
  - SMPL
  - SMPL-H
- SMPL-X toplogy (10475 vertices)
  - SMPL-X
  - SUPR
  - SKEL

\
\
Use `smpl2smplx_deformation_matrix.pkl` when converting from SMPL topology to SMPL-X topology.\
Use `smplx2smpl_deformation_matrix.pkl` when converting from SMPL-X topology to SMPL topology.
