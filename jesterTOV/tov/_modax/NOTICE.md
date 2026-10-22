Adapted from lawrenceberry/modax, commit
f4687d97ffdeedc789f80ab31fc26ce912f003b5, under its MIT license (LICENSE).
This private subset keeps Tsit5 and its JAX/CUDA bridge. It omits Enzyme and
autodiff, initializes incomplete terminal saves to NaN, and adds per-lane
mass cutoff and accuracy scaling for Jester. It is not the upstream package.
