## Welcome to tflow: analysis package for experimental 2D PIV/3D PTV data

"tflow" contains funcions to analyze various flows (Stokes to turbulent regimes). Basic features include
- temporal averaging of energy/enstrophy
- spacial averaging of energy/enstrophy
- computing vorticity, shear, rates, rate-of-strain tensor
- computing streamfunctions (Stokes flow)
- conducting Reynolds decomposition (turbulent flow)
- computing 1D/3D energy spectra and n-th orderstructure functions (turbulent flow)
- computing two-point velocity correlction function (spatial autocorrelation function) (turbulent flow)
- computing quadratic inviscid invariants of hydrodynamics (energy, helicity, linear momentum, and angular momentum)

### Philosophy
To make the package compatible for theoretical and experimental studies, the input data is just a numpy array which I refer as ```udata```.

```udata``` has a shape of (dimension, nrows, ncols, (nsteps if applicable), duration)
- ```udata[0, ...]```, ```udata[1, ...]```, ```udata[2, ...]``` represent x-, y-, and  z-component of a velocity field.
- ```udata[0, ..., 100]``` represents the x-component of the velocity field at the 100th frame. 
- ```udata``` assumes an evenly spaced grid. The corresponding positional grid can be generated like 
```markdown
    import numpy as np
    n = 101 # number of points along x and y
    L = np.pi # size of the box
    # 2D grid
    x, y = np.linspace(-L/2., L/2., n), np.linspace(-L/2., L/2., n)
    xx, yy = np.meshgrid(x, y)
    # 3D grid
    z = np.linspace(-L/2., L/2., n)
    xxx, yyy, zzz = np.meshgrid(x, y, z)
```

### Example analysis pipeline
1. Format your PIV-/PTV-extracted velocity field in the format above
... For DaVis (ver.10.1-, LaVision Inc.) users, you may use davis2hdf5.py.
2. ```import tflow.velocity as vel```
3. Load your velocity field data like ```udata = vel.get_udata(path2udata)```
4. Run analysis functions such as ```get_energy(udata)```


### Documentation
The documentation to the key modules (velocity, graph, davis2hdf5) can be found [here](https://github.com/tmatsuzawa/tflow/tree/gh-pages/docs/build/html/index.html).
[1](docs/build/html/index.html), [2](build/html/index.html)



### Contact
If you have questions or suggestions, contact me on my [Github](https://github.com/tmatsuzawa/tflow) or by email (tmatsuzawa_at_uchicago.edu)
