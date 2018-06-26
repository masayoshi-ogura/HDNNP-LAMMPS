# HDNNP-LAMMPS

LAMMPS-extending program that consists of following 4 .h and .cpp files

- neural_network_potential.h
- neural_network_potential.cpp
- pair_nnp.h
- pair_nnp.cpp

# setup and compile

```
$ cd path_to_lammps
$ cd src/
$ ln -s path_to_this/neural_newtork_potential.h
$ ln -s path_to_this/neural_newtork_potential.cpp
$ ln -s path_to_this/pair_nnp.h
$ ln -s path_to_this/pair_nnp.cpp
```

then, this pair potential use Eigen library, and MKL library.  
so you have to install them.  
in order to use MKL in Eigen, edit a makefile in `path_to_lammps/src/MAKE/*/Makefile.*`.  
in the makefile, add `-mkl` option as follows.

```
...
LIB = -mkl
...
```

finally, you can compile lammps with Neural Network Potential

```
$ make mpi  #=> lmp_mpi will be created in src/
```

# how to use

all parameters are set in one potential file in txt format.
sample parameter file `coeff_sample` is also included in this repository

## LAMMPS command

2 changes in LAMMPS script,

```
pair_style nnp 
pair_coeff * * coeff_sample Ga N
```

you have to change only `coeff_sample Ga N` part.
all parameters should be written in the potential file.


# HDNNP program

[HDNNP](https://github.com/ogura-edu/HDNNP)
