# Project Summary
## Parallel computing on 2D&3D object morphing with AR exhibition

> This repo is the final project of ME 599 2018 Winter.<br>
> Author: Tianqi Li & Junlin Li

## Process
### 1. 2D morphing
> Create a 3D object
### 2. 3D morphing
> Create a sequence of 3D object morph (impossible to display 4D outcome in a 3D world)
### 3. Speed up with parallel computing
> Huge improvement from serial to parallel (> 300 times faster)
### 4. Polygon mesh and output
> The object should be agreed with the type that Hololens supports.
> Blender is the tool to use to convert objects into XXX file format.

### 5. App built with Visual Studio and Unity

> Add detectable gesture and the ability of scaling, rotating and placing the object.

### 6. Deploy to Hololen glasses

## Problems encountered
* Correctness of the shape function of the object
* Parallelism on 3D computing
* Arrange correct object file format to meet the requirement of polygon meshing so that it can be read in Blender and Unity
* Version mathch of the softwares
* Hololen connection with Visual Studio

## Possible improvement and future todos

## Comments on the environment for AR developer
> MS is SHIT!!! Unless it gives me offer!!
* Inconsistence programming language (Python --- C#)
* Cross platform development (Python ---> Blender ---> Unity ---> Visual Studio ---> Hololens)
* Difficult to preview (must deploy it to see whether it goes as we thought).
