# Project Summary
## Parallel computing on 2D&3D object morphing with AR exhibition

> This repo is the final project of ME 599 2018 Winter.<br>
> Author: Tianqi Li & Junlin Li

## Process
### 1. 2D morphing
> It's like a process of creating a 3D object  
>(insert images of these two 2D shape and the final 3D object)  

##### There are two 2D shape functions, named f1 and f2. The idea is we combine these two function with a weight t (can think of it as time) (within [0, 1]) as f' = f1 × (1 - t) + f2 × t.
##### For the sequence of t within [0, 1], we obtain a series of intermediate function f', each of which represents the intermediate status of shape f1 morphing into shape f2. If we place the sequence of f' into 3-dimensional space, we can obtain a 3D surface of an object, representing the whole "continuous" process of morphing (but actually it is discrete since we use a finite weight step Δt within the range [0, 1]).

##### For 2D, we used two ellipses, one with the semi-major axis on x-axis(left) and the other one with the semi-major axis on y-axis(right).

Functions are as follow:  
(1) f1(x,y) = x^2 + y^2 / 0.25 - 1  
(2) f2(x,y) = x^2 / 0.25 + y^2 / 1 - 1

<img src="/pics/ellipse_x.jpg" width="230">
<img src="/pics/ellipse_y.jpg" width="230">

### 2. 3D morphing
> It's like a process of creating a 4D object since we created a sequence of 3D object morph (but impossible to display 4D outcome in a 3D world)  

##### We can get the idea of morphing in step 1. Then We can forward into higher dimension space. Two 3D shape functions are chosen, named F1 and F2. The idea is similar to the previous one. We combine these two function with a weight t (within [0, 1]) as F' = F1 × (1 - t) + F2 × t.
##### A series of intermediate function F' is obtained. But because of the 3-dimensional real world, it is impossible to show the whole 'continuous' sequence of morphing like step 1. Instead, we sequently take one t' = k × Δt ( k means the kth point in time sequence, k ϵ Z[0, 1/Δt] (k is integer)). We show the intermediate morphing shape in t' (F'(t') = F1 × (1 - t') + F2 × t')) step by step. Then we can see the process of an object morphing into another object.

##### For 3D, we used a cube with round corners and an ellipsoid (with semi-major axis on x-axis).  
Functions are as follow:  
(1) f1(x,y,z) = x^10 + y^10 + z^10 -1 (left)
(2) f2(x,y,z) = z^2 / 0.5^2 + x^2 + y^2 / 0.5^2 - 1 (right)

<img src="/pics/cube.jpg" width="230">
<img src="/pics/ellipsoid.jpg" width="230">

### 3. Speed up with parallel computing
> Huge improvement from serial to parallel (> 300 times faster)  
>Since the calculation of each point is independent, the whole computation process >can be paralleled.

#### Parallel computation scheme in 2D morphing.

1. Create a mesh grid on x ϵ [-m, m] and y ϵ [-m, m], centered at (0, 0).
2. For each point(x, y) in a small grid in the thread, we put the result of fk' = f1(x, y) × (1 - tk) + f2(x, y) × tk in a 3D grid F(x, y, tk).


#### Parallel computation scheme in 3D morphing.
1. Create a mesh grid on x ϵ [-m, m], y ϵ [-m, m] and z ϵ [-m, m], centered at (0, 0, 0).
2. For each point(x, y, z) in a small grid in the thread, we put the result of fk' = f1(x, y, z) × (1 - tk) + f2(x, y, z) × tk in a 4D grid F(x, y, z, tk).

### 4. Polygon mesh and output
> The object should be agreed with the type that Hololens supports.
> Blender is the tool to use to convert objects into fbx file format.  

In step 3, we have the result matrix. Then we use the plotUtils.py to obtain the output file containing the information of the matrix (object). In 2D, we just export the 3D result object in vertices and faces format (PLY file) directly. But in 3D, we export a sequence of files each of which stores the intermediate object in time tk, using the same file format (PLY).

Import these PLY files into Blender. XXXXX

### 5. App built with Visual Studio and Unity

> Add detectable gesture and the ability of scaling, rotating and placing the object.

##### Import the fbx files into Unity and place them accordingly. Then we add gesture and manipulation properties to those objects so that they can be scaled, moved, rotated in AR environment. We coded the interactive properties via C# in Visual Studio. Configure deployment settings correctly and then build the VS solution.

### 6. Deploy to Hololen glasses

##### Open up the solution we built in last step in Visual Studio. Deploy as "Master x86" via remote machine (using WIFI, if used USB connection, "Device" should be chosen as the target deploy place). Then VS will upload all the stuffs and install the app in Hololens if the device is perfectly paired with PC.
##### Get in Hololens and we can "click" the app we just built to launch and see how it goes. Then the morphing objects are exhibited to us in AR world.
##### If object is selected and clicked to adjust, then you can do manipulations to this object.

## Problems encountered
* Correctness of the shape function of the object
* Parallelism on 3D computing
* Arrange correct object file format to meet the requirement of polygon meshing so that it can be read in Blender and Unity
* Version mathch of the softwares
* Hololen connection with Visual Studio

## Possible improvement and future to-dos

1. Render a nicer background. (Need solid Unity modeling skills)

2. Figure out more complex functions. (Mathmetica or other software that can generate functions for complex shape may be required. And have to figure out the sharp turns when building models)

3. More complex interactive properties. Like dynamically get an intermediate morphed shape with specific weight t by sliding your hand.

4. Then it rises the next improvement: real time (online) parallel model calculation and AR exhibition. This may need to find a way to make python and Unity directly communicate, which is not likely right now said by the community of Unity developers. (If Unity has cuda support in C#, then the whole process can be run in Unity only, which may make real-time possible.)

## Comments on the environment for AR developer
* Inconsistence programming language. (Python --- C#)
* Unable to make real-time object rendering. (Calculate in Python and update shape in Unity)
* Cross platform development. (Python ---> Blender ---> Unity ---> Visual Studio ---> Hololens)
* Difficult to preview. (must deploy it to see whether it goes as we thought).
