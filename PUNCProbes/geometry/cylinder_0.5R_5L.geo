// I usually do steps 1-2 in a text editor, steps 3-11 in the Gmsh GUI.

// STEP 1: SET VARIABLES

debye = 0.00690; // Electron debye length for n=1e11 and T=1000
r = 0.5*debye;   // Inner radius
R = r+10*debye;  // Outer radius
l = 5*debye;     // Inner length
L = l+20*debye;  // Outer length
Res = 1.5*debye; // Resolution on outer boundary
res = r/5;       // Resolution on inner boundary

// STEP 2: PLACE POINTS (0D ENTITIES)

// Outer Boundary
Point(1)  = { 0, -L/2,  0, Res};
Point(2)  = {+R, -L/2,  0, Res};
Point(3)  = {-R, -L/2,  0, Res};
Point(4)  = { 0, -L/2, +R, Res};
Point(5)  = { 0, -L/2, -R, Res};
Point(6) = { 0,  L/2,  0, Res};
Point(7) = {+R,  L/2,  0, Res};
Point(8) = {-R,  L/2,  0, Res};
Point(9) = { 0,  L/2, +R, Res};
Point(10) = { 0,  L/2, -R, Res};

// Inner Boundary
Point(11) = { 0, -l/2,  0, res};
Point(12) = {+r, -l/2,  0, res};
Point(13) = {-r, -l/2,  0, res};
Point(14) = { 0, -l/2, +r, res};
Point(15) = { 0, -l/2, -r, res};
Point(16) = { 0,    0,  0, res};
Point(17) = {+r,    0,  0, res};
Point(18) = {-r,    0,  0, res};
Point(19) = { 0,    0, +r, res};
Point(20) = { 0,    0, -r, res};
Point(21) = { 0,  l/2,  0, res};
Point(22) = {+r,  l/2,  0, res};
Point(23) = {-r,  l/2,  0, res};
Point(24) = { 0,  l/2, +r, res};
Point(25) = { 0,  l/2, -r, res};

// STEP 3: CONNECT POINTS TO LINES AND ARCS (1D ENTITIES)

Circle(1) = {3, 1, 4};
Circle(2) = {4, 1, 2};
Circle(3) = {2, 1, 5};
Circle(4) = {5, 1, 3};
Circle(5) = {9, 6, 8};
Circle(6) = {8, 6, 10};
Circle(7) = {10, 6, 7};
Circle(8) = {7, 6, 9};
Line(9) = {9, 4};
Line(10) = {8, 3};
Line(11) = {10, 5};
Line(12) = {7, 2};
Circle(13) = {14, 11, 12};
Circle(14) = {12, 11, 15};
Circle(15) = {15, 11, 13};
Circle(16) = {13, 11, 14};
Circle(17) = {19, 16, 17};
Circle(18) = {17, 16, 20};
Circle(19) = {20, 16, 18};
Circle(20) = {18, 16, 19};
Circle(21) = {24, 21, 22};
Circle(22) = {22, 21, 25};
Circle(23) = {25, 21, 23};
Circle(24) = {23, 21, 24};
Line(25) = {19, 14};
Line(26) = {17, 12};
Line(27) = {20, 15};
Line(28) = {18, 13};
Line(29) = {24, 19};
Line(30) = {22, 17};
Line(31) = {25, 20};
Line(32) = {23, 18};

// STEP 4: FILL PLANE AND CURVED SURFACES (2D ENTITIES) BETWEEN LINES AND ARCS

Line Loop(1) = {4, 1, 2, 3};
Plane Surface(1) = {1};
Line Loop(2) = {8, 5, 6, 7};
Plane Surface(2) = {2};
Line Loop(3) = {13, 14, 15, 16};
Plane Surface(3) = {3};
Line Loop(4) = {21, 22, 23, 24};
Plane Surface(4) = {4};
Line Loop(5) = {9, 2, -12, 8};
Surface(5) = {5};
Line Loop(6) = {3, -11, 7, 12};
Surface(6) = {6};
Line Loop(7) = {11, 4, -10, 6};
Surface(7) = {7};
Line Loop(8) = {10, 1, -9, 5};
Surface(8) = {8};
Line Loop(9) = {16, -25, -20, 28};
Surface(9) = {9};
Line Loop(10) = {19, 28, -15, -27};
Surface(10) = {10};
Line Loop(11) = {27, -14, -26, 18};
Surface(11) = {11};
Line Loop(12) = {17, 26, -13, -25};
Surface(12) = {12};
Line Loop(13) = {29, -20, -32, 24};
Surface(13) = {13};
Line Loop(14) = {21, 30, -17, -29};
Surface(14) = {14};
Line Loop(15) = {18, -31, -22, 30};
Surface(15) = {15};
Line Loop(16) = {23, 32, -19, -31};
Surface(16) = {16};

// STEP 5: FILL VOLUME (3D ENTITY) BETWEEN SURFACES

Surface Loop(1) = {2, 5, 8, 7, 6, 1};
Surface Loop(2) = {14, 4, 15, 11, 10, 16, 13, 9, 3, 12};
Volume(1) = {1, 2};

// STEP 6: CREATE PHYSICAL GROUPS OF SURFACES 

// Outer surface
Physical Surface(1) = {2, 8, 5, 7, 6, 1};

// Top segment of inner surface
Physical Surface(2) = {16, 13, 14, 15, 4};

// Bottom segment of inner surface
Physical Surface(3) = {10, 11, 12, 9, 3};

// STEP 7: CREATE A PHYSICAL GROUP FOR THE VOLUME

Physical Volume(4) = {1};

// STEP 8: GENERATE 2D AND 3D MESH

// Make sure both checkboxes starting with "Optimize quality of tetrahedra..."
// is activated in Tools -> Options -> Mesh -> Advanced -> Optimize with
// NetGen... is activated

// STEP 10: INSPECT QUALITY 

// Tools -> Statistics -> Mesh

// Number of tetrahedra indicate computational cost of Poisson solver.

// Thin/sliver cells lead to less accurate solutions and slower convergence.
// Gmsh reports the quality factor Gamma, which indicate how regular the cells
// are. A cell with Gamma=1 is perfectly equilateral, whereas a cell with
// Gamma=0 is degenerate and will break the simulation (other tools may use
// other definitions of the quality factor). Click Update to compute the Gamma
// value for all cells. The range and average will be shown in the text box,
// and a X-Y plot of the distribution can be shown. The higher the average, the
// better. If Gamma is practically zero (1e-15) it indicates an error in the
// geometry, for instance duplicate points. For simple geometries the minimum
// Gamma is usually above 0.3, but for more difficult geometries it can be less
// than 0.05. This makes the simulation more difficult, and at some point Gamma
// may be so small the simulation breaks. NetGen optimization is important for
// good quality tetrahedra.

// STEP 11: SAVE MESH TO *.msh

// STEP 12: CONVERT MESH TO *.topo FOR USE WITH PTETRA

// 1. Place a symlink to msh2topo in the Geometry folder
// 2. Copy msh2topo.dat to the Geometry folder
// 3. Enter the filename of the msh-file in msh2topo.dat
// 4. Run msh2topo.dat
// 5. Rename msh2topo.out to *.topo
