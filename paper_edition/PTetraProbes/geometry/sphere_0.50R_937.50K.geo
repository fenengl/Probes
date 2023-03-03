// I usually do steps 1-2 in a text editor, steps 3-11 in the Gmsh GUI.

// STEP 1: SET VARIABLES

debye = 0.006682;
r = 0.50*debye;
R = r+10*debye;  // Outer radius
Res = 1.5*debye; // Resolution on outer boundary
res = r/5;       // Resolution on inner boundary

// STEP 2: PLACE POINTS (0D ENTITIES)

// Center
Point(1) = {0, 0, 0, Res};

// Outer boundary
Point(2) = {R, 0, 0, Res};
Point(3) = {0, R, 0, Res};
Point(4) = {0, 0, R, Res};
Point(5) = {-R, 0, 0, Res};
Point(6) = {0, -R, 0, Res};
Point(7) = {0, 0, -R, Res};

// Inner boundary
Point(8) = {r, 0, 0, res};
Point(9) = {0, r, 0, res};
Point(10) = {0, 0, r, res};
Point(11) = {-r, 0, 0, res};
Point(12) = {0, -r, 0, res};
Point(13) = {0, 0, -r, res};

// STEP 3: CONNECT POINTS TO LINES AND ARCS (1D ENTITIES)

Circle(1) = {5, 1, 4};
Circle(2) = {4, 1, 2};
Circle(3) = {2, 1, 7};
Circle(4) = {7, 1, 5};
Circle(5) = {4, 1, 3};
Circle(6) = {3, 1, 7};
Circle(7) = {7, 1, 6};
Circle(8) = {6, 1, 4};
Circle(9) = {5, 1, 3};
Circle(10) = {3, 1, 2};
Circle(11) = {2, 1, 6};
Circle(12) = {6, 1, 5};
Circle(13) = {11, 1, 10};
Circle(14) = {10, 1, 8};
Circle(15) = {8, 1, 13};
Circle(16) = {13, 1, 11};
Circle(17) = {11, 1, 9};
Circle(18) = {9, 1, 8};
Circle(19) = {8, 1, 12};
Circle(20) = {12, 1, 11};
Circle(21) = {10, 1, 9};
Circle(22) = {9, 1, 13};
Circle(23) = {13, 1, 12};
Circle(24) = {12, 1, 10};

// STEP 4: FILL PLANE AND CURVED SURFACES (2D ENTITIES) BETWEEN LINES AND ARCS

Line Loop(1) = {9, -5, -1};
Surface(1) = {1};
Line Loop(2) = {5, 10, -2};
Surface(2) = {2};
Line Loop(3) = {10, 3, -6};
Surface(3) = {3};
Line Loop(4) = {4, 9, 6};
Surface(4) = {4};
Line Loop(5) = {12, 1, -8};
Surface(5) = {5};
Line Loop(6) = {8, 2, 11};
Surface(6) = {6};
Line Loop(7) = {11, -7, -3};
Surface(7) = {7};
Line Loop(8) = {7, 12, -4};
Surface(8) = {8};
Line Loop(9) = {21, 18, -14};
Surface(9) = {9};
Line Loop(10) = {18, 15, -22};
Surface(10) = {10};
Line Loop(11) = {22, 16, 17};
Surface(11) = {11};
Line Loop(12) = {17, -21, -13};
Surface(12) = {12};
Line Loop(13) = {24, 14, 19};
Surface(13) = {13};
Line Loop(14) = {19, -23, -15};
Surface(14) = {14};
Line Loop(15) = {16, -20, -23};
Surface(15) = {15};
Line Loop(16) = {20, 13, -24};
Surface(16) = {16};

// STEP 5: FILL VOLUME (3D ENTITY) BETWEEN SURFACES

Surface Loop(1) = {2, 1, 4, 8, 7, 6, 5, 3};
Surface Loop(2) = {9, 12, 11, 10, 14, 13, 16, 15};
Volume(1) = {1, 2};

// STEP 6: CREATE PHYSICAL GROUPS OF SURFACES

// Outer surface
Physical Surface(1) = {1, 2, 3, 4, 5, 7, 8, 6};

// Top segment of inner surface
Physical Surface(2) = {9, 12, 11, 10};

// Bottom segment of inner surface
Physical Surface(3) = {13, 14, 15, 16};

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
