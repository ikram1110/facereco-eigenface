# Face Recognize - Eigenface

## Description

Implementation of load imbalance in face recognition case studies with the eigenface algorithm using the C ++ programming language.

## Installation

For serial program with commands

```bash
$ g++ -Wall -std=c++11 main.cpp -fopenmp -o serial
```

And parallel program, use the OpenMP Library to compile the program with `-fopenmp` option

```bash
$ g++ -Wall -std=c++11 ompmain.cpp -fopenmp -o parallel
```

Then run it for serial program

```bash
$ ./serial
```

for parallel program

```bash
$ ./parallel
```

and with number of threads option with command

```bash
$ ./parallel [numberOfThread]
```

like ...

```bash
$ ./parallel 4
```

## Usage

In order to provide the images to the program, one should follow the structure:

- Each image should be in PGM format of the same resolution
- Each face has a separate folder named `s*`, where `*` is an identifier of the face from 1 to the number of faces
- Each face's images are in its folder, numbered from 1 to the number of samples

An example of such structure can be seen in the program's faces folder.

The next step is to determine the configuration at the top of the `main.cpp` file:

```c
const int Faces = 40;  // Number of faces
const int Samples = 9; // Number of samples per face
const int Width = 92; // Width of each image
const int Height = 112; // Height of each image
const int Eigenfaces = 28; // Number of eigenfaces taken in the algorithm
const std::string DataPath = "faces/"; // Path to the folder with data
const int N = Faces; // Total number of images
const int M = Width * Height; // Resolution
const std::string SampleName = "10"; // Name of the file to classify
const int MaxValue = 255; // Maximum pixel value of the images (see PGM format specification)
```

With this configuration, the program will train itself on 360 photos located in the `faces` folder, expecting each photo to have 92112 pixels. Then it will classify photos named `10.pgm` in each subfolder of the `faces` folder, using 28 eigenfaces. `MaxValue` is used in the output of images generated by a program.

## Output

The output of the program for the previously discussed configuration is:

```bash
Execution time create mean image : 31.90ms
Execution time mean subtraction : 376.91ms
Execution time get covariant matrix : 3650.40ms
Execution time get eigenvalues : 100.80ms
1. 1
2. 2
3. 3
4. 4
5. 5
6. 6
7. 7
8. 8
9. 9
10. 10
11. 11
12. 12
13. 13
14. 14
15. 15
16. 16
17. 17
18. 18
19. 19
20. 20
21. 21
22. 22
23. 38
24. 24
25. 25
26. 26
27. 27
28. 28
29. 29
30. 30
31. 31
32. 32
33. 33
34. 34
35. 40
36. 36
37. 37
38. 38
39. 22
40. 5
Accuracy : 1.00

```

The execution time shows the processing time step by step in the eigenface algorithm.

The 9th row can be read as: "The image named `10.pgm` in the `faces/s1/` folder is classified to be a photo of a person number 1".

`Accuracy:` row contains recognize rate.

## Additional output

The program also generates additional output that may be interesting to the user. It is located in the `output` folder.

- `eigenfaces` folder contains images given by the eigenvectros of the covariance matrix. They form a basis for the set of faces.
- `normalized` folder contains images from the training data with mean image subtracted.
- `meanimage.pgm` is the mean image of the training data.
