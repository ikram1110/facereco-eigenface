#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <time.h>

const int Faces = 40;
const int Samples = 9;
const int Width = 92;
const int Height = 112;
const int Eigenfaces = 28;
const std::string DataPath = "faces/";
const int N = Faces;
const int M = Width * Height;
const std::string SampleName = "10";
const int MaxValue = 255;
double **facearray, **A, **B, **S, **V, **U, **W, **X, **P, **Atrans, **tempS;
clock_t first, t, tsc;

void read_training_data() {
	facearray = (double**) malloc(Samples*sizeof(double*));
	for(int x=0; x<Samples; x++) {
		facearray[x] = (double*) malloc(M*sizeof(double));
		memset(facearray[x],0,M*sizeof(double));
	}
	for(int face=0; face<Faces; ++face) {
		// perulangan setiap foto
		for(int sample=0; sample<Samples; ++sample) {
			std::stringstream filename;
			filename << DataPath << "s" << face + 1 << "/" << sample  + 1 << ".pgm";
			std::ifstream image(filename.str().c_str());

			if (image.is_open()) {
				// baca foto
				
				std::string line;
				getline(image, line); // Skip P2 line
				getline(image, line); // Skip width line
				getline(image, line); // Skip height line
				getline(image, line); // Skip max value line
		
				int val;
				int x = 0;
				while(image >> val) {
				facearray[sample][x] = val;
					x++;
				}
				
				image.close();
			} else {
				std::cout << "Image was not opened.";
			}
		}

		// mencari citra rata-rata
		for(int x=0; x<M; ++x) {
			double sum = 0;
			for(int y=0; y<Samples; ++y) {
				sum += facearray[y][x];
			}
			A[face][x] = sum/Samples;
		}
	}
	free(facearray);
}

void create_mean_image() {
	// temukan mean image
	tsc = clock();
	#pragma omp parallel for
	for(int c=0; c<M; ++c) {
		double sum = 0;
		for(int r=0; r<N; ++r) {
			sum += A[r][c];
		}
		B[0][c] = sum / N;
	}
	tsc = clock() - tsc;
	printf("Execution time create mean image : %.2fms\n", (float)(tsc)/CLOCKS_PER_SEC*1000);

	// output citra rata-rata
	std::stringstream filename;
	filename << "output/meanimage.pgm";
	std::ofstream image_file(filename.str().c_str());
	image_file << "P2" << std::endl << Width << std::endl << Height << std::endl << MaxValue << std::endl;
	for(int x=0; x<M; ++x) {
		int val = B[0][x];
		if(val < 0) {
			val = 0;
		}
		image_file << val << " ";
	}
	image_file.close();
}

void normalized() {
	// kurangkan mean dari setiap gambar
	tsc = clock();
	#pragma omp parallel for
	for(int r=0; r<N; ++r) {
		for(int c=0; c<M; ++c) {
			A[r][c] -= B[0][c];
			if(A[r][c] < 0) {
				A[r][c] = 0;
			}
		}
	}
	tsc = clock() - tsc;
	printf("Execution time mean subtraction : %.2fms\n", (float)(tsc)/CLOCKS_PER_SEC*1000);
	
	// output citra normalisasi
	for(int x=0; x<N; ++x) {
		std::ostringstream filename;
		filename << "output/normalized/" << x << ".pgm";
		std::ofstream image_file(filename.str().c_str());
		image_file << "P2" << std::endl << Width << std::endl << Height << std::endl << MaxValue << std::endl;
		for(int y=0; y<M; ++y) {
			int val = A[x][y];
			if(val < 0) {
				val = 0;
			}
			image_file << val << " ";
		}
		image_file.close();
	}
}

void transpose_matrixA() {
	for(int r=0; r<M; ++r) {
		for(int c=0; c<N; ++c) {
			Atrans[r][c] = A[c][r];
		}
	}
}

void get_covariant_matrix() {
	int cl, kl;
	#pragma omp parallel for private(cl,kl)
	// #pragma omp parallel for private(cl,kl)
	// #pragma omp parallel for schedule(dynamic, 8)
	// #pragma omp parallel for schedule(dynamic, 8) private(cl,kl)
	for(int r=0; r<N; ++r) {
		for(cl=0; cl<N; ++cl) {
			S[r][cl] = 0;
			for(kl=0; kl<M; ++kl) {
				S[r][cl] += A[r][kl] * Atrans[kl][cl];
			}
		}
	}
}

void calculate_eigenvalues() {
	// eigenvector pada matriks covarian
	// eigensystem(S).second.transpose()
	for(int r=0; r<N; ++r) {
		for(int c=0; c<N; ++c) {
			if (c == r) {
				P[c][r] = 1;
			}
		}
	}
	// std::cout<<"P[0][2] : "<<P[0][2]<<std::endl;
	int max_rotation = 5 * N * N;
	for(int i=0; i<N; ++i) {
		for(int j=0; j<N; ++j) {
			tempS[i][j] = S[i][j];
		}
	}
	
	double *eigenvalues;
	eigenvalues = (double*) malloc(N*sizeof(double));
	int small = 0;
	
	for(int it=0; it<max_rotation; ++it) {
		double max = 0;
		int k, l;
		// temukan elemen off-diagonal terbesar
		for(int r=0; r<N-1; ++r) {
			for(int c=r+1; c<N; ++c) {
				if (fabs(tempS[r][c]) >= max) {
					max = fabs(tempS[r][c]);
					k = r;
					l = c;
				}
			}
		}
		
		if(max < 1.0e-12) {
			for(int i=0; i<N; ++i) {
				eigenvalues[i] = tempS[i][i];
			}
			// normalisasi P
			#pragma omp parallel for
			for(int c=0; c<N; ++c) {
				double length = 0;
				for(int r=0; r<N; ++r) {
					length += P[r][c] * P[r][c];
				}
				for(int r=0; r<N; ++r) {
					P[r][c] = P[r][c] / length;
				}
			}
			small = 1;
			break;
		}
		else {
			// melakukan rotasi
			double diff = tempS[l][l] - tempS[k][k];
			
			double t;
			if(fabs(tempS[k][l]) < fabs(diff)*1.0e-36) {
				t = tempS[k][l] / diff;
			}
			else {
				double phi = diff / (2.0 * tempS[k][l]);
				t = 1.0 / (fabs(phi) + sqrt(phi * phi + 1.0));
				if(phi < 0) {
					t = -t;
				}
			}
			double c = 1.0 / sqrt(t * t + 1.0);
			double s = t * c;
			double tau = s / (1.0 + c);
			double temp = tempS[k][l];
			tempS[k][l] = 0;
			tempS[k][k] = tempS[k][k] - t * temp;
			tempS[l][l] = tempS[l][l] + t * temp;
			
			for(int i=0; i<k; ++i) {
				temp = tempS[i][k];
				tempS[i][k] = temp - s * (tempS[i][l] + tau * temp);
				tempS[i][l] = tempS[i][l] + s * (temp - tau * tempS[i][l]);
			}
			for(int i=k+1; i<l; ++i) {
				temp = tempS[k][i];
				tempS[k][i] = temp - s * (tempS[i][l] + tau * tempS[k][i]);
				tempS[i][l] = tempS[i][l] + s * (temp - tau * tempS[i][l]);
			}
			for(int i=l+1; i<N; ++i) {
				temp = tempS[k][i];
				tempS[k][i] = temp - s * (tempS[l][i] + tau * temp);
				tempS[l][i] = tempS[l][i] + s * (temp - tau * tempS[l][i]);
			}
			for(int i=0; i<N; ++i) {
				temp = P[i][k];
				P[i][k] = temp - s * (P[i][l] + tau * P[i][k]);
				P[i][l] = P[i][l] + s * (temp - tau * P[i][l]);
			}
		}
	}
	
	if(small == 0) {
		std::cout << "Metode Jacobi tidak sesuai." << std::endl;
		for(int i=0; i<N; ++i) {
			eigenvalues[i] = tempS[i][i];
		}
	}

	// urutkan berdasarkan eigenvalues
	// first = eigenvalues
	// second = P
	for(int i=0; i<N-1; ++i) {
		int index = i;
		double value = eigenvalues[i];
		for(int j=i+1; j<N; ++j) {
			if(eigenvalues[j] > value) {
				index = j;
				value = eigenvalues[j];
			}
		}
		if(index != i) {
			std::swap(eigenvalues[i], eigenvalues[index]);
			for(int r=0; r<N; ++r) {
				std::swap(P[r][i], P[r][index]);
			}
		}
	}
	for(int r=0; r<N; ++r) {
		for(int c=0; c<N; ++c) {
			V[r][c] = P[c][r];
		}
	}
}

void calculate_eigenfaces() {
	double **Urow, **eigenface;
	Urow = (double**) malloc(1*sizeof(double*));
	for(int x=0; x<1; x++) {
		Urow[x] = (double*) malloc(M*sizeof(double));
		memset(Urow[x],0,M*sizeof(double));
	}
	eigenface = (double**) malloc(1*sizeof(double*));
	for(int x=0; x<1; x++) {
		eigenface[x] = (double*) malloc(M*sizeof(double));
		memset(eigenface[x],0,M*sizeof(double));
	}

	for(int r=0; r<Eigenfaces; ++r) {
		for(int c=0; c<M; ++c) {
			eigenface[0][c] = 0;
		}
		for(int re=0; re<1; re++) {
			for(int c=0; c<M; ++c) {
				for(int k=0; k<N; ++k) {
					eigenface[re][c] += V[r][k] * A[k][c];
				}
			}
		}
		
		for(int c=0; c<M; ++c) {
			U[r][c] = eigenface[0][c];
		}
		
		double norm = 0;
		for(int i=0; i<M; i++) {
			norm += pow(U[r][i], 2);
		}
		norm = sqrt(norm);
		for(int i=0; i<M; i++) {
			U[r][i] /= norm;
		}
		
		// output eigenface
		
		// eigenface <- scale(U[r])
		// temukan minimum dan maksimum saat ini
		for(int c=0; c<M; ++c) {
			Urow[0][c] = U[r][c];
		}
		
		double min = 0, max = 255;
		double m_min = Urow[0][0];
		double m_max = Urow[0][0];
		for(int rs=0; rs<1; ++rs) {
				for(int c=0; c<M; ++c) {
						if(Urow[rs][c] < m_min) {
								m_min = Urow[rs][c];
						}
						if(Urow[rs][c] > m_max) {
								m_max = Urow[rs][c];
						}
				}
		}
		
		double old_range = m_max - m_min;
		double new_range = max - min;
		
		// buat matriks baru dengan elemen berskala
		for(int re=0; re<1; ++re) {
			for(int c=0; c<M; ++c) {
				eigenface[re][c] = (Urow[re][c] - m_min) * new_range / old_range + min;
			}
		}
		
		std::ostringstream filename;
		filename << "output/eigenfaces/" << r << ".pgm";
		
		// write pgm
		std::ofstream image_file(filename.str().c_str());
		image_file << "P2" << std::endl << Width << std::endl << Height << std::endl << MaxValue << std::endl;
		for(int c=0; c<M; ++c) {
			int val = eigenface[0][c];
			if(val < 0) {
				val = 0;
			}
			image_file << val << " ";
		}
		image_file.close();
	}
	
	free(Urow);
	free(eigenface);
}

void calculate_weight() {
	double **Arow, **ArowTrans;
	Arow = (double**) malloc(1*sizeof(double*));
	for(int x=0; x<1; x++) {
		Arow[x] = (double*) malloc(M*sizeof(double));
		memset(Arow[x],0,M*sizeof(double));
	}
	ArowTrans = (double**) malloc(M*sizeof(double*));
	for(int x=0; x<M; x++) {
		ArowTrans[x] = (double*) malloc(1*sizeof(double));
		memset(ArowTrans[x],0,1*sizeof(double));
	}

	for(int re=0; re<Eigenfaces; ++re) {
		for(int ce=0; ce<N; ++ce) {
			double befW = 0;
			for(int c=0; c<M; ++c) {
				Arow[0][c] = A[ce][c];
			}
			
			// A[ce] transpose
			for(int r=0; r<M; ++r) {
				for(int c=0; c<1; ++c) {
					ArowTrans[r][c] = Arow[c][r];
				}
			}
			
			// befW = U[re] * A[ce] transpose
			for(int r=0; r<1; r++) {
				for(int c=0; c<1; ++c) {
					for(int k=0; k<M; ++k) {
						befW += U[re][k] * ArowTrans[k][c];
					}
				}
			}
			
			W[re][ce] = befW;
		}
	}
	
	free(Arow);
	free(ArowTrans);
}

void calculate_accuracy() {
	double **Wx, **Xtrans;
	Wx = (double**) malloc(Eigenfaces*sizeof(double*));
	for(int x=0; x<Eigenfaces; x++) {
		Wx[x] = (double*) malloc(1*sizeof(double));
		memset(Wx[x],0,1*sizeof(double));
	}
	Xtrans = (double**) malloc(M*sizeof(double*));
	for(int x=0; x<M; x++) {
		Xtrans[x] = (double*) malloc(1*sizeof(double));
		memset(Xtrans[x],0,1*sizeof(double));
	}

	double accuracy = 0;
	for(int i=1; i<=N; ++i) {
		// baca sample gambar
		std::stringstream filesample;
		filesample << DataPath << "s" << i << "/" << SampleName << ".pgm";
		std::ifstream imagesample(filesample.str().c_str());
		
		if (imagesample.is_open()) {
			// baca foto
			std::string line;
			getline(imagesample, line); // Skip P2 line
			getline(imagesample, line); // Skip width line
			getline(imagesample, line); // Skip height line
			getline(imagesample, line); // Skip max value line
	
			int val;
			int x = 0;
			while(imagesample >> val) {
			X[0][x] = val;
				x++;
			}
			
			imagesample.close();
		} else {
			std::cout << "Image was not opened.";
		}
        
		// pengenalan (X, B, U, W)
		// kurangi gambar rata-rata
		for(int c=0; c<M; ++c) {
			X[0][c] -= B[0][c];
			if(X[0][c] < 0) {
				X[0][c] = 0;
			}
		}
		// temukan bobot
		for(int rw=0; rw<Eigenfaces; ++rw) {
			double befWx = 0;
			
			// transpose X
			for(int r=0; r<M; ++r) {
				for(int c=0; c<1; ++c) {
					Xtrans[r][c] = X[c][r];
				}
			}
			
			// befWx = U[rw] * X transpose
			for(int r=0; r<1; r++) {
				for(int c=0; c<1; ++c) {
					for(int k=0; k<M; ++k) {
						befWx += U[rw][k] * Xtrans[k][c];
					}
				}
			}
			Wx[rw][0] = befWx;
		}
		
		// temukan wajah terdekat dari set pelatihan
		double min_distance = 0;
		int image_number = 0;
		for(int img=0; img<N; ++img) {
			double distance = 0;
			for(int eface=0; eface<Eigenfaces; ++eface) {
				distance += fabs(W[eface][img] - Wx[eface][0]);
			}
			if(distance < min_distance || img == 0) {
				min_distance = distance;
				image_number = img;
			}
		}
        
		std::cout << i << ". " << image_number + 1 << std::endl;
		if(i == image_number + 1) {
			accuracy = accuracy + 1;
		}
	}
	
	std::cout << "Accuracy : " << std::fixed << std::setprecision(2) << accuracy / N << std::endl;
	
	free(Wx);
	free(Xtrans);
}

int main(int argc, char *argv[]) {
	// if(argc == 2){
  //   Eigenfaces = atoi(argv[1]);
  // }

	srand(time(NULL));
	first = clock();

	omp_set_num_threads(8);
	
	// A berisi gambar sebagai baris. A adalah NxM, [A] i, j adalah nilai piksel ke-j gambar ke-i.
	A = (double**) malloc(N*sizeof(double*));
	for(int x=0; x<N; x++) {
		A[x] = (double*) malloc(M*sizeof(double));
		memset(A[x],0,M*sizeof(double));
	}
	
	// baca data latih
	t = clock();
	read_training_data();
	t = clock() - t;
	printf("Execution time read training data : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	// B berisi citra rata-rata. B adalah matriks 1xM, [B] 0, j adalah nilai piksel ke-j dari citra rata-rata.
	B = (double**) malloc(1*sizeof(double*));
	for(int x=0; x<1; x++) {
		B[x] = (double*) malloc(M*sizeof(double));
		memset(B[x],0,M*sizeof(double));
	}
	
	t = clock();
	create_mean_image();
	t = clock() - t;
	printf("Execution time create mean image : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	t = clock();
	normalized();
	t = clock() - t;
	printf("Execution time normalized : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	// transpose matriks A
	Atrans = (double**) malloc(M*sizeof(double*));
	for(int x=0; x<M; x++) {
		Atrans[x] = (double*) malloc(N*sizeof(double));
		memset(Atrans[x],0,N*sizeof(double));
	}
	t = clock();
	transpose_matrixA();
	t = clock() - t;
	printf("Execution time transpose matrix A : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	// matriks covarian [A * Atranspose]
	// S -> M*M
	// tempS -> M*M
	S = (double**) malloc(N*sizeof(double*));
	tempS = (double**) malloc(N*sizeof(double*));
	P = (double**) malloc(N*sizeof(double*));
	V = (double**) malloc(N*sizeof(double*));
	for(int x=0; x<N; x++) {
		S[x] = (double*) malloc(N*sizeof(double));
		tempS[x] = (double*) malloc(N*sizeof(double));
		P[x] = (double*) malloc(N*sizeof(double));
		V[x] = (double*) malloc(N*sizeof(double));
		memset(S[x],0,N*sizeof(double));
		memset(tempS[x],0,N*sizeof(double));
		memset(P[x],0,N*sizeof(double));
		memset(V[x],0,N*sizeof(double));
	}
	
	t = clock();
	// serial
	get_covariant_matrix();
	t = clock() - t;
	printf("Execution time get covariant matrix (Multiply the matriks A and its transpose) : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	t = clock();
	calculate_eigenvalues();
	free(tempS);
	t = clock() - t;
	printf("Execution time get eigenvalues : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	// temukan eigenface
	U = (double**) malloc(Eigenfaces*sizeof(double*));
	for(int x=0; x<Eigenfaces; x++) {
		U[x] = (double*) malloc(M*sizeof(double));
		memset(U[x],0,M*sizeof(double));
	}
	
	t = clock();
	calculate_eigenfaces();
	t = clock() - t;
	printf("Execution time get eigenfaces : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	// temukan bobot
	W = (double**) malloc(Eigenfaces*sizeof(double*));
	for(int x=0; x<Eigenfaces; x++) {
		W[x] = (double*) malloc(N*sizeof(double));
		memset(W[x],0,N*sizeof(double));
	}
	
	t = clock();
	calculate_weight();
	t = clock() - t;
	printf("Execution time get weight : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	
	// menghitung akurasi
	X = (double**) malloc(1*sizeof(double*));
	for(int x=0; x<1; x++) {
		X[x] = (double*) malloc(M*sizeof(double));
		memset(X[x],0,M*sizeof(double));
	}
	
	t = clock();
	calculate_accuracy();
	t = clock() - t;
	printf("Execution time get accuracy : %.2fms\n", (float)(t)/CLOCKS_PER_SEC*1000);
	first = clock() - first;
	printf("Total Execution time : %.2fms\n", (float)(first)/CLOCKS_PER_SEC*1000);
	
	free(X);
	free(A);
	free(Atrans);
	free(B);
	free(S);
	free(P);
	free(U);
	free(V);
	free(W);
  return 0;
}