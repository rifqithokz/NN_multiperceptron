#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <array>
#include <math.h>

using namespace std;

const int jlhData = 4;
const int jlhFitur = 2;
const int jlhBiasInput = 1;
const int jlhHiddenUnit = 2;
const int jlhBiashidden = 1;
const int epoch = 100;
												//bias, I1, I2
double inputData[jlhData][jlhFitur + jlhBiasInput] = { {1,0,0}, {1,0,1}, {1,1,0}, {1,1,1} };
double target[jlhData] = { 0,1,1,0 };
double weightInput[jlhHiddenUnit][jlhFitur+jlhBiasInput];

double weightHidden[jlhHiddenUnit + jlhBiashidden];
double sumInput[jlhHiddenUnit];
double inputHidden[jlhBiashidden + jlhHiddenUnit];	//urutannya>> bias, sumInput1, sumInput2
double sumHidden;
double outHidden;
double miu = 0.1;
double squareError;
double mSE[epoch];

template<size_t SIZE, class T> inline size_t array_size(T(&arr)[SIZE]) {
	return SIZE;
}
double getRandom();
void saveMSEtoCSV();
void saveWeighttoCSV();

int main() {
	int i, j, k, l;

	//input weight manual
	//weightInput[0][0] = -0.4;	//weight bias	hidden unit 1
	//weightInput[0][1] = -0.2;	//weight I1		hidden unit 1
	//weightInput[0][2] = 0.8;	//weight I2		hidden unit 1
	//weightInput[1][0] = -0.9;	//weight bias	hidden unit 2
	//weightInput[1][1] = 0;		//weight I1		hidden unit 2
	//weightInput[1][2] = -0.7;	//weight I2		hidden unit 2
	//weightHidden[0] = 0.5;
	//weightHidden[1] = -0.7;
	//weightHidden[2] = -0.2;
	clock_t t;
	t = clock();

	srand(time(NULL));
	//generate random number untuk weight input
	for (i = 0; i < jlhHiddenUnit; i++) {
		for (j = 0; j < jlhFitur+jlhBiasInput; j++) {
			weightInput[i][j] = getRandom();
			cout << "weight input [" << i << "][" << j << "] : " << weightInput[i][j] << endl;
		}
	}

	//generate random number untuk weight hidden
	for (i = 0; i < jlhHiddenUnit + jlhBiashidden; i++) {
		weightHidden[i] = getRandom();
		cout << "weight hidden [" << i << "] : " << weightHidden[i] << endl;
	}
	
	cout << "===============================================\n";
	cout << "                Learning\n";
	cout << "===============================================\n";
	
	i = 0;
	j = 0;
	for (i = 0; i < epoch; i++) {
		cout << "iterasi ke-" << i + 1 << " . . . ";
		squareError = 0;
		for (j = 0; j < jlhData; j++) {
			//Feed Forward
			////input ke hidden
			//////masukkan nilai bias hidden layer ke inputHiddenUnit
			for (k = 0; k < jlhBiashidden; k++) {
				inputHidden[k] = 1;
			}
			//////hitung nilai sumInput, sampai jadi inputHidden
			for (k = 0; k < jlhHiddenUnit; k++) {
				//init sumInput
				sumInput[k] = 0;
				////hitung sumInput
				for (l = 0; l < jlhBiasInput + jlhFitur; l++) {
					sumInput[k] += inputData[j][l] * weightInput[k][l];
				}
				//cout << "sumInput[" << k << "] : " << sumInput[k] << endl;
				////masukkan sumInput ke inputHidden, proses dulu pakai fungsi aktivasi(sigmoid)
				inputHidden[jlhBiashidden + k] = 1 / (1 + (pow(10, ((-1) * sumInput[k]))));
			}

			//////tampilkan nilai di inputHidden
			//cout << "\n\ninput hidden : \n";
			//for (k = 0; k < array_size(inputHidden); k++) {
			//	cout << inputHidden[k] << endl;
			//}

			//////hitung nilai sumHidden
			sumHidden = 0;	//init
			for (k = 0; k < array_size(inputHidden); k++) {
				sumHidden += inputHidden[k] * weightHidden[k];
			}
			//cout << "\nsumHidden : " << sumHidden << endl;

			//////fungsi aktivasi, sigmoid untuk dapat outHidden
			outHidden = 1 / (1 + (pow(10, ((-1) * sumHidden))));
			//cout << "\noutHidden : " << outHidden << endl;

			squareError += pow((target[j] - outHidden), 2);

			//Back Propagation
			////hitung dC ==> balik outHidden
			double dC = outHidden * (1 - outHidden) * (target[j] - outHidden);
			//cout << "\ndC : " << dC << endl;
			////hitung dH ==> balik inputHidden ke sumInput
			double dH[jlhHiddenUnit];
			for (k = 0; k < jlhHiddenUnit; k++) {
				int idx = k + jlhBiashidden;
				dH[k] = inputHidden[idx] * (1 - inputHidden[idx]) * weightHidden[idx] * dC;
			//	cout << "iH[" << idx << "] : " << inputHidden[idx] << ", wH[" << idx << "] : " << weightHidden[idx] << " >> dH[" << k+1 << "] : " << dH[k] << endl;
			}

			////update weightHidden
			//cout << "\nUpdate weightHidden : \n";
			for (k = 0; k < array_size(weightHidden); k++) {
				double dwH = miu * inputHidden[k] * dC;
				weightHidden[k] += dwH;
			//	cout << "weightHidden[" << k << "] : " << weightHidden[k] << endl;
			}

			////update weightInput
			for (k = 0; k < jlhHiddenUnit; k++) {
				for (l = 0; l < jlhBiasInput + jlhFitur; l++) {
					double dwi = miu * inputData[j][l] * dH[k];
				//	cout << "dwI[" << k << "][" << l << "] : " << dwi << endl;
					weightInput[k][l] += dwi;
				}
			}
			//cout << "\nUpdate weightInput : \n";
			//for (k = 0; k < jlhHiddenUnit; k++) {
			//	for (l = 0; l < jlhBiasInput + jlhFitur; l++) {
			//		cout << "weightInput[" << k << "][" << l << "] : " << weightInput[k][l] << endl;
			//	}
			//}
		}
		mSE[i] = squareError / jlhData;
		cout << " >> selesai\n";
	}
	t = clock() - t;
	
	cout << "\n\n===============================================\n";
	cout << "                Weight update\n";
	cout << "===============================================\n";
	for (i = 0; i < jlhHiddenUnit; i++) {
		for (j = 0; j < jlhFitur + jlhBiasInput; j++) {
			cout << "weight input [" << i << "][" << j << "] : " << weightInput[i][j] << endl;
		}
	}

	for (i = 0; i < jlhHiddenUnit + jlhBiashidden; i++) {
		cout << "weight hidden [" << i << "] : " << weightHidden[i] << endl;
	}
	
	//cout << "\n\nMSE[" << i << "] : " << mSE[i] << endl;
	cout << "\n\nSimpan MSE ke csv...";
	saveMSEtoCSV();
	cout << " >> selesai\n";

	cout << "Simpan weight terupdate ke csv...";
	saveWeighttoCSV();
	cout << " >> selesai\n\n";

	cout << "Learning time : " << t << " miliseconds => " << t * 1.0 / CLOCKS_PER_SEC << " seconds" << endl;

	system("pause");
	return 0;
}

double getRandom() {
	//generate angka dari -10 sampai 10, kemudian dibagi 10 biar dapat -1 sampai 1
	return (((double)(rand() % 20) + (-10)) / 10);
}

void saveMSEtoCSV()
{
	fstream fout;

	fout.open("MSE.csv", ios::out);

	int i;
	fout << "Iterasi" << "," << "MSE\n";
	for (i = 0; i < epoch; i++) {
		fout << i + 1 << "," << mSE[i] << "\n";
	}
	fout.close();
}

void saveWeighttoCSV()
{
	fstream fout;
	fout.open("weight.csv", ios::out);

	int i, j;
	fout << "weight input\n";
	for (i = 0; i < jlhHiddenUnit; i++) {
		fout << "Hidden layer ke-" << i + 1 << endl;
		for (j = 0; j < jlhBiasInput + jlhFitur; j++) {
			fout << "W[" << i << "][" << j << "]," << weightInput[i][j] << endl;
		}
	}

	fout << "\n\nweight hidden layer\n";
	for (i = 0; i < jlhBiashidden+jlhHiddenUnit; i++) {
		fout << "wH[" << i << "]," << weightHidden[i] << "\n";
	}
	fout.close();
}