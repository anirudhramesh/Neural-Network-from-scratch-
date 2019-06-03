#include <iostream>
#include <random>
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
using namespace std;

random_device seed;
default_random_engine randEngine(seed());
uniform_real_distribution<double> distribution(-1,1);

class neuron
{
	vector<double> weight;
	public:
		neuron() { }
		neuron(int nextLayerNodeCount){
			weight.resize(nextLayerNodeCount, distribution(randEngine));
		}
		neuron(vector<double>& weights){
			weight.resize(weights.size());
			for(int i=0; i<weights.size(); i++)
				weight[i] = weights[i];
		}
		const double operator[](int index) const{
				return weight[index];
		}
		double& operator[](int index){
				return weight[index];
		}
		int size(){
			return weight.size();
		}
};

class hiddenLayer
{
	vector<neuron> layerNeurons;
	neuron layerBias; //input for bias nodes is always just an identity value
	public:
		hiddenLayer(int neuronCount, int nextLayerNodeCount){
			layerNeurons.resize(neuronCount, nextLayerNodeCount);
			layerBias = neuron(nextLayerNodeCount);
		}
		hiddenLayer(vector<vector<double> >& weights){ // assume first element of the vector of vectors is weights for the bias variable
			layerBias = neuron(weights[0]);
				
			for(int i=1; i<weights.size(); i++)
				layerNeurons.push_back(neuron(weights[i]));
		}
		const neuron operator[](int index) const{
			if (index==0)
				return layerBias;
			else
				return layerNeurons[index-1];
		}
		neuron& operator[](int index){
			if(index==0)
				return layerBias;
			else
				return layerNeurons[index-1];
		}
		int size(){
			return layerNeurons.size();
		}
};

class ANN
{
	vector<hiddenLayer> layers;
	double learningRate = 0.005, errorTolerance = 0.005, maxCycles=100;
	public:
		ANN(int, int*);
		ANN(string);
		ANN() { };
		
		void train(string inputFile = "TrainingData.csv", string outputFile = "TrainingDataDependentVariable.csv");
		void Predict(string testFile = "TestingData.csv");
};

double activationFunction(double);
double activationDerivative(double);

int main()
{
	ANN myNeuralNetwork;
	int option, layers, *neurons;
	string inputFile, outputFile, weightsFile;
	
	do{
		cout << endl << "Menu:"
			 << endl << "1) Create a new neural network"
			 << endl << "2) Load pretrained network and make a prediction"
			 << endl << "3) Quit"
			 << endl << "Enter option: ";
		cin >> option;
		
		switch (option)
		{
			case 1 : cout << endl << "Enter the number of layers (excluding input and output layers): "; cin >> layers;
					 cout << "Enter the number of neurons in each layer:" << endl;
					 neurons = new int [layers];
					 for(int i=0; i<layers; i++){
					 	cout << "Layer " << i+1 << ": ";
					 	cin >> neurons[i];
					 }
					 myNeuralNetwork = ANN(layers, neurons);
					 cout << "Enter the file name for input data (to use default \"TrainingData.csv\", enter nothing): "; cin.get();
					 getline(cin, inputFile);
					 cout << "Enter the file name for in sample output data (to use default \"TrainingDataDependentVariable.csv\", enter nothing): ";
					 getline(cin, outputFile);
					 if(!inputFile.empty() && !outputFile.empty())
					 	myNeuralNetwork.train(inputFile, outputFile);
					else
						myNeuralNetwork.train();
					 break;
					 
			case 2 : cout << endl << "Enter the name of the weights file to load (to use default \"NetworkWeights.csv\", enter nothing): "; cin.get();
					 getline(cin, weightsFile);
					 if(!weightsFile.empty())
					 	myNeuralNetwork = ANN(weightsFile);
					 else
					 	myNeuralNetwork = ANN("NetworkWeights.csv");
					 cout << "Enter input file name to make a prediction (to use default \"TestingData.csv\", enter nothing): ";
					 getline(cin, inputFile);
					 if(!inputFile.empty())
					 	myNeuralNetwork.Predict(inputFile);
					else
					 	myNeuralNetwork.Predict();
					 break;
					 
			case 3 : cout << endl << endl << "Thank You!" << endl << endl;
					 break;
					 
			default: cout << endl << "Invalid option! Select again." << endl << endl;
		}
	} while (option != 3);
	
	return 0;
}

ANN::ANN(int layerCount, int *neurons)
{
	//input layer not initialised here
	for(int i=0; i<layerCount-1; i++)
		layers.push_back(hiddenLayer(neurons[i], neurons[i+1]));
	layers.push_back(hiddenLayer(neurons[layerCount-1], 1)); //last layer has only one output
}

ANN::ANN(string weightsFile)
{
	fstream f;
	f.open(weightsFile, ios::in);
	
	string line;
	stringstream s;
	vector< vector<double> > readWeights;
	
	// Weights file contains input layer Weights at the start
	// each layer is separated from the next one by an empty line
	// the first line of Weights in each layer is the bias Weights
	for(int i=0; !f.eof(); i++)
	{		
		f >> line;
		if(line[0]==','){ // checking for empty line between layers and initialising the layer with the individual readWeights recorded until then
			layers.push_back(hiddenLayer(readWeights));
			readWeights.clear();
			i=-1;
			continue;
		}
		readWeights.resize(i+1);
		s << line;
		while(getline(s, line, ','))
			readWeights[i].push_back(stod(line));
		
		s.clear();
	}
	layers.push_back(hiddenLayer(readWeights));
	
	f.close(); f.clear();
}

void ANN::train(string inputFile, string outputFile)
{
	fstream f, f1;
	string line, temp;
	stringstream ss;
	
	f.open(inputFile, ios::in);
	f1.open(outputFile, ios::in);
	
	double inputSize,actualOutput, totalError, finalOutput;
	vector<double> *inputs;
	inputs = new vector<double> [layers.size()+1];
	int cycle=0, nodes=0;
	
	f >> line;
	f.seekg(0, ios::beg);
	inputSize = count(line.begin(), line.end(), ',') + 1;
	
	layers.insert(layers.begin(), hiddenLayer(inputSize, layers[0].size()));
	nodes += inputSize;
	inputs[0].resize(layers[0].size()+1);
	for(int i=1; i<layers.size(); i++){
		inputs[i].resize(layers[i].size()+1);
		nodes += layers[i].size()+1;
	}
	
	do{
		totalError = 0;
		while(!f.eof())
		{
			//forward propogation
			f >> line; ss << line;
			inputs[0][0]=1;
			for(int i=1; getline(ss, line, ','); i++)
				inputs[0][i] = stod(line);
			f1 >> actualOutput;
			
			for(int i=0; i<layers.size()-1; i++)
			{
				for(int j=0; j<layers[i+1].size()+1; j++)
				{
					if(j==0){
						inputs[i+1][j]=1;
						continue;
					}
					inputs[i+1][j] = layers[i][0][j];
					for(int k=0; k<layers[i].size()+1; k++)
						inputs[i+1][j] += inputs[i][k] * layers[i][k][j];
					inputs[i+1][j] = activationFunction(inputs[i+1][j]);
				}
			}
			
			finalOutput = layers[layers.size()-1][0][0];
			for(int i=0; i<layers[layers.size()-1].size(); i++)
				finalOutput += inputs[layers.size()-1][i] * layers[layers.size()-1][i][0];
			
			
			//back propagation
			double deltaY = (finalOutput - activationFunction(finalOutput)) * activationDerivative(finalOutput);
			totalError+=deltaY;
			
			for(int i=0; i<layers[layers.size()-1].size()+1; i++)
			{
				totalError += learningRate * deltaY * finalOutput;
				layers[layers.size()-1][i][0] += learningRate * deltaY * finalOutput;
			}
			
			for(int i=layers.size()-2; i>-1; i--)
				for(int j=0; j<layers[i].size()+1; j++)
					for(int k=0; k<layers[i+1].size()+1; k++){
						totalError += learningRate * deltaY * inputs[i+1][k];
						layers[i][j][k] += learningRate * deltaY * inputs[i+1][k];
					}
					
			ss.clear();
		}
		totalError /= nodes;
		f.clear(); f1.clear();
		f.seekg(0, ios_base::beg); f1.seekg(0, ios_base::beg);
		cycle++;
	} while(cycle < maxCycles && abs(totalError) > errorTolerance);
	f.close(); f.clear();
	f1.close(); f1.clear();
	
	f.open("NetworkWeights.csv", ios::out);
	for(int i=0; i<layers.size(); i++)
	{
		for(int j=0; j<layers[i].size()+1; j++)
		{
			for(int k=0; k<layers[i][j].size(); k++)
			{
				f << layers[i][j][k];
				if(k!=layers[i][j].size()-1)
					f << ",";
			}
			f << endl;
		}
		if(i!=layers.size()-1)
			f << "," << endl;
	}
	f.close(); f.clear();
	
	cout << "Weights trained and saved into NetworkWeights.csv file." << endl
		 << "Final error value was: " << totalError << endl
		 << "No. of cycles calculated was: " << cycle << endl;
}

void ANN::Predict(string testFile)
{
	fstream f, f1;
	string line;
	stringstream s;
	f.open(testFile, ios::in);
	f1.open("TestDataOutput.csv", ios::out);
	
	double finalOutput;
	vector<double> *inputs;
	inputs = new vector<double> [layers.size()];
	
	inputs[0].resize(layers[0].size()+1);
	for(int i=1; i<layers.size(); i++)
		inputs[i].resize(layers[i].size());
	
	while(!f.eof())
	{
		//forward propogation
		f >> line; s << line;
		for(int i=1; getline(s, line, ','); i++)
			inputs[0][i] = stod(line);
		
		for(int i=0; i<layers.size()-1; i++)
		{
			for(int j=0; j<layers[i+1].size()+1; j++)
			{
				if(j==0){
					inputs[i+1][j]=1;
					continue;
				}
				inputs[i+1][j] = layers[i][0][j];
				for(int k=0; k<layers[i].size()+1; k++){
					inputs[i+1][j] += inputs[i][k] * layers[i][k][j];
				}
				inputs[i+1][j] = activationFunction(inputs[i+1][j]);
			}
		}
		
		finalOutput = layers[layers.size()-1][0][0];
		for(int i=0; i<layers[layers.size()-1].size(); i++)
			finalOutput += inputs[layers.size()-1][i] * layers[layers.size()-1][i][0];
		
		f1 << finalOutput << endl;
		s.clear();
	}
	
	f.close(); f.clear();
	f1.close(); f1.clear();
	
	cout << "Predicted output saved in TestDataOutput.csv" << endl;
}

double activationFunction(double val)
{
	return 2/(1+exp(-val))-1;
}

double activationDerivative(double val)
{
	return 0.5*(1+activationFunction(val))*(1-activationFunction(val));
}
