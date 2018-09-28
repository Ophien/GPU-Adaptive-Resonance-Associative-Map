//---------------------------------------------------------------------
// Inclusões principais CUDA
//---------------------------------------------------------------------

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <cuda_runtime.h>

// Deve definir depois do runtime para não dar pau
#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

//---------------------------------------------------------------------
// Includes c++
//---------------------------------------------------------------------

#include <stdio.h>		// Input e output do C
#include <iostream>		// Strem de dados para input e output
#include <math.h>		// Lib. de matemática
#include <vector>		// Estrutura de dados
//#include <Windows.h>	// Api do windows para utilizar chamadas de sistema
#include <climits>		// Definição dos limites máximos das variáveis
#include <string>		// Estrutura de dados para manipulação de strings
#include <sstream>		// Estrutura de dados para executar algoritmos em strings
#include <algorithm>	// Algoritmos para serem executados em estruturas de dados
#include <iterator>		// Iterators para manipulação de estruturas de dados
#include <cstring>		// String do C
#include <fstream>		// Leitura de arquivo
#include <time.h>
#include <omp.h>		// Open MP
#include <list>
#include <iomanip>      // std::setprecision

void debug(const char *fmt, ...);

//---------------------------------------------------------------------
// Namespaces utilizados
//---------------------------------------------------------------------

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

#define ARS_MAX_THREADS 1024
#define ARS_MAX_BLOCKS  1024 * 1024
#define ARS_MAX_SHARED_BLOCKS 1024
#define ARS_MIN_THREADS 1
#define ARS_MIN_BLOCKS  1
#define ARS_MAX_NEURONS 20000

//---------------------------------------------------------------------
// defines
//---------------------------------------------------------------------

//#define __VERBOSE__
//#define __T_VECTOR_DEBUG__
//#define __RESONANCE_VECTOR_DEBUG__
#define __PER_FIELD_SUM__
//#define __FIELD_REDUCTION__
//#define __DEBUG__

// -------------------------------------------------------------------------

std::string to_string(double val)
{
	std::stringstream stream;
	stream << val;
	return stream.str();
}

//---------------------------------------------------------------------
// Utilidades para o host
//---------------------------------------------------------------------
void split(const std::string& s, char c,
	std::vector<std::string>& v) {
	v.clear();
	std::string::size_type i = 0;
	std::string::size_type j = s.find(c);

	while (j != std::string::npos) {
		v.push_back(s.substr(i, j - i));
		i = ++j;
		j = s.find(c, j);

		if (j == std::string::npos)
			v.push_back(s.substr(i, s.length()));
	}
}

void calculateBlocks(int array_size, int* blocksInGrid, int* threadsInBlock){
	*threadsInBlock = ARS_MAX_THREADS;

	if (array_size < *threadsInBlock)
	{
		*threadsInBlock = array_size;
		*blocksInGrid = ARS_MIN_BLOCKS;
	}
	else{
		if (array_size == 0){
			*threadsInBlock = ARS_MIN_THREADS;
			*blocksInGrid = ARS_MIN_BLOCKS;
		}
		else{
			*blocksInGrid = array_size / *threadsInBlock;

			if ((*blocksInGrid) * (*threadsInBlock) < (array_size))
				*blocksInGrid += 1;
		}
	}
}

__global__ void fill(double* weights, int vecSize, double value){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < vecSize)
		weights[index] = value;
}

__global__ void setValue(double* out, double value, int pos){
	out[pos] = value;
}

//---------------------------------------------------------------------
// Redução simples utilizando bloco compartilhado
// double* input: entrada a ser reduzida
// double* globalBlockData: local onde a redução será executada
// int arrSize: tamanho total do vetor a ser reduzido
// NOTA: calcular blocos e threads de acordo com o tamanho de arrSize
//---------------------------------------------------------------------

__global__ void simple_reduction_shared(int* input, int* globalBlockData, int* arrSize_input){
	extern __shared__ int shared_operation_vector[];

	// Inicialização
	int threadId = threadIdx.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int arrSize = (*arrSize_input);

	// Faz copia para memoria de trabalho apenas quando for necessario, nesse caso ate o tamanho do vetor de busca
	if (index < arrSize && threadId < arrSize){
		shared_operation_vector[threadId] = input[index];
	}
	else{
		shared_operation_vector[threadId] = 0.0;
	}

	// Garantir que o loop nunca vai passar do tamanho do array
	int limit = arrSize;
	if (limit > blockDim.x)
		limit = blockDim.x;

	__syncthreads();

	// Reducao simples
	for (int stride = 1; stride < limit; stride *= 2) {
		int index_reduc = 2 * stride * threadId;

		// Garantir que o passo da redução nunca sairá do limite
		if (index_reduc < limit && (index_reduc + stride) < limit)
		{
			// Verifica qual é maios valor e armazena indice e valor em vetores globais
			shared_operation_vector[index_reduc] += shared_operation_vector[index_reduc + stride];
		}

		__syncthreads();
	}

	// Guarda o resultado na memoria global
	if (threadId == 0)
		globalBlockData[blockIdx.x] = shared_operation_vector[0];
}

//---------------------------------------------------------------------
// Redução simples para achar max utilizando bloco compartilhado
// double* input: entrada a ser reduzida
// double* globalBlockData: local onde a redução será executada
// int* globalBlockDataMax: local onde armazenar os indices dos máximos
// int arrSize: tamanho do espaço onde procurar pelo max
// bool: first: identifica se é a primeira execução para inicialização
// NOTA: calcular blocos e threads de acordo com o tamanho de arrSize
//---------------------------------------------------------------------

__global__ void simple_reduction_shared_max(double* input, double* globalBlockData, int* globalBlockDataMax, int* arrSize_input, bool first){
	extern __shared__ int shared_op_vec[];

	// Configurado variaveis
	double* shared_operation_vector = (double*)&shared_op_vec[blockDim.x];
	int   * shared_operation_vector_max = shared_op_vec;

	// Inicialização
	int threadId = threadIdx.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int arrSize = (*arrSize_input);

	// Faz copia para memoria de trabalho apenas quando for necessario, nesse caso ate o tamanho do vetor de busca
	if (index < arrSize && threadId < arrSize){
		shared_operation_vector[threadId] = input[index];

		// Os valores de indices que determinam o indice onde se encontra o maximo devem ser inicializados com um cuidado espacial,
		// Se for a primeira execução inicializar o indice, caso contrario inicializar com os dados já pre-calculados
		if (first)
			shared_operation_vector_max[threadId] = index;
		else
			shared_operation_vector_max[threadId] = globalBlockDataMax[index];
	}
	else{
		shared_operation_vector[threadId] = 0.0;
		shared_operation_vector_max[threadId] = -1;
	}

	// Garantir que o loop nunca vai passar do tamanho do array
	int limit = arrSize;
	if (limit > blockDim.x)
		limit = blockDim.x;

	__syncthreads();

	// Reducao simples
	for (int stride = 1; stride < limit; stride *= 2) {
		int index_reduc = 2 * stride * threadId;

		// Garantir que o passo da redução nunca sairá do limite
		if (index_reduc < limit && (index_reduc + stride) < limit)
		{
			// Verifica qual é maios valor e armazena indice e valor em vetores globais
			if (shared_operation_vector[index_reduc] < shared_operation_vector[index_reduc + stride])
			{
				shared_operation_vector[index_reduc] = shared_operation_vector[index_reduc + stride];
				shared_operation_vector_max[index_reduc] = shared_operation_vector_max[index_reduc + stride];
			}
		}

		__syncthreads();
	}

	// write result for this block to global mem
	if (threadId == 0) {
		globalBlockData[blockIdx.x] = shared_operation_vector[0];
		globalBlockDataMax[blockIdx.x] = shared_operation_vector_max[0];
	}
}

//---------------------------------------------------------------------
// Função FACADE para a redução, poís deve ser feita em passos
// double* input: o espaço a ser reduzido
// double* global: o local onde a redução será executada
// int arrSize: o tamanho do espaço onde efetuar a busca
// bool debug: flag que identifica se será impresso na tela dados de debug
//---------------------------------------------------------------------

void callReduction(int* input, int* global, int* arrSize, int maxArrSize, bool debug){
	// Quantidade de blocos e threads a serem utilizados
	int blocks = 0;
	int threads = 0;

	// Apenas calcula quantidade de blocos e threads
	calculateBlocks(maxArrSize, &blocks, &threads);

	// Calcula total de passos a mais a serem executados quando houverem mais blocos
	int total_pass_limit_block = blocks / ARS_MAX_THREADS;

	// Total de passos padrões para 1 bloco
	int total_pass = 1;

	// Somar 1 passo caso tenha mais de 1 bloco
	if (blocks > 1)
		total_pass = 2;

	// Inicializa tamanho padrão da memoria compartilha, sempre 1024 para não dar problemas...
	int sharedMemSize = sizeof(int) * ARS_MAX_THREADS;

	// Faz a redução inicial
	simple_reduction_shared << <blocks, threads, sharedMemSize >> >(input, global, arrSize);

	// Chama todos os passos restantes para terminar a redução dentro do vetor global
	for (int i = 1; i < total_pass_limit_block + total_pass; i++){

		simple_reduction_shared << <blocks, threads, sharedMemSize >> >(global, global, arrSize);
	}

	// Depuração
	if (debug){
		int maxValue;
		cudaMemcpy(&maxValue, &global[0], sizeof(int), cudaMemcpyDeviceToHost);

		int sum = (maxArrSize * (maxArrSize + 1)) / 2;

		if (maxValue == sum){
			printf("Real sum: %d \t\t calculated sum: %d TRUE\n", sum, maxValue);
		}
		else{
			printf("Real sum: %d \t\t calculated sum: %d FALSE-------------------------\n", sum, maxValue);
		}
	}
}

//---------------------------------------------------------------------
// Função FACADE para a redução de max, poís deve ser feita em passos
// double* input: o espaço a ser reduzido
// double* global: o local onde a redução será executada
// int* max: o local onde serão armazenados os indices de max
// int arrSize: o tamanho do espaço onde efetuar a busca
// bool debug: flag que identifica se será impresso na tela dados de debug
//---------------------------------------------------------------------

void callReductionMax(double* input, double* global, int* max, int* arrSize, int maxArrSize, bool debug){
	// Tamanho de blocos e threads
	int blocks = 0;
	int threads = 0;

	// Apenas calcula blocos e threads a serem utilizados
	calculateBlocks(maxArrSize, &blocks, &threads);

	// Calcula total de passos extras a serem executados para finalizar a redução
	int total_pass_limit_block = blocks / ARS_MAX_THREADS;

	// Inicia quantidade de passos padrõa
	int total_pass = 1;

	// Caso tenha mais de 1 bloco deve-se fazer em dois passos
	if (blocks > 1)
		total_pass = 2;

	// Torna par o número de threads para evitar erros de acesso a memória compartilhada
	if (threads % 2 != 0)
		threads += 1;

	// Memoria compartilhada padrão de dois tipos de variáveis
	int sharedMemSize = sizeof(double) * ARS_MAX_THREADS + sizeof(int) * ARS_MAX_THREADS;

	// Chama primeiro passo da redução
	simple_reduction_shared_max << <blocks, threads, sharedMemSize >> >(input, global, max, arrSize, true);

	// Chama restande dos passos para quantidade de blocos iniciais, calcula resto da redução em global
	for (int i = 1; i < total_pass_limit_block + total_pass; i++)
		simple_reduction_shared_max << <blocks, threads, sharedMemSize >> >(global, global, max, arrSize, false);

	// Impressão de depuração
	if (debug){
		double result;
		cudaMemcpy(&result, global, sizeof(double), cudaMemcpyDeviceToHost);
		int index = 0;
		cudaMemcpy(&index, max, sizeof(int), cudaMemcpyDeviceToHost);

		printf("Max value: %f, Max index: %d\n", result, index);
	}
}

typedef struct aux_variables_CPU{
	// Auxiliares para operações de double
	double* op_aux_double_1;
	double* op_aux_double_2;
	double* op_aux_double_3;
	double* op_aux_double_4;
	double* op_aux_double_5;
	double* op_aux_double_6;

	// Auxiliares para operações de int
	int* op_aux_int_1;
	int* op_aux_int_2;
	int* op_aux_int_3;
	int* op_aux_int_4;
	int* op_aux_int_5;
	int* op_aux_int_6;
}AUX_OP_VARIABLES_CPU;

typedef struct aux_variables_GPU{
	// Auxiliares para operações de double
	double* op_aux_double_1;
	double* op_aux_double_2;
	double* op_aux_double_3;
	double* op_aux_double_4;
	double* op_aux_double_5;
	double* op_aux_double_6;
	double* op_aux_double_7;
	double* op_aux_double_8;
	double* op_aux_double_9;

	// ART 2
	double* op_aux_double_10;
	double* op_aux_double_11;
	double* op_aux_double_12;
	double* op_aux_double_13;
	double* op_aux_double_14;
	double* op_aux_double_15;
	double* op_aux_double_16;

	// Auxiliares para operações de int
	int* op_aux_int_1;
	int* op_aux_int_2;
	int* op_aux_int_3;
	int* op_aux_int_4;
	int* op_aux_int_5;
	int* op_aux_int_6;
}AUX_OP_VARIABLES_GPU;

typedef struct fusion_art_CPU{
	// Quantidade de neuronios
	int total_neurons;

	// Entradas
	double* neurons_activities;

	// Tamanho dos campos
	int* fields;

	// Total de memoria para todos os campos juntos
	int total_fields_reserved_memmory;

	// Total de campos
	int total_fields;

	// Neuronios utilizados
	int utilized_neurons;

	double* learning_rate_decay;

	// Para aprendizado
	double* gammas;
	double* alphas;
	double* vigiliances;
	double* betas;
	int* art_to_use;
}FUSION_ART_CPU;

typedef struct fusion_art_GPU{
	// Quantidade de neuronios utilizados
	int* utilized_neurons;

	// Quantidade total de neuronios
	int* total_neurons;

	// Entradas
	double* neurons_activities;

	// Rede principal
	double* neurons_weights;

	// Tamanho dos campos
	int* fields;

	// Total de memoria para todos os campos juntos
	int* total_fields_reserved_memmory;

	// Total de campos
	int* total_fields;

	double* learning_rate_decay;

	// Para aprendizado
	double* gammas;
	double* alphas;
	double* vigiliances;
	double* betas;
	int* art_to_use;
}FUSION_ART_GPU;

// Cria rede na GPU
//createNetwork(&config, &network);
FUSION_ART_GPU net_gpu;
FUSION_ART_CPU net_cpu;
AUX_OP_VARIABLES_CPU aux_cpu;
AUX_OP_VARIABLES_GPU aux_gpu;

// Contagem de execução
bool learning;
int iteractions;
int totalLeraning;
int command;
int total_iteractions;
double sum_total_iterations;
double execution_time_sum;
double data_input_time_sum;
double data_output_time_sum;

cudaEvent_t start, stop;
float elapsed;

void init_network_param(FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_CPU* aux_cpu, AUX_OP_VARIABLES_GPU* aux_gpu){
	cudaMemcpy(net_gpu->gammas, net_cpu->gammas, sizeof(double) * net_cpu->total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->alphas, net_cpu->alphas, sizeof(double) * net_cpu->total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->vigiliances, net_cpu->vigiliances, sizeof(double) * net_cpu->total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->betas, net_cpu->betas, sizeof(double) * net_cpu->total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->learning_rate_decay, net_cpu->learning_rate_decay, sizeof(double) * net_cpu->total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->art_to_use, net_cpu->art_to_use, sizeof(int) * net_cpu->total_fields, cudaMemcpyHostToDevice);

	int total_fields = 0;
	for (int i = 0; i < net_cpu->total_fields; i++){
		if (net_cpu->gammas[i] > 0.0){
			aux_cpu->op_aux_int_3[i] = 1;
			total_fields++;
		}
	}

	cudaMemcpy(aux_gpu->op_aux_int_3, aux_cpu->op_aux_int_3, sizeof(int) * net_cpu->total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(aux_gpu->op_aux_int_4, &total_fields, sizeof(int), cudaMemcpyHostToDevice);
}

void clearNetCPU(FUSION_ART_CPU* net_cpu, AUX_OP_VARIABLES_CPU* aux_cpu){
	free(net_cpu->fields);
	free(aux_cpu->op_aux_int_1);
	free(aux_cpu->op_aux_int_3);

	free(net_cpu->gammas);
	free(net_cpu->alphas);
	free(net_cpu->vigiliances);
	free(net_cpu->betas);
	free(net_cpu->learning_rate_decay);
	free(net_cpu->art_to_use);

	net_cpu->fields = NULL;
	aux_cpu->op_aux_int_1 = NULL;
	aux_cpu->op_aux_int_3 = NULL;
	net_cpu->gammas = NULL;
	net_cpu->alphas = NULL;
	net_cpu->vigiliances = NULL;
	net_cpu->betas = NULL;
	net_cpu->learning_rate_decay = NULL;
	net_cpu->art_to_use = NULL;
}

void clearNetGPU(FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_CPU* aux_cpu, AUX_OP_VARIABLES_GPU* aux_gpu){
	free(net_cpu->neurons_activities);
	cudaFree(net_gpu->utilized_neurons);
	cudaFree(net_gpu->total_neurons);
	cudaFree(net_gpu->neurons_activities);
	cudaFree(net_gpu->neurons_weights);
	cudaFree(net_gpu->fields);
	cudaFree(net_gpu->total_fields_reserved_memmory);
	cudaFree(net_gpu->total_fields);
	cudaFree(aux_gpu->op_aux_int_1);
	cudaFree(aux_gpu->op_aux_int_2);
	cudaFree(aux_gpu->op_aux_int_3);
	cudaFree(aux_gpu->op_aux_int_4);
	cudaFree(aux_gpu->op_aux_double_1);
	cudaFree(aux_gpu->op_aux_double_2);
	cudaFree(aux_gpu->op_aux_double_3);
	cudaFree(aux_gpu->op_aux_double_4);
	cudaFree(aux_gpu->op_aux_double_5);
	cudaFree(aux_gpu->op_aux_double_6);
	cudaFree(aux_gpu->op_aux_double_7);
	cudaFree(aux_gpu->op_aux_double_8);
	cudaFree(aux_gpu->op_aux_double_9);
	cudaFree(aux_gpu->op_aux_double_10);
	cudaFree(aux_gpu->op_aux_double_11);
	cudaFree(aux_gpu->op_aux_double_12);
	cudaFree(aux_gpu->op_aux_double_13);
	cudaFree(aux_gpu->op_aux_double_14);
	cudaFree(aux_gpu->op_aux_double_15);
	cudaFree(aux_gpu->op_aux_double_16);

	cudaFree(net_gpu->gammas);
	cudaFree(net_gpu->alphas);
	cudaFree(net_gpu->vigiliances);
	cudaFree(net_gpu->betas);
	cudaFree(net_gpu->learning_rate_decay);
	cudaFree(net_gpu->art_to_use);


	net_cpu->neurons_activities = NULL;
	net_gpu->utilized_neurons = NULL;
	net_gpu->total_neurons = NULL;
	net_gpu->neurons_activities = NULL;
	net_gpu->neurons_weights = NULL;
	net_gpu->fields = NULL;
	net_gpu->total_fields_reserved_memmory = NULL;
	net_gpu->total_fields = NULL;
	aux_gpu->op_aux_int_1 = NULL;
	aux_gpu->op_aux_int_2 = NULL;
	aux_gpu->op_aux_int_3 = NULL;
	aux_gpu->op_aux_int_4 = NULL;
	aux_gpu->op_aux_double_1 = NULL;
	aux_gpu->op_aux_double_2 = NULL;
	aux_gpu->op_aux_double_3 = NULL;
	aux_gpu->op_aux_double_4 = NULL;
	aux_gpu->op_aux_double_5 = NULL;
	aux_gpu->op_aux_double_6 = NULL;
	aux_gpu->op_aux_double_7 = NULL;
	aux_gpu->op_aux_double_8 = NULL;
	aux_gpu->op_aux_double_9 = NULL;
	aux_gpu->op_aux_double_10 = NULL;
	aux_gpu->op_aux_double_11 = NULL;
	aux_gpu->op_aux_double_12 = NULL;
	aux_gpu->op_aux_double_13 = NULL;
	aux_gpu->op_aux_double_14 = NULL;
	aux_gpu->op_aux_double_15 = NULL;
	aux_gpu->op_aux_double_16 = NULL;

	net_gpu->gammas = NULL;
	net_gpu->alphas = NULL;
	net_gpu->vigiliances = NULL;
	net_gpu->betas = NULL;
	net_gpu->learning_rate_decay = NULL;
	net_gpu->art_to_use = NULL;
}

void init_network(FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_CPU* aux_cpu, AUX_OP_VARIABLES_GPU* aux_gpu, double* default_weights, int default_weights_size){
	// Antes de inicializar limpar toda a memoria
	clearNetGPU(net_cpu, net_gpu, aux_cpu, aux_gpu);

	// Inicialização cpu
	net_cpu->neurons_activities = (double*)malloc(sizeof(double) * net_cpu->total_fields_reserved_memmory);

	// Inicialização gpu
	cudaMalloc((void**)&net_gpu->utilized_neurons, sizeof(int));
	cudaMalloc((void**)&net_gpu->total_neurons, sizeof(int));
	cudaMalloc((void**)&net_gpu->neurons_activities, sizeof(double) * net_cpu->total_fields_reserved_memmory);
	cudaMalloc((void**)&net_gpu->neurons_weights, sizeof(double) * net_cpu->total_fields_reserved_memmory * net_cpu->total_neurons);
	cudaMalloc((void**)&net_gpu->fields, sizeof(int)	* net_cpu->total_fields);
	cudaMalloc((void**)&net_gpu->total_fields_reserved_memmory, sizeof(int));
	cudaMalloc((void**)&net_gpu->total_fields, sizeof(int));

	// Parametros
	cudaMalloc((void**)&net_gpu->gammas, sizeof(double) * net_cpu->total_fields);
	cudaMalloc((void**)&net_gpu->alphas, sizeof(double) * net_cpu->total_fields);
	cudaMalloc((void**)&net_gpu->vigiliances, sizeof(double) * net_cpu->total_fields);
	cudaMalloc((void**)&net_gpu->betas, sizeof(double) * net_cpu->total_fields);
	cudaMalloc((void**)&net_gpu->learning_rate_decay, sizeof(double) * net_cpu->total_fields);
	cudaMalloc((void**)&net_gpu->art_to_use, sizeof(int) * net_cpu->total_fields);

	// Inicializa neuronios com 1.0
	int tot = net_cpu->total_fields_reserved_memmory * net_cpu->total_neurons;
	int t, b;
	calculateBlocks(tot, &b, &t);
	fill << <b, t >> >(net_gpu->neurons_weights, tot, 1.0);

	// Inicialização de pesos padrão
	if (default_weights != 0x0)
		cudaMemcpy(net_gpu->neurons_weights, default_weights, sizeof(double) * default_weights_size, cudaMemcpyHostToDevice);

	// Inicialização das variaveis
	int utilized_neurons = net_cpu->utilized_neurons;
	cudaMemcpy(net_gpu->utilized_neurons, &utilized_neurons, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->total_neurons, &net_cpu->total_neurons, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->total_fields, &net_cpu->total_fields, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->total_fields_reserved_memmory, &net_cpu->total_fields_reserved_memmory, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(net_gpu->fields, net_cpu->fields, sizeof(int) * net_cpu->total_fields, cudaMemcpyHostToDevice);

	// aux_int_1 é utilizado para mapear a posição inicial de um campo dentro de neurons_weights. O mapeamento é
	// feito de 0 a n, onde n é o total de campos somados de todos os neuronios
	cudaMalloc((void**)&aux_gpu->op_aux_int_1, sizeof(int) * net_cpu->total_neurons * net_cpu->total_fields);
	cudaMemcpy(aux_gpu->op_aux_int_1, aux_cpu->op_aux_int_1, sizeof(int) * net_cpu->total_neurons * net_cpu->total_fields, cudaMemcpyHostToDevice);

	// aux_int_2 é utilizado para realização de reduções para achar indice do valor máximo
	cudaMalloc((void**)&aux_gpu->op_aux_int_2, sizeof(int) * net_cpu->total_neurons);

	// aux_int_3 é utilizado para guardar o endereço do indice maximo das reducoes
	cudaMalloc((void**)&aux_gpu->op_aux_int_3, sizeof(int) * net_cpu->total_fields);

	// aux_int_4 quantos campos estao ativos
	cudaMalloc((void**)&aux_gpu->op_aux_int_4, sizeof(int));

	// aux_double_1 é utilizado para guardar o fuzzyand
	cudaMalloc((void**)&aux_gpu->op_aux_double_1, sizeof(double) * net_cpu->total_fields_reserved_memmory * net_cpu->total_neurons);

	// aux_double_2 é utilizado para guardar a norma do fuzzyand
	cudaMalloc((void**)&aux_gpu->op_aux_double_2, sizeof(double) * net_cpu->total_fields * net_cpu->total_neurons);

	// aux_double_3 é utilizado para guardar a norma de neurons_weight e a divisao de t
	cudaMalloc((void**)&aux_gpu->op_aux_double_3, sizeof(double) * net_cpu->total_fields * net_cpu->total_neurons);

	// aux_double_4 é utilizado para guardar o t_vector completo
	cudaMalloc((void**)&aux_gpu->op_aux_double_4, sizeof(double) * net_cpu->total_neurons);

	// aux_double_5 é utilizado para guardar a norma da atividade
	cudaMalloc((void**)&aux_gpu->op_aux_double_5, sizeof(double) * net_cpu->total_fields);

	// aux_double_6 é utilizado para guardar cada valor de m_jk
	cudaMalloc((void**)&aux_gpu->op_aux_double_6, sizeof(double) * net_cpu->total_fields * net_cpu->total_neurons);

	// aux_double_7 é utilizado para guardar cada valor de resonancia
	cudaMalloc((void**)&aux_gpu->op_aux_double_7, sizeof(double) * net_cpu->total_neurons);

	// aux_double_8 é utilizado para guardar a soma de T com as resonancias
	cudaMalloc((void**)&aux_gpu->op_aux_double_8, sizeof(double) * net_cpu->total_neurons);

	// aux_double_9 é utilizado para realização de reduções para achar valor máximo
	cudaMalloc((void**)&aux_gpu->op_aux_double_9, sizeof(double) * net_cpu->total_neurons);

	// aux_double_10 é utilizado para guardar a multiplicacao por cada elemento de cada neuronio
	cudaMalloc((void**)&aux_gpu->op_aux_double_10, sizeof(double) * net_cpu->total_fields_reserved_memmory * net_cpu->total_neurons);

	// aux_double 11 é utilizado para guardar o quadrado da atividade
	cudaMalloc((void**)&aux_gpu->op_aux_double_11, sizeof(double) * net_cpu->total_fields_reserved_memmory);

	// aux_double 12 é utilizado para guardar o quadrado de cada valor de cada neuronio
	cudaMalloc((void**)&aux_gpu->op_aux_double_12, sizeof(double) * net_cpu->total_fields_reserved_memmory * net_cpu->total_neurons);

	// aux_double 13 é utilizado para guardar o dot entra atividade e pesos da rede para cada neuronio
	cudaMalloc((void**)&aux_gpu->op_aux_double_13, sizeof(double) * net_cpu->total_fields * net_cpu->total_neurons);

	// aux_double 14 é utilizado para guardar a norma do quadrado da atividade
	cudaMalloc((void**)&aux_gpu->op_aux_double_14, sizeof(double) * net_cpu->total_fields);

	// aux_double 15 é utilizado para guardar a norma do quadrado dos pesos
	cudaMalloc((void**)&aux_gpu->op_aux_double_15, sizeof(double) * net_cpu->total_fields * net_cpu->total_neurons);

	// aux_double 16 é utilizado para guardar as divisoes do cos(theta)
	cudaMalloc((void**)&aux_gpu->op_aux_double_16, sizeof(double) * net_cpu->total_fields * net_cpu->total_neurons);
}

void activityToGPU(double* activity_cpu, double* activity_gpu, int total_fields_reserved_memmory){
	cudaMemcpy(activity_gpu, activity_cpu, sizeof(double) * total_fields_reserved_memmory, cudaMemcpyHostToDevice);
}

void activityToCPU(double* activity_cpu, double* activity_gpu, int total_fields_reserved_memmory){
	cudaMemcpy(activity_cpu, activity_gpu, sizeof(double) * total_fields_reserved_memmory, cudaMemcpyDeviceToHost);
}

__global__ void debug_print_vec(double* value, int* total_fields_reserved_memmory){
	printf("Debug print:\t");
	for (int i = 0; i < *total_fields_reserved_memmory; i++){
		printf("%f ", value[i]);
	}
	printf("\n");
}

__global__ void debug_print_weights(double* value, int* available_neurons, int* total_fields_reserved_memmory){
	printf("Debug print:\t");
	for (int i = 0; i < *available_neurons * *total_fields_reserved_memmory; i++){
		if (i % *total_fields_reserved_memmory == 0)
			printf("\n");
		printf("%f ", value[i]);
	}
	printf("\n");
}

__global__ void debug_print_full_network(double* value, int* available_neurons, int* fields, int* totalFields, int* total_fields_reserved_memmory, int* neuron_mapping_index){
	printf("Network:\n");
	for (int i = 0; i < *available_neurons; i++){
		int neuronStartingIndex = i * *totalFields;

		for (int j = 0; j < *totalFields; j++){
			int index_field = neuron_mapping_index[neuronStartingIndex + j];

			printf("N. %d Field %d: ", i, j);
			for (int f = 0; f < fields[j]; f++)
				printf("%f ", value[index_field + f]);
			printf("\n");
		}
	}
	printf("\n");
}


__global__ void FUSION_fuzzy_art_and(double* weights, double* activity, double* result, int* arrSizeActivity, int* availableNeurons, int* total_fields_reserved_memmory, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int activity_index = index % *arrSizeActivity;

	if (index < *availableNeurons * *arrSizeActivity){
		if (weights[index] < activity[activity_index])
			result[index] = weights[index];
		else
			result[index] = activity[activity_index];
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy and:\t");
			for (int i = 0; i < *availableNeurons * *arrSizeActivity; i++){
				if (i % *total_fields_reserved_memmory == 0)
					printf("\n");
				printf("%f ", result[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_fuzzy_art_IImult(double* weights, double* activity, double* result, int* arrSizeActivity, int* availableNeurons, int* total_fields_reserved_memmory, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int activity_index = index % *arrSizeActivity;

	if (index < *availableNeurons * *arrSizeActivity){
		result[index] = weights[index] * activity[activity_index];
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy mult:\t");
			for (int i = 0; i < *availableNeurons * *arrSizeActivity; i++){
				if (i % *total_fields_reserved_memmory == 0)
					printf("\n");
				printf("%f ", result[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_fields_fuzzy_norm(double* input, double* output, double* alphas, int* fields, int* fields_mapping, int* totalFields, int* availableNeurons, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *availableNeurons * *totalFields){
		int field = index % *totalFields;
		int fieldStartPosition = fields_mapping[index];
		double sum = 0.0;

		for (int i = 0; i < fields[field]; i++){
			sum += input[fieldStartPosition + i];
		}

		if (alphas != 0x0)
			sum += alphas[field];

		output[index] = 1.0 - (sum / fields[field]);
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Norm:\t");
			for (int i = 0; i < *availableNeurons * *totalFields; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_activity_fuzzy_norm(double* activity, double* output, double* alphas, int* fields, int* fields_mapping, int* totalFields, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *totalFields){
		int fieldStartPosition = fields_mapping[index];
		double sum = 0.0;

		for (int i = 0; i < fields[index]; i++){
			sum += activity[fieldStartPosition + i];
		}

		if (alphas != 0x0)
			sum += alphas[index];

		output[index] = sum;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Activity norm:\t");
			for (int i = 0; i < *totalFields; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_fields_fuzzyDiv(double* numerator, double* denominator, double* output, double* gammas, int* totalFields, int* availableNeurons, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *availableNeurons * *totalFields)
	{
		double division = 0.0;

		//if (numerator[index] > 0.0 && denominator[index] > 0.0){
		int field = index % *totalFields;
		division = numerator[index] / denominator[index];
		division = division * gammas[field];
		//}

		output[index] = division;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy div:\t");
			for (int i = 0; i < *availableNeurons * *totalFields; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_calculate_T(double* ARTI, double* ARTII, int* art_to_use, int* utilizedFields, int* totalUtilizedFields, double* output, int* totalFields, int* availableNeurons, bool readout, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int neuron_index = index * *totalFields;

	if (index < *availableNeurons)
	{
		double sum = 0.0;

		for (int i = 0; i < *totalFields; i++){
			if (utilizedFields[i] > 0){
				if (art_to_use[i] == 1)
					sum += ARTI[neuron_index + i];
				if (art_to_use[i] == 2)
					sum += ARTII[neuron_index + i];
			}
		}

		// Normalizar soma
		sum /= (double)*totalUtilizedFields;

		if (!readout){
			if (index == *availableNeurons - 1)
				sum = 1.5;
		}
		// else
		// {
		//	if (index == *availableNeurons - 1)
		//		sum = 0.0;
		//}

		output[index] = sum;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy Tvec:\t");
			for (int i = 0; i < *availableNeurons; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_calculate_SUPER_RESO_AND_T(
	double* ARTI,
	double* ARTII,
	int* art_to_use,
	double* gammas,
	double* vigiliances,
	double* output,
	int* utilizedFields,
	int* totalUtilizedFields,
	int* totalFields,
	int* availableNeurons,
	bool readout,
	bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int neuron_index = index * *totalFields;

	if (index < *availableNeurons)
	{
		double res = 2.0;
		double sum = 0.0;
		double valueToCheck = 0.0;
		double vigiliance = 0.0;

		for (int i = 0; i < *totalFields; i++){
			// Calcula do T
			// Calculo da resonancia
			if (utilizedFields[i] > 0){
				if (art_to_use[i] == 1)
					sum += ARTI[neuron_index + i];
				if (art_to_use[i] == 2)
					sum += ARTII[neuron_index + i];

				valueToCheck = ARTII[neuron_index + i];
				vigiliance = vigiliances[i];

				if (valueToCheck < vigiliance)
					res = 0.0;
			}
		}

		// Normalizar soma
		sum /= (double)*totalUtilizedFields;

		//if (!readout){
		//if (index == *availableNeurons - 1)
		//	sum = 1.5;
		//}

		//double result_to_write = 0.0;

		// Soma a resonancia com o sum
		output[index] = sum + res;
	}

	if (true){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy Tvec:\t");
			for (int i = 0; i < *availableNeurons; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_calculate_resonance(double* neurons_and_activity, double* activity, double* output, int* utilizedFields, double* vigiliances, int* totalFields, int* availableNeurons, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *availableNeurons){
		double res = 2.0;

		int field_index = index * *totalFields;

		for (int i = 0; i < *totalFields; i++){
			if (utilizedFields[i] > 0){
				double div = 0.0;
				//if (neurons_and_activity[field_index + i] > 0.0 && activity[i] > 0.0)

				div = neurons_and_activity[field_index + i] / activity[i];

				if (div < vigiliances[i]){
					res = 0.0;
					i = *totalFields;
				}
			}
		}

		if (index == *availableNeurons - 1)
			res = 0.0;

		output[index] = res;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy reso:\t");
			for (int i = 0; i < *availableNeurons; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_calculate_resonanceARTII(double* ARTII, double* output, double* gammas, double* vigiliances, int* totalFields, int* availableNeurons, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *availableNeurons){
		double res = 2.0;

		int field_index = index * *totalFields;

		for (int i = 0; i < *totalFields; i++){
			//double div = 0.0;
			double valueToCheck = ARTII[field_index + i];
			double vigiliance = vigiliances[i] * gammas[i];
			if (valueToCheck < vigiliance){
				res = 0.0;
				i = *totalFields;
			}
		}

		if (index == *availableNeurons - 1)
			res = 0.0;

		output[index] = res;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy reso:\t");
			for (int i = 0; i < *availableNeurons; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_calculate_resonance_sum(double* resonance, double* t_vector, double* output, int* availableNeurons, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *availableNeurons){
		output[index] = resonance[index] + t_vector[index];
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Tvec + reso:\t");
			for (int i = 0; i < *availableNeurons; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_learn_and_readout(double* neurons_weights, double* neurons_and_activity, int* art_to_use, double* activity, double* betas, int* fields_mapping, int* fields, int* totalFields, int* total_fields_reserved_memmory, int* selectedNeuron, int* availableNeurons, bool learning, double* learning_rate_decay, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int neuron_desloc = *selectedNeuron * *totalFields;
	int field_start = fields_mapping[index + neuron_desloc];

	int cumul_activity_index = 0;

	if (index > 0)
		for (int i = 1; i <= index; i++)
			cumul_activity_index += fields[index - i];

	for (int i = field_start; i < field_start + fields[index]; i++){
		int act_index = i - field_start + cumul_activity_index;

		if (learning){
			double rating = betas[index];
			double inverseRate = 1.0 - rating;

			if (inverseRate > 1.0)
				inverseRate = 1.0;
			if (inverseRate < 0.0)
				inverseRate = 0.0;

			if (art_to_use[index] == 1)
				neurons_weights[i] = inverseRate * neurons_weights[i] + rating * neurons_and_activity[i];
			if (art_to_use[index] == 2)
				neurons_weights[i] = inverseRate * neurons_weights[i] + rating * activity[act_index];

			// Decaimento do aprendizado
			betas[index] -= learning_rate_decay[index];

			if (betas[index] < 0.005)
				betas[index] = 0.005;
		}

		//if (art_to_use[index] == 1)
		//activity[act_index] = neurons_and_activity[i];

		//if (art_to_use[index] == 2)
		activity[act_index] = neurons_weights[i];
	}

	if (learning)
		if (selectedNeuron[0] == *availableNeurons - 1 &&
			*availableNeurons < ARS_MAX_NEURONS)
			*availableNeurons += 1;

	if (debug){
		__syncthreads();

		if (index == 0){
			if (learning){
				printf("learned in %d:\t", *selectedNeuron);
				for (int i = field_start; i < field_start + *total_fields_reserved_memmory; i++){
					printf("%f ", neurons_weights[i]);
				}
				printf("\n");
			}

			printf("readout in %d:\t", *selectedNeuron);
			for (int i = 0; i < *total_fields_reserved_memmory; i++){
				printf("%f ", activity[i]);
			}
			printf("\n");
		}
	}
}

__global__
void FUSION_activityPOW(double* activity, double* result, int* vecSize){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *vecSize)
		result[index] = activity[index] * activity[index];
}

__global__
void FUSION_weightsPOW(double* weights, double* result, int* totalNeurons, int* totalFieldsReservedMemmory){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int total = *totalNeurons * *totalFieldsReservedMemmory;

	if (index < total)
		result[index] = weights[index] * weights[index];
}

__global__
void FUSION_activitySQRT(double* input, double* result, int* vecSize){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *vecSize)
		result[index] = sqrtf(input[index]);
}

__global__
void FUSION_weightsSQRT(double* input, double* result, int* totalNeurons, int* totalFields){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int total = *totalNeurons * *totalFields;

	if (index < total)
		result[index] = sqrtf(input[index]);
}

__global__ void FUSION_fields_fuzzyDivARTII(double* numerator, double* activityDenominator, double* weightsDenominator, double* output, double* gammas, int* totalFields, int* availableNeurons, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *availableNeurons * *totalFields)
	{
		double division = 0.0;

		//if (numerator[index] > 0.0 && denominator[index] > 0.0){
		int field = index % *totalFields;
		double denominator = (activityDenominator[field] * weightsDenominator[index]);

		if (denominator <= 0.0)
			division = 0.0;
		else
			division = numerator[index] / denominator;

		division = division * gammas[field];

		output[index] = division;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy div:\t");
			for (int i = 0; i < *availableNeurons * *totalFields; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

void calculateFuzzyARTI(FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_GPU* aux_gpu, bool debug){
	int blocks, threads;
	//	cudaError_t cudaerr;

	// Calcula x ^ w
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields_reserved_memmory, &blocks, &threads);
	FUSION_fuzzy_art_and << <blocks, threads >> >(net_gpu->neurons_weights, net_gpu->neurons_activities, aux_gpu->op_aux_double_1, net_gpu->total_fields_reserved_memmory, net_gpu->utilized_neurons, net_gpu->total_fields_reserved_memmory, debug);

	// Calcula |x ^ w|
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_fields_fuzzy_norm << <blocks, threads >> >(aux_gpu->op_aux_double_1, aux_gpu->op_aux_double_2, 0x0, net_gpu->fields, aux_gpu->op_aux_int_1, net_gpu->total_fields, net_gpu->utilized_neurons, debug);

	// Calcula | w | + alpha
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_fields_fuzzy_norm << <blocks, threads >> >(net_gpu->neurons_weights, aux_gpu->op_aux_double_3, net_gpu->alphas, net_gpu->fields, aux_gpu->op_aux_int_1, net_gpu->total_fields, net_gpu->utilized_neurons, debug);

	// Calcula |x ^ w| / | w | * gamma
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_fields_fuzzyDiv << <blocks, threads >> >(aux_gpu->op_aux_double_2, aux_gpu->op_aux_double_3, aux_gpu->op_aux_double_3, net_gpu->gammas, net_gpu->total_fields, net_gpu->utilized_neurons, debug);
}

void calculateFuzzyARTII(FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_GPU* aux_gpu, bool debug){
	int blocks, threads;
	//	cudaError_t cudaerr;

	// Calcula x*w
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields_reserved_memmory, &blocks, &threads);
	FUSION_fuzzy_art_IImult << <blocks, threads >> >(net_gpu->neurons_weights, net_gpu->neurons_activities, aux_gpu->op_aux_double_10, net_gpu->total_fields_reserved_memmory, net_gpu->utilized_neurons, net_gpu->total_fields_reserved_memmory, debug);

	// Calcula x^2
	calculateBlocks(net_cpu->total_fields_reserved_memmory, &blocks, &threads);
	FUSION_activityPOW << <blocks, threads >> >(net_gpu->neurons_activities, aux_gpu->op_aux_double_11, net_gpu->total_fields_reserved_memmory);

	// Calcula w^2
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields_reserved_memmory, &blocks, &threads);
	FUSION_weightsPOW << <blocks, threads >> >(net_gpu->neurons_weights, aux_gpu->op_aux_double_12, net_gpu->total_neurons, net_gpu->total_fields_reserved_memmory);

	// Calcula dot x.w
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_fields_fuzzy_norm << <blocks, threads >> >(aux_gpu->op_aux_double_10, aux_gpu->op_aux_double_13, net_gpu->alphas, net_gpu->fields, aux_gpu->op_aux_int_1, net_gpu->total_fields, net_gpu->utilized_neurons, debug);

	// Calcula |x^2|
	calculateBlocks(net_cpu->total_fields, &blocks, &threads);
	FUSION_activity_fuzzy_norm << <blocks, threads >> >(aux_gpu->op_aux_double_11, aux_gpu->op_aux_double_14, net_gpu->alphas, net_gpu->fields, aux_gpu->op_aux_int_1, net_gpu->total_fields, debug);

	// Calcula |w^2|
	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_fields_fuzzy_norm << <blocks, threads >> >(aux_gpu->op_aux_double_12, aux_gpu->op_aux_double_15, net_gpu->alphas, net_gpu->fields, aux_gpu->op_aux_int_1, net_gpu->total_fields, net_gpu->utilized_neurons, debug);

	calculateBlocks(net_cpu->total_fields, &blocks, &threads);
	FUSION_activitySQRT << < blocks, threads >> > (aux_gpu->op_aux_double_14, aux_gpu->op_aux_double_14, net_gpu->total_fields);


	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_weightsSQRT << < blocks, threads >> > (aux_gpu->op_aux_double_15, aux_gpu->op_aux_double_15, net_gpu->utilized_neurons, net_gpu->total_fields);

	calculateBlocks(net_cpu->total_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_fields_fuzzyDivARTII << < blocks, threads >> > (aux_gpu->op_aux_double_13, aux_gpu->op_aux_double_14, aux_gpu->op_aux_double_15, aux_gpu->op_aux_double_16, net_gpu->gammas, net_gpu->total_fields, net_gpu->utilized_neurons, debug);
}

void runNetwork(FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_GPU* aux_gpu, bool learning, bool debug){
	int blocks, threads;
	//	cudaError_t cudaerr;

	calculateFuzzyARTI(net_cpu, net_gpu, aux_gpu, debug);
	calculateFuzzyARTII(net_cpu, net_gpu, aux_gpu, debug);

	// Calcula sum {ART}
	calculateBlocks(net_cpu->total_neurons, &blocks, &threads);
	FUSION_calculate_T << <blocks, threads >> >(aux_gpu->op_aux_double_3, aux_gpu->op_aux_double_16, net_gpu->art_to_use, aux_gpu->op_aux_int_3, aux_gpu->op_aux_int_4, aux_gpu->op_aux_double_4, net_gpu->total_fields, net_gpu->utilized_neurons, !learning, debug);

	if (!learning){
		// Pega max t_vec
		callReductionMax(aux_gpu->op_aux_double_4, aux_gpu->op_aux_double_9, aux_gpu->op_aux_int_2, net_gpu->utilized_neurons, net_cpu->total_neurons, debug);
	}
	else{
		// Calcula | x |
		calculateBlocks(net_cpu->total_fields, &blocks, &threads);
		FUSION_activity_fuzzy_norm << <blocks, threads >> >(net_gpu->neurons_activities, aux_gpu->op_aux_double_5, 0x0, net_gpu->fields, aux_gpu->op_aux_int_1, net_gpu->total_fields, debug);

		// Calcula | x ^ w | / | x | >= vigiliance
		calculateBlocks(net_cpu->total_neurons, &blocks, &threads);
		//FUSION_calculate_resonanceARTII << <blocks, threads >> >(aux_gpu->op_aux_double_16, aux_gpu->op_aux_double_7, net_gpu->gammas, net_gpu->vigiliances, net_gpu->total_fields, net_gpu->utilized_neurons, debug);
		FUSION_calculate_resonance << <blocks, threads >> >(aux_gpu->op_aux_double_2, aux_gpu->op_aux_double_5, aux_gpu->op_aux_double_7, aux_gpu->op_aux_int_3, net_gpu->vigiliances, net_gpu->total_fields, net_gpu->utilized_neurons, debug);

		// Calcula reso + t_vec
		calculateBlocks(net_cpu->total_neurons, &blocks, &threads);
		FUSION_calculate_resonance_sum << <blocks, threads >> >(aux_gpu->op_aux_double_7, aux_gpu->op_aux_double_4, aux_gpu->op_aux_double_8, net_gpu->utilized_neurons, debug);

		// Pega max reso + t_vec
		callReductionMax(aux_gpu->op_aux_double_8, aux_gpu->op_aux_double_9, aux_gpu->op_aux_int_2, net_gpu->utilized_neurons, net_cpu->total_neurons, debug);
	}

	// Aprendizado e criação de novos neuronios
	calculateBlocks(net_cpu->total_fields, &blocks, &threads);
	FUSION_learn_and_readout << <blocks, threads >> >(net_gpu->neurons_weights, aux_gpu->op_aux_double_1, net_gpu->art_to_use, net_gpu->neurons_activities, net_gpu->betas, aux_gpu->op_aux_int_1, net_gpu->fields, net_gpu->total_fields, net_gpu->total_fields_reserved_memmory, aux_gpu->op_aux_int_2, net_gpu->utilized_neurons, learning, net_gpu->learning_rate_decay, debug);
}

__global__ void calculateMean(double* fieldNorm, double* mean, int* fields, int* totalFields, bool debug){
	int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;

	int myField = threadIndex % *totalFields;
	int fieldSize = fields[myField];

	mean[threadIndex] = fieldNorm[threadIndex] / (double)fieldSize;
}

__global__ void distMeanNeurons(double* neurons, double* neuronMeans, double* output, int* availableNeurons, int* fieldMapping, int* totalfield, int* fieldReservedMem, bool debug){
	int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
	int neuronStart = threadIndex ** fieldReservedMem;
	int myField = threadIndex % *totalfield;
	int fieldSize = totalfield[myField];

	for (int i = 0; i < fieldSize; i++){
		output[neuronStart + i] = neurons[neuronStart + i] - neuronMeans[threadIndex];
	}

	if (debug){
		__syncthreads();

		if (threadIndex == 0){
			printf("ALLY METRIC:\t");
			for (int i = 0; i < *availableNeurons ** fieldReservedMem; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_ALY_taxicab_sub(double* weights, double* activity, double* result, int* arrSizeActivity, int* availableNeurons, int* total_fields_reserved_memmory, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int activity_index = index % *arrSizeActivity;

	if (index < *availableNeurons * *arrSizeActivity){
		result[index] = abs(weights[index] - activity[activity_index]);
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Abs dist:\t");
			for (int i = 0; i < *availableNeurons * *arrSizeActivity; i++){
				if (i % *total_fields_reserved_memmory == 0)
					printf("\n");
				printf("%f ", result[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_taxicab_calculate_T(double* fields, int* utilizedFields, int* totalUtilizedFields, double* output, int* totalFields, int* availableNeurons, bool readout, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int neuron_index = index * *totalFields;

	if (index < *availableNeurons)
	{
		double sum = 0.0;

		for (int i = 0; i < *totalFields; i++){
			if (utilizedFields[i] > 0){
				sum += fields[neuron_index + i];
			}
		}

		if (index == *availableNeurons - 1 && !readout)
			sum = 1.5;
		else{
			sum /= *totalUtilizedFields;
		}

		output[index] = sum;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Fuzzy Tvec:\t");
			for (int i = 0; i < *availableNeurons; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_calculate_resonance_taxicab(double* input, double* output, int* utilizedFields, double* vigiliances, int* totalFields, int* availableNeurons, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < *availableNeurons){
		double res = 2.0;

		int field_index = index * *totalFields;

		for (int i = 0; i < *totalFields; i++){
			if (utilizedFields[i] > 0){
				if (input[field_index + i] < vigiliances[i]){
					res = 0.0;
					i = *totalFields;
				}
			}
		}

		if (index == *availableNeurons - 1)
			res = 0.0;

		output[index] = res;
	}

	if (debug){
		__syncthreads();

		if (index == 0){
			printf("Reso:\t");
			for (int i = 0; i < *availableNeurons; i++){
				printf("%f ", output[i]);
			}
			printf("\n");
		}
	}
}

__global__ void FUSION_learn_and_readout_taxicab(double* neurons_weights, double* activity, double* betas, int* fields_mapping, int* fields, int* totalFields, int* total_fields_reserved_memmory, int* selectedNeuron, int* availableNeurons, bool learning, double* learning_rate_decay, bool debug){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int neuron_desloc = *selectedNeuron * *totalFields;
	int field_start = fields_mapping[index + neuron_desloc];

	int cumul_activity_index = 0;

	if (index > 0)
		for (int i = 1; i <= index; i++)
			cumul_activity_index += fields[index - i];

	for (int i = field_start; i < field_start + fields[index]; i++){
		int act_index = i - field_start + cumul_activity_index;

		if (learning){
			double rating = betas[index];
			double inverseRate = 1.0 - rating;

			if (inverseRate > 1.0)
				inverseRate = 1.0;
			if (inverseRate < 0.0)
				inverseRate = 0.0;

			neurons_weights[i] = inverseRate * neurons_weights[i] + rating * activity[act_index];

			// Decaimento do aprendizado
			betas[index] -= learning_rate_decay[index];

			if (betas[index] < 0.005)
				betas[index] = 0.005;
		}
		//else{
		activity[act_index] = neurons_weights[i];
		//	}
	}

	if (learning)
		if (selectedNeuron[0] == *availableNeurons - 1 &&
			*availableNeurons < ARS_MAX_NEURONS)
			*availableNeurons += 1;

	if (debug){
		__syncthreads();

		if (index == 0){
			if (learning){
				printf("learned in %d:\t", *selectedNeuron);
				for (int i = field_start; i < field_start + *total_fields_reserved_memmory; i++){
					printf("%f ", neurons_weights[i]);
				}
				printf("\n");
			}

			printf("readout in %d:\t", *selectedNeuron);
			for (int i = 0; i < *total_fields_reserved_memmory; i++){
				printf("%f ", activity[i]);
			}
			printf("\n");
		}
	}
}

void runNetworkB(FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_GPU* aux_gpu, bool learning, bool debug){
	int threads, blocks;
	//cudaError_t cudaerr;

	int available_neurons = 0;
	cudaMemcpy(&available_neurons, net_gpu->utilized_neurons, sizeof(int), cudaMemcpyDeviceToHost);

	// Calcula abs (w - x)
	calculateBlocks(available_neurons * net_cpu->total_fields_reserved_memmory, &blocks, &threads);
	FUSION_ALY_taxicab_sub << <blocks, threads >> >(net_gpu->neurons_weights, net_gpu->neurons_activities, aux_gpu->op_aux_double_1, net_gpu->total_fields_reserved_memmory, net_gpu->utilized_neurons, net_gpu->total_fields_reserved_memmory, debug);

	// Calcula |abs (w - x)|
	calculateBlocks(available_neurons * net_cpu->total_fields, &blocks, &threads);
	FUSION_fields_fuzzy_norm << <blocks, threads >> >(aux_gpu->op_aux_double_1, aux_gpu->op_aux_double_2, 0x0, net_gpu->fields, aux_gpu->op_aux_int_1, net_gpu->total_fields, net_gpu->utilized_neurons, debug);

	// Sum |abs (w - x)|
	calculateBlocks(available_neurons, &blocks, &threads);
	FUSION_taxicab_calculate_T << <blocks, threads >> >(aux_gpu->op_aux_double_2, aux_gpu->op_aux_int_3, aux_gpu->op_aux_int_4, aux_gpu->op_aux_double_4, net_gpu->total_fields, net_gpu->utilized_neurons, !learning, debug);

	// Calcula vigiliance
	calculateBlocks(available_neurons, &blocks, &threads);
	FUSION_calculate_resonance_taxicab << <blocks, threads >> >(aux_gpu->op_aux_double_2, aux_gpu->op_aux_double_7, aux_gpu->op_aux_int_3, net_gpu->vigiliances, net_gpu->total_fields, net_gpu->utilized_neurons, debug);

	// Calcula reso + t_vec
	calculateBlocks(available_neurons, &blocks, &threads);
	FUSION_calculate_resonance_sum << <blocks, threads >> >(aux_gpu->op_aux_double_7, aux_gpu->op_aux_double_4, aux_gpu->op_aux_double_8, net_gpu->utilized_neurons, debug);

	// Pega max reso + t_vec
	callReductionMax(aux_gpu->op_aux_double_8, aux_gpu->op_aux_double_9, aux_gpu->op_aux_int_2, net_gpu->utilized_neurons, net_cpu->total_neurons, debug);

	calculateBlocks(net_cpu->total_fields, &blocks, &threads);
	FUSION_learn_and_readout_taxicab << <blocks, threads >> >(net_gpu->neurons_weights, net_gpu->neurons_activities, net_gpu->betas, aux_gpu->op_aux_int_1, net_gpu->fields, net_gpu->total_fields, net_gpu->total_fields_reserved_memmory, aux_gpu->op_aux_int_2, net_gpu->utilized_neurons, learning, net_gpu->learning_rate_decay, debug);
}

void writeNetwork(std::string fileName, FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_CPU* aux_cpu){
	// Stream de arquivo para saida de dados
	std::ofstream file;

	// Copia valores da GPU para a CPU para persistir a rede
	file.open(fileName.c_str(), std::ios::out);

	if (file.is_open()){
		// Quantidade de neuronios utilizados
		int utilizedNeurons;
		cudaMemcpy(&utilizedNeurons, net_gpu->utilized_neurons, sizeof(int), cudaMemcpyDeviceToHost);

		// Todos os pesos dos neuronios utilizados
		double* neurons;
		neurons = (double*)malloc(sizeof(double) * utilizedNeurons * net_cpu->total_fields_reserved_memmory);

		// Copia neuronios para cpu para persistir a rede
		cudaMemcpy(neurons, net_gpu->neurons_weights, sizeof(double) * utilizedNeurons * net_cpu->total_fields_reserved_memmory, cudaMemcpyDeviceToHost);

		std::string write_aux = "";

		// Grava total de neuronios a serem lidos
		write_aux = to_string(utilizedNeurons);
		file.write(write_aux.c_str(), write_aux.length());

		// Quebra linha
		write_aux = "\n";
		file.write(write_aux.c_str(), write_aux.length());

		// Grava tamanho de cada campo
		for (int i = 0; i < net_cpu->total_fields; i++){
			int fieldSize = net_cpu->fields[i];
			write_aux = to_string(fieldSize);
			file.write(write_aux.c_str(), write_aux.length());

			if (i == net_cpu->total_fields - 1){
				write_aux = "\n";
			}
			else{
				write_aux = " ";
			}

			file.write(write_aux.c_str(), write_aux.length());
		}

		// Gravar no arquivo
		for (int i = 0; i < utilizedNeurons; i++){
			int neuron_start = i * net_cpu->total_fields;
			for (int c_field = 0; c_field < net_cpu->total_fields; c_field++){
				int field_index = aux_cpu->op_aux_int_1[neuron_start + c_field];

				// String que contera os pesos a serem gravados
				std::string to_write = "";

				// Percorre o campo
				for (int field = 0; field < net_cpu->fields[c_field]; field++){
					double value = neurons[field_index + field];

					std::stringstream str;
					str << std::fixed << std::setprecision(15) << value;

					to_write.append(str.str());

					if (field != net_cpu->fields[c_field] - 1)
						to_write.append(" ");
					else
						to_write.append("\n");
				}

				// Grava peso
				file.write(to_write.c_str(), to_write.size());
			}
		}

		free(neurons);

		file.close();
	}
	else{
		printf("Nao foi possivel criar arquivo para salvar a rede...\n");
	}
}

void writeConfFile(std::string fileName, FUSION_ART_CPU* net_cpu){
	// Grava arquivo de configuração
	std::ofstream configFile; configFile.open(fileName.c_str(), std::ios::out);

	if (configFile.is_open()){
		// Grava betas
		// Label
		std::string write_aux = "BETAS_UTILIZADOS ";
		configFile.write(write_aux.c_str(), write_aux.length());

		for (int i = 0; i < net_cpu->total_fields; i++){
			double value = net_cpu->betas[i];
			write_aux = to_string(value);
			configFile.write(write_aux.c_str(), write_aux.length());

			if (i == net_cpu->total_fields - 1){
				write_aux = "\n";
			}
			else{
				write_aux = " ";
			}

			configFile.write(write_aux.c_str(), write_aux.length());
		}

		// Grava alphas
		// Label
		write_aux = "ALPHAS_UTILIZADOS ";
		configFile.write(write_aux.c_str(), write_aux.length());

		for (int i = 0; i < net_cpu->total_fields; i++){
			double value = net_cpu->alphas[i];
			write_aux = to_string(value);
			configFile.write(write_aux.c_str(), write_aux.length());

			if (i == net_cpu->total_fields - 1){
				write_aux = "\n";
			}
			else{
				write_aux = " ";
			}

			configFile.write(write_aux.c_str(), write_aux.length());
		}

		// Grava gammas
		// Label
		write_aux = "GAMMAS_UTILIZADOS ";
		configFile.write(write_aux.c_str(), write_aux.length());

		for (int i = 0; i < net_cpu->total_fields; i++){
			double value = net_cpu->gammas[i];
			write_aux = to_string(value);
			configFile.write(write_aux.c_str(), write_aux.length());

			if (i == net_cpu->total_fields - 1){
				write_aux = "\n";
			}
			else{
				write_aux = " ";
			}

			configFile.write(write_aux.c_str(), write_aux.length());
		}

		// Grava vigilancia
		// Label
		write_aux = "VIGILANCIA_UTILIZADOS ";
		configFile.write(write_aux.c_str(), write_aux.length());

		for (int i = 0; i < net_cpu->total_fields; i++){
			double value = net_cpu->vigiliances[i];
			write_aux = to_string(value);
			configFile.write(write_aux.c_str(), write_aux.length());

			if (i == net_cpu->total_fields - 1){
				write_aux = "\n";
			}
			else{
				write_aux = " ";
			}

			configFile.write(write_aux.c_str(), write_aux.length());
		}

		// Grava decaimento do aprendizado
		// Label
		write_aux = "DECAIMENTO_APRENDIZADO ";
		configFile.write(write_aux.c_str(), write_aux.length());

		for (int i = 0; i < net_cpu->total_fields; i++){
			double value = net_cpu->learning_rate_decay[i];
			write_aux = to_string(value);
			configFile.write(write_aux.c_str(), write_aux.length());

			if (i != net_cpu->total_fields - 1){
				write_aux = " ";
				configFile.write(write_aux.c_str(), write_aux.length());
			}
		}

		// Grava decaimento do aprendizado
		// Label
		write_aux = "ART_OP ";
		configFile.write(write_aux.c_str(), write_aux.length());

		for (int i = 0; i < net_cpu->total_fields; i++){
			int value = net_cpu->art_to_use[i];
			write_aux = to_string(value);
			configFile.write(write_aux.c_str(), write_aux.length());

			if (i != net_cpu->total_fields - 1){
				write_aux = " ";
				configFile.write(write_aux.c_str(), write_aux.length());
			}
		}
	}
}

void loadNetwork(std::string fileName, FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_CPU* aux_cpu, AUX_OP_VARIABLES_GPU* aux_gpu){
	std::ifstream file;
	file.open(fileName.c_str(), std::ios::in);

	if (file.is_open()){
		std::string readed;

		// Total de neuronios
		std::getline(file, readed);
		int utilized_neurons = atoi(readed.c_str());

		// Campos
		std::getline(file, readed);
		std::vector<std::string> fields;
		split(readed, ' ', fields);

		int totalFields = fields.size();
		int fieldsReservedMemmory = 0;

		// Configuração da rede
		net_cpu->total_neurons = ARS_MAX_NEURONS;

		// configurações
		net_cpu->total_fields = totalFields;

		clearNetCPU(net_cpu, aux_cpu);

		net_cpu->fields = (int*)malloc(sizeof(int) * net_cpu->total_fields);

		for (unsigned int i = 0; i < fields.size(); i++){
			int fieldSize = atoi(fields[i].c_str());
			fieldsReservedMemmory += fieldSize;
			net_cpu->fields[i] = fieldSize;
		}

		net_cpu->total_fields_reserved_memmory = fieldsReservedMemmory;

		// aux 1 é utilizado para mapear a posição inicial de um campo dentro de neurons_weights. O mapeamento é
		// feito de 0 a n, onde n é o total de campos considerando todos os neuronios.
		aux_cpu->op_aux_int_1 = (int*)malloc(sizeof(int) * net_cpu->total_neurons * net_cpu->total_fields);

		double* weights;
		int cpy_aux = utilized_neurons * fieldsReservedMemmory;

		// Pesos iniciais
		if (utilized_neurons != 0){
			net_cpu->utilized_neurons = utilized_neurons;
			weights = (double*)malloc(sizeof(double) * utilized_neurons * fieldsReservedMemmory);
		}
		else{
			net_cpu->utilized_neurons = 1;
			weights = 0x0;
		}

		// aloca memoria pinada
		net_cpu->gammas = (double*)malloc(sizeof(double) * net_cpu->total_fields);
		net_cpu->alphas = (double*)malloc(sizeof(double) * net_cpu->total_fields);
		net_cpu->vigiliances = (double*)malloc(sizeof(double) * net_cpu->total_fields);
		net_cpu->betas = (double*)malloc(sizeof(double) * net_cpu->total_fields);
		net_cpu->learning_rate_decay = (double*)malloc(sizeof(double) * net_cpu->total_fields);
		net_cpu->art_to_use = (int*)malloc(sizeof(int) * net_cpu->total_fields);
		aux_cpu->op_aux_int_3 = (int*)malloc(sizeof(int) * net_cpu->total_fields);

		int index = 0;
		for (int i = 0; i < utilized_neurons * totalFields; i++){
			std::getline(file, readed);
			std::vector<std::string> values; split(readed, ' ', values);

			for (unsigned int j = 0; j < values.size(); j++, index++){
				double value = std::strtod(values[j].c_str(), NULL);
				weights[index] = value;
			}
		}

		// Inicialização indices para auxiliar nas operações
		index = 0;
		for (int i = 0; i < net_cpu->total_neurons * net_cpu->total_fields; i++){
			int field = i % net_cpu->total_fields;
			int current_field_size = net_cpu->fields[field];
			aux_cpu->op_aux_int_1[i] = index;
			index += current_field_size;
		}

		// Inicialização padrão
		init_network(net_cpu, net_gpu, aux_cpu, aux_gpu, weights, cpy_aux);

		// Libera bariavel auxiliar
		free(weights);
	}
}

void loadConfigFile(std::string fileName, FUSION_ART_CPU* net_cpu, FUSION_ART_GPU* net_gpu, AUX_OP_VARIABLES_CPU* aux_cpu, AUX_OP_VARIABLES_GPU* aux_gpu){
	std::ifstream file;
	file.open(fileName.c_str(), std::ios::in);

	if (file.is_open()){
		std::string readed = "";
		std::vector<std::string> input;

		// Lendo betas
		std::getline(file, readed);
		split(readed, ' ', input);
		for (int i = 1; i <= net_cpu->total_fields; i++)
			net_cpu->betas[i - 1] = atof(input[i].c_str());

		// Lendo alphas
		std::getline(file, readed);
		split(readed, ' ', input);
		for (int i = 1; i <= net_cpu->total_fields; i++)
			net_cpu->alphas[i - 1] = atof(input[i].c_str());

		// Lendo gammas
		std::getline(file, readed);
		split(readed, ' ', input);
		for (int i = 1; i <= net_cpu->total_fields; i++)
			net_cpu->gammas[i - 1] = atof(input[i].c_str());

		// Lendo vigilancias
		std::getline(file, readed);
		split(readed, ' ', input);
		for (int i = 1; i <= net_cpu->total_fields; i++)
			net_cpu->vigiliances[i - 1] = atof(input[i].c_str());

		// Lendo decaimento aprendizado
		std::getline(file, readed);
		split(readed, ' ', input);
		for (int i = 1; i <= net_cpu->total_fields; i++)
			net_cpu->learning_rate_decay[i - 1] = atof(input[i].c_str());

		std::getline(file, readed);
		split(readed, ' ', input);
		for (int i = 1; i <= net_cpu->total_fields; i++)
			net_cpu->art_to_use[i - 1] = atoi(input[i].c_str());

		init_network_param(net_cpu, net_gpu, aux_cpu, aux_gpu);
	}
}

//---------------------------------------------------------------------
// Função de teste das rotinar de redução e max
// Deve imprimir valores consistentes em todo o teste, 
// caso contrario tem algum problema
//---------------------------------------------------------------------

void reduction_test(){
	int arrSize = 100001;

	int* A; cudaMalloc((void**)&A, sizeof(int) * arrSize);
	int* C; cudaMalloc((void**)&C, sizeof(int) * arrSize);
	int* arrSize_GPU; cudaMalloc((void**)&arrSize_GPU, sizeof(int));
	int*  max; cudaMalloc((void**)&max, sizeof(int) * arrSize);

	int* a = new int[arrSize];
	for (int i = 0; i < arrSize; i++)
		a[i] = i + 1;

	cudaMemcpy(A, a, sizeof(int)*arrSize, cudaMemcpyHostToDevice);

	for (int i = 1; i <= 50; i++){
		int size = i;
		cudaMemcpy(arrSize_GPU, &size, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(C, 0, sizeof(int) * arrSize);
		callReduction(A, C, arrSize_GPU, size, true);
	}

	//for (int i = 1; i < 100000; i++){
	//int size = 9998;
	//cudaMemcpy(arrSize_GPU, &size, sizeof(int), cudaMemcpyHostToDevice);
	//callReductionMax(A, C, max, arrSize_GPU, size, true);
	//}
}

//---------------------------------------------------------------------
// Apenas imprime a memoria livre e utilizada na GPU
//---------------------------------------------------------------------

void printFreeMem(){
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	free /= 1000000;
	total /= 1000000;
	debug("Total de memoria utilizado: %d MegaBytes\n", total - free);
	debug("Total de memoria liberado: %d MegaBytes\n", free);
	debug("Total de memoria: %d MegaBytes\n", total);
}

//---------------------------------------------------------------------
// Função de execução da rede
//---------------------------------------------------------------------

double truncF(double val, int digits)
{
	double temp = 0.0;
	temp = (int)(val * pow(10, digits));
	temp /= pow(10, digits);
	return temp;
}

double GetdoublePrecision(double value, double precision)
{
	return (floor((value * pow(10, precision) + 0.5)) / pow(10, precision));
}

double strToF(std::string value){
	double fv = 0.0;
	std::vector<std::string> spl; split(value, '.', spl);

	std::string d = "0.321";
	double ted = std::strtod(d.c_str(), NULL);

	int dec = spl[0].size();

	double div = pow(10.0, dec);

	int fac = atoi(spl[1].c_str());
	double shifted = (double)fac / div;

	fv += atoi(spl[0].c_str());
	fv += shifted;

	return fv;
}

std::string convert(double value)
{
	std::stringstream ss;
	ss << std::setprecision(3);
	ss << value;
	return ss.str();
}

void call(double* stimulus_arr, int stimulus_size, double** output_arr, int* arr_size)
{
	if (stimulus_size != net_cpu.total_fields_reserved_memmory){
		debug("Erro...\n");
		static double* defaultOut = NULL;

		if (defaultOut == NULL){
			defaultOut = new double[net_cpu.total_fields_reserved_memmory];
			memset(defaultOut, 0, sizeof(double) * net_cpu.total_fields_reserved_memmory);
		}

		*output_arr = net_cpu.neurons_activities;
		*arr_size = net_cpu.total_fields_reserved_memmory;
		return;
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	activityToGPU(stimulus_arr, net_gpu.neurons_activities, net_cpu.total_fields_reserved_memmory);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed, start, stop);
	data_input_time_sum += elapsed;

	// Medir tempo de execução
	cudaEventRecord(start);
	runNetwork(&net_cpu, &net_gpu, &aux_gpu, learning, false);
	//runNetworkB(&net_cpu, &net_gpu, &aux_gpu, learning, false);
	//cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed, start, stop);
	execution_time_sum += elapsed;


	cudaEventRecord(start);
	activityToCPU(net_cpu.neurons_activities, net_gpu.neurons_activities, net_cpu.total_fields_reserved_memmory);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&elapsed, start, stop);
	data_output_time_sum += elapsed;

	*output_arr = net_cpu.neurons_activities;
	*arr_size = net_cpu.total_fields_reserved_memmory;

	// Total de iterações do algoritmo
	iteractions++;
	total_iteractions++;

	if (learning)
		totalLeraning++;

	// Imprimir apenas após mil execuções
	if (iteractions % 10000 == 0){
		//debug = true;
		debug("\nMedia do tempo de execucao copia para input: %fs\n", data_input_time_sum / (double)iteractions / 1000.0);
		debug("Media do tempo de execucao copia para output: %fs\n", data_output_time_sum / (double)iteractions / 1000.0);
		debug("Media do tempo de execucao das rotinas da rede: %fs\n", execution_time_sum / (double)iteractions / 1000.0);
		printFreeMem();

		int available_neurons = 0;
		cudaMemcpy(&available_neurons, net_gpu.utilized_neurons, sizeof(int), cudaMemcpyDeviceToHost);

		debug("Total de neuronios utilizados ate o momento: %d\n", available_neurons);
		debug("Total de iteracoes ate o momento: %d\n", total_iteractions);
		debug("Total de iteracoes de aprendizado ate o momento: %d\n", totalLeraning);
		debug("----------------------------------------------------------\n");

		// Reseta variaveis para nao poluir avaliação
		iteractions = 0;
		execution_time_sum = 0.0;
		data_output_time_sum = 0.0;
		data_input_time_sum = 0.0;
	}
}

int getNeuronCount(){
	int available_neurons = 0;
	cudaMemcpy(&available_neurons, net_gpu.utilized_neurons, sizeof(int), cudaMemcpyDeviceToHost);
	return available_neurons;
}

void setTraining(bool boolean){
	learning = boolean;
}

void loadConfig(const char* filename){
	learning = true;
	iteractions = 0;
	totalLeraning = 0;
	total_iteractions = 0;
	sum_total_iterations = 0.0;
	execution_time_sum = 0.0;
	elapsed = 0.0f;

	// Carrega rede
	loadConfigFile(filename, &net_cpu, &net_gpu, &aux_cpu, &aux_gpu);
}

void _loadNetwork(const char* fileName){
	// Carrega rede
	loadNetwork(fileName, &net_cpu, &net_gpu, &aux_cpu, &aux_gpu);
}

void _saveNetwork(const char* fileName){
	// Grava rede
	writeNetwork(fileName, &net_cpu, &net_gpu, &aux_cpu);
}