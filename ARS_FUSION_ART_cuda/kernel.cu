/*MIT License
*
*Copyright (c) 2018 Alysson Ribeiro da Silva
*
*Permission is hereby granted, free of charge, to any person obtaining a copy 
*of this software and associated documentation files (the "Software"), to deal 
*in the Software without restriction, including *without limitation the rights 
*to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
*copies of the Software, and to permit persons to whom the Software is furnished 
*to do so, subject *to the following conditions:
*
*The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*
*THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
*EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
*FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. *IN NO EVENT SHALL THE AUTHORS 
*OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN 
*AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH 
*THE *SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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
#include <Windows.h>	// Api do windows para utilizar chamadas de sistema
#include <climits>		// Definição dos limites máximos das variáveis
#include <string>		// Estrutura de dados para manipulação de strings
#include <sstream>		// Estrutura de dados para executar algoritmos em strings
#include <algorithm>	// Algoritmos para serem executados em estruturas de dados
#include <iterator>		// Iterators para manipulação de estruturas de dados
#include <cstring>		// String do C
#include <time.h>
#include <omp.h>		// Open MP

//---------------------------------------------------------------------
// Namespaces utilizados
//---------------------------------------------------------------------

using namespace std;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

#define ARS_MAX_THREADS 128
#define ARS_MAX_BLOCKS  4096 * 4096
#define ARS_MIN_THREADS 1
#define ARS_MIN_BLOCKS  1
#define ARS_MAX_NEURONS 15000000

//---------------------------------------------------------------------
// defines
//---------------------------------------------------------------------

//#define __VERBOSE__
//#define __T_VECTOR_DEBUG__
//#define __RESONANCE_VECTOR_DEBUG__
#define __PER_FIELD_SUM__
//#define __FIELD_REDUCTION__
//#define __DEBUG__

//---------------------------------------------------------------------
// Estrutura para configuração da FUSION ART
//---------------------------------------------------------------------

typedef struct networkConfig{
	int* fields_sizes;
	float* fields_gammas;
	int total_fields;

}NET_CONFIG;

//---------------------------------------------------------------------
// Estrutura para armazenamento da FUSION ART
//---------------------------------------------------------------------

typedef struct myNetwork{
	// Variaveis fixas ------------------------------------------------
	float* neurons;

	// Variaveis de operação ------------------------------------------
	float* activity;
	float* fuzzy_and;
	float* neurons_norm_field;
	float* fuzzy_and_norm_field;
	float* t_vector_division;
	float* resonance_division;
	float* sum_reduction_aux;
	float* network_reduction_aux_T_vector;
	int*   network_reduction_aux_T_vector_index;
	int*   network_field_thread_index_aux;

	// Variaveis para guardar informações da rede e constantes
	int*	field_sizes;
	float* fields_gammas;

	// Resultados finais ----------------------------------------------
	float* t_vector;
	bool*  resonance;
	float* ressonating_neurons;	
	float* t_vecPlusResonating_neurons;
	int*   last_selected_max_index;

	// Variaveis para acesso na CPU e comunicação com GPU
	float* neurons_CPU;
	float* activity_CPU;
	float* readout_CPU;
	float* field_gammas_CPU;
	int*   field_sizes_CPU;
	int*   network_field_thread_index_aux_CPU;

	// Variaveis para manipulação diretamente na CPU
	std::vector<float*> fields_input;

	int available_neurons;
	int all_fields_reserved_space;
	int total_fields;
	int neuron_count;
}FUSION_ART;

//---------------------------------------------------------------------
// Inicialização da rede e todas as suas variáveis
// NET_CONFIG* config: estrutura de configuração da rede
// FUSION_ART* my_network: rede a ser instanciada
//---------------------------------------------------------------------

void createNetwork(NET_CONFIG* config, FUSION_ART* my_network){
	// Configura o tamanho de cada field da rede
	my_network->all_fields_reserved_space = 0;	// tamanho total ocupado por todos os campos juntos

	// Calcula o total reservado para todos os fields e prepara locais de input pela CPU
	for (int i = 0; i < config->total_fields; i++){
		// Configura variaveis de controle para saber tamanho total dos campos na GPU
		my_network->all_fields_reserved_space += config->fields_sizes[i];

		// Configura as variaveis reservadas para input pela CPU
		float* new_field; cudaMallocHost((void**)&new_field, sizeof(float) * config->fields_sizes[i]); // new float[config->fields_sizes[i]];
		//memset(new_field, 0.0, sizeof(float) * config->fields_sizes[i]);
		my_network->fields_input.push_back(new_field);
	}

	// Inicializa variaveis básicas
	my_network->field_sizes_CPU = config->fields_sizes;
	my_network->field_gammas_CPU = config->fields_gammas;
	my_network->total_fields = config->total_fields;				// quantidade de campos
	my_network->neuron_count = ARS_MAX_NEURONS / my_network->all_fields_reserved_space;

	// Variaveis para facilitar manipulação
	int all_fields_reserved_space = my_network->all_fields_reserved_space;
	int total_fields = my_network->total_fields;
	int neuron_count = my_network->neuron_count;

	// Variaveis fixas -----------------------------------------------------------------------------------------------------------------
	cudaMalloc((void**)&my_network->neurons, sizeof(float) * neuron_count * all_fields_reserved_space);

	// Variaveis de operação -----------------------------------------------------------------------------------------------------------
	cudaMalloc((void**)&my_network->activity, sizeof(float) * neuron_count * all_fields_reserved_space);
	cudaMalloc((void**)&my_network->fuzzy_and, sizeof(float) * neuron_count * all_fields_reserved_space);
	cudaMalloc((void**)&my_network->neurons_norm_field, sizeof(float) * neuron_count * total_fields);
	cudaMalloc((void**)&my_network->fuzzy_and_norm_field, sizeof(float) * neuron_count * total_fields);
	cudaMalloc((void**)&my_network->t_vector_division, sizeof(float) * neuron_count * total_fields);
	cudaMalloc((void**)&my_network->resonance_division, sizeof(float) * neuron_count * total_fields);
	cudaMalloc((void**)&my_network->network_field_thread_index_aux, sizeof(int) * neuron_count * total_fields);
	cudaMalloc((void**)&my_network->sum_reduction_aux, sizeof(float) * all_fields_reserved_space);

	// Resultados finais -----------------------------------------------------------------------------------
	cudaMalloc((void**)&my_network->t_vector, sizeof(float) * neuron_count);
	cudaMalloc((void**)&my_network->network_reduction_aux_T_vector, sizeof(float) * neuron_count);
	cudaMalloc((void**)&my_network->network_reduction_aux_T_vector_index, sizeof(int) * neuron_count);
	cudaMalloc((void**)&my_network->resonance, sizeof(bool));
	cudaMalloc((void**)&my_network->ressonating_neurons, sizeof(float) * neuron_count);
	cudaMalloc((void**)&my_network->t_vecPlusResonating_neurons, sizeof(float) * neuron_count);
	cudaMalloc((void**)&my_network->field_sizes, sizeof(int) * total_fields);
	cudaMalloc((void**)&my_network->fields_gammas, sizeof(float) * total_fields);
	cudaMalloc((void**)&my_network->last_selected_max_index, sizeof(int));

	// Variaveis para acesso na CPU
	cudaMallocHost((void**)&my_network->neurons_CPU, sizeof(float) * neuron_count * all_fields_reserved_space);
	cudaMallocHost((void**)&my_network->activity_CPU, sizeof(float) * neuron_count * all_fields_reserved_space);
	cudaMallocHost((void**)&my_network->network_field_thread_index_aux_CPU, sizeof(int) * neuron_count * total_fields);
	cudaMallocHost((void**)&my_network->readout_CPU, sizeof(float) * all_fields_reserved_space);

	// inicialização dos neuronios
	for (int i = 0; i < neuron_count * all_fields_reserved_space; i++){
		// inicialização dos pesos padrões
		my_network->neurons_CPU[i] = 1.0;

		// inicialização da atividade, input para testes
		my_network->activity_CPU[i] = 0.0;
	}

	// Incialização do vetor de indice inicial dos campos para cada neuronio
	int startPosition = 0;
	for (int neuron = 0; neuron < neuron_count; neuron++){
		int neuron_index = neuron * total_fields;
		for (int field = 0; field < total_fields; field++){
			my_network->network_field_thread_index_aux_CPU[neuron_index + field] = startPosition;
			startPosition += my_network->field_sizes_CPU[field];
		}
	}

	// copia os dados da rede para de GPU
	cudaMemcpy(my_network->neurons, my_network->neurons_CPU, sizeof(float) * neuron_count * all_fields_reserved_space, cudaMemcpyHostToDevice);
	cudaMemcpy(my_network->activity, my_network->activity_CPU, sizeof(float) * neuron_count * all_fields_reserved_space, cudaMemcpyHostToDevice);
	cudaMemcpy(my_network->field_sizes, my_network->field_sizes_CPU, sizeof(int) * total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(my_network->fields_gammas, my_network->field_gammas_CPU, sizeof(float) * total_fields, cudaMemcpyHostToDevice);
	cudaMemcpy(my_network->network_reduction_aux_T_vector, my_network->activity_CPU, sizeof(float) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(my_network->network_reduction_aux_T_vector_index, my_network->activity_CPU, sizeof(float) * neuron_count, cudaMemcpyHostToDevice);
	cudaMemcpy(my_network->network_field_thread_index_aux, my_network->network_field_thread_index_aux_CPU, sizeof(int) * neuron_count * total_fields, cudaMemcpyHostToDevice);

	// Inicializa quantidade de neurônios utilizáveis
	my_network->available_neurons = 1;
}

//---------------------------------------------------------------------
// Utilidades para o host
//---------------------------------------------------------------------

void calculateBlocks(int* array_size, int* threadsInBlock, int* blocksInGrid){
	*threadsInBlock = ARS_MAX_THREADS;

	if (*array_size < *threadsInBlock)
	{
		*threadsInBlock = *array_size;
		*blocksInGrid = ARS_MIN_BLOCKS;
	}
	else{
		if (*array_size == 0){
			*threadsInBlock = ARS_MIN_THREADS;
			*blocksInGrid = ARS_MIN_BLOCKS;
		}
		else{
			*blocksInGrid = *array_size / *threadsInBlock;

			if ((*blocksInGrid) * (*threadsInBlock) < (*array_size))
				*blocksInGrid += 1;
		}
	}
}

//---------------------------------------------------------------------
// Redução simples utilizando bloco compartilhado
// float* input: entrada a ser reduzida
// float* globalBlockData: local onde a redução será executada
// int arrSize: tamanho total do vetor a ser reduzido
// NOTA: calcular blocos e threads de acordo com o tamanho de arrSize
//---------------------------------------------------------------------

__global__ void simple_reduction_shared(float* input, float* globalBlockData, int arrSize){
	extern __shared__ float shared_operation_vector[];

	// Inicialização
	unsigned int threadId = threadIdx.x;
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;

	// Evitar utilizar memoria compartilhada fora dos limites do arrSiz
	if (index < arrSize)
		shared_operation_vector[threadId] = input[index];
	else
		shared_operation_vector[threadId] = 0.0;

	// Calcula limite para saber quantas vezes deverá gerar um stride
	int limit = arrSize;
	if (limit > blockDim.x)
		limit = blockDim.x;

	__syncthreads();

	// Reducao simples
	for (unsigned int stride = 1; stride < limit; stride *= 2) {
		int index_reduc = 2 * stride * threadId;

		if (index_reduc < limit && (index_reduc + stride) < arrSize)
		{
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
// float* input: entrada a ser reduzida
// float* globalBlockData: local onde a redução será executada
// int* globalBlockDataMax: local onde armazenar os indices dos máximos
// int arrSize: tamanho do espaço onde procurar pelo max
// bool: first: identifica se é a primeira execução para inicialização
// NOTA: calcular blocos e threads de acordo com o tamanho de arrSize
//---------------------------------------------------------------------

__global__ void simple_reduction_shared_max(float* input, float* globalBlockData, int* globalBlockDataMax, int* last_selected_max, int arrSize, bool first){
	extern __shared__ int shared_op_vec[];

	// Configurado variaveis
	float* shared_operation_vector = (float*)&shared_op_vec[blockDim.x];
	int   * shared_operation_vector_max = shared_op_vec;

	// Inicialização
	int threadId = threadIdx.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	// Faz copia para memoria de trabalho apenas quando for necessario, nesse caso ate o tamanho do vetor de busca
	if (index < arrSize){
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
		*last_selected_max = shared_operation_vector_max[0];
	}
}

//---------------------------------------------------------------------
// Função FACADE para a redução, poís deve ser feita em passos
// float* input: o espaço a ser reduzido
// float* global: o local onde a redução será executada
// int arrSize: o tamanho do espaço onde efetuar a busca
// bool debug: flag que identifica se será impresso na tela dados de debug
//---------------------------------------------------------------------

void callReduction(float* input, float* global, int arrSize, bool debug){
	// Quantidade de blocos e threads a serem utilizados
	int blocks = 0;
	int threads = 0;

	// Apenas calcula quantidade de blocos e threads
	calculateBlocks(&arrSize, &threads, &blocks);

	// Calcula total de passos a mais a serem executados quando houverem mais blocos
	int total_pass_limit_block = blocks / 1024;

	// Total de passos padrões para 1 bloco
	int total_pass = 1;

	// Somar 1 passo caso tenha mais de 1 bloco
	if (blocks > 1)
		total_pass = 2;

	// Inicializa tamanho padrão da memoria compartilha, sempre 1024 para não dar problemas...
	int sharedMemSize = sizeof(float) * 1024;

	// Faz a redução inicial
	simple_reduction_shared << <blocks, threads, sharedMemSize >> >(input, global, arrSize);

	// Chama todos os passos restantes para terminar a redução dentro do vetor global
	for (int i = 1; i < total_pass_limit_block + total_pass; i++)
		simple_reduction_shared << <blocks, threads, sharedMemSize >> >(global, global, arrSize);

	// Depuração
	if (debug){
		float maxValue;
		cudaMemcpy(&maxValue, global, sizeof(float), cudaMemcpyDeviceToHost);

		printf("Sum: %f\n", maxValue);
	}
}

//---------------------------------------------------------------------
// Função FACADE para a redução de max, poís deve ser feita em passos
// float* input: o espaço a ser reduzido
// float* global: o local onde a redução será executada
// int* max: o local onde serão armazenados os indices de max
// int arrSize: o tamanho do espaço onde efetuar a busca
// bool debug: flag que identifica se será impresso na tela dados de debug
//---------------------------------------------------------------------

void callReductionMax(float* input, float* global, int* max, int* reduced_index_mem, int arrSize, bool debug){
	// Tamanho de blocos e threads
	int blocks = 0;
	int threads = 0;

	// Apenas calcula blocos e threads a serem utilizados
	calculateBlocks(&arrSize, &threads, &blocks);

	// Calcula total de passos extras a serem executados para finalizar a redução
	int total_pass_limit_block = blocks / 1024;

	// Inicia quantidade de passos padrõa
	int total_pass = 1;

	// Caso tenha mais de 1 bloco deve-se fazer em dois passos
	if (blocks > 1)
		total_pass = 2;

	// Torna par o número de threads para evitar erros de acesso a memória compartilhada
	if (threads % 2 != 0)
		threads += 1;

	// Memoria compartilhada padrão de dois tipos de variáveis
	int sharedMemSize = sizeof(float) * 1024 + sizeof(int) * 1024;

	// Chama primeiro passo da redução
	simple_reduction_shared_max << <blocks, threads, sharedMemSize >> >(input, global, max, reduced_index_mem, arrSize, true);

	// Chama restande dos passos para quantidade de blocos iniciais, calcula resto da redução em global
	for (int i = 1; i < total_pass_limit_block + total_pass; i++)
		simple_reduction_shared_max << <blocks, threads, sharedMemSize >> >(global, global, max, reduced_index_mem, arrSize, false);

	// Impressão de depuração
	if (debug){
		float result;
		cudaMemcpy(&result, global, sizeof(float), cudaMemcpyDeviceToHost);
		int index = 0;
		cudaMemcpy(&index, max, sizeof(int), cudaMemcpyDeviceToHost);

		printf("Max value: %f, Max index: %d\n", result, index);
	}

#if defined __DEBUG__
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != CUDA_SUCCESS)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
#endif
}

//---------------------------------------------------------------------
// Verificação de resonancia - apenas verifica a resonancia de um neuronio
// float* t_vector: vetor contendo os valores T de cada neuronio
// float* resonated_neurons: vetor contendo a resonancia de cada neuronio
// bool* resonated: flag de controle para determinar se resonou ou nao
// int* chekingIndex: o índice do neuronio a ser verificado
// int arrSize: o tamanho do espaço a ser feita a verificação
// NOTA: todas as variáveis devem estar na memória da GPU para melhor desempenho
//---------------------------------------------------------------------

__global__ void
check_resonance(float* t_vector, float* resonated_neurons, bool* resonated, int* checkingIndex, int arrSize){
	// Verifica resonancia no indice que estou, caso o indice seja o tamanho do vetor então é um neuronio nao comitado
	if (resonated_neurons[checkingIndex[0]] == 1.0 || checkingIndex[0] == (arrSize - 1)){
		(*resonated) = true;
	}
	else{
		t_vector[checkingIndex[0]] = -1.0;
	}
}

//---------------------------------------------------------------------
// Reseta a resonancia na GPU
// bool* resonated: variável que identifica a resonancia
// NOTA: todas as variáveis devem estar na memória da GPU
//---------------------------------------------------------------------

__global__ void
reset_resonance(bool* resonated){
	(*resonated) = false;
}

//---------------------------------------------------------------------
// Calcula a soma dos campos, onde cada thread processa 1 neuronio
// float* neurons: vetor de peso dos neuronios
// float* field_pre_sum: vetor onde será armazenada a soma final
// int* field_sizes: tamanho de cada campo da rede
// int total_neurons: total de neuronios a serem calculados
// int total_fields: total de campos
// int all_fields_reserved_space: espaço reservado acumulado de todos os campos
//---------------------------------------------------------------------

__global__
void block_Calculate_fields_sum(float* neurons, float* field_pre_sum, int* field_sizes, int total_neurons, int total_fields, int all_fields_reserved_space){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int field_index = index * all_fields_reserved_space;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	// Cada thread irá somar seus campos de forma independente
	int j = field_index;
	for (int i = 0; i < total_fields; i++){
		int		current_field_size = field_sizes[i];
		float	field_sum = 0.0;
		int		condition = j + current_field_size;
	
		// Faz a soma por campo
		for (; j < condition; j++){
			field_sum += neurons[j];
		}

		// Indexa de acordo com a pos~ição do campo
		int field_pre_sum_index = index * total_fields + i;
		field_pre_sum[field_pre_sum_index] = field_sum;
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO

	// Impressão de debug
#if defined __VERBOSE__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("fields_sum: ");
		for (int i = 0; i < total_neurons * total_fields; i++){
			printf("%f ", field_pre_sum[i]);
		}
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// Calcula a soma dos campos, onde cada thread processa 1 neuronio
// float* neurons: vetor de peso dos neuronios
// float* field_pre_sum: vetor onde será armazenada a soma final
// int* field_sizes: tamanho de cada campo da rede
// int total_neurons: total de neuronios a serem calculados
// int total_fields: total de campos
// int all_fields_reserved_space: espaço reservado acumulado de todos os campos
//---------------------------------------------------------------------

__global__
void block_Calculate_fields_sumC(float* neurons, float* field_pre_sum, int* field_sizes, int* field_thread_index_aux, int total_fields, int arrSize){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	// Cada thread irá somar seus campos de forma independente
	if (index < arrSize){
		//int field_index = index % total_fields;
		int field_index = (index - (index / total_fields) * total_fields);
		float sum = 0.0;
		int starting_index = field_thread_index_aux[index];

		for (int i = 0; i < field_sizes[field_index]; i++){
			sum += neurons[starting_index + i];
		}

		field_pre_sum[index] = sum;
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO

	// Impressão de debug
#if defined __VERBOSE__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("fields_sum: ");
		for (int i = 0; i < total_neurons * total_fields; i++){
			printf("%f ", field_pre_sum[i]);
		}
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// void att utilizado para copiar um valor para outra variavel
// float* input: vetor de entrada
// float* value: vetor com valor a ser copiado
// int position: posição onde guardar o valor no input
// NOTA: o valor copiado é sempre a primeira posição do vetor value
//---------------------------------------------------------------------

__global__ void att(float* input, float* value, int position){
	input[position] = value[0];
}

//---------------------------------------------------------------------
// Calcula a soma dos campos com reduções, porém faz o loop dos neuronios na CPU
// FUSION_ART* network: rede onde estão todas as variáveis
// float* input: entrada, todos os neuronios e campos
// float* aux_vector: vetor auxiliar para efetuar a redução
// float* result: vetor onde as reduções devem ser guardadas
//---------------------------------------------------------------------

void block_Calculate_fields_sumB(FUSION_ART* network, float* input, float* aux_vector, float* result){
	// Para cada neuronio
	for (int neuron = 0; neuron < network->available_neurons; neuron++)
	{
		// Para cada campo
		for (int field = 0; field < network->total_fields; field++){
			// Acha deslocamento do campo a ser reduzido
			int desloc = neuron * network->all_fields_reserved_space;
			if (field > 0)
				desloc += network->field_sizes_CPU[field - 1];

			// Efetua redução do campo
			callReduction(input + desloc, aux_vector, network->field_sizes_CPU[field], false);

			// Copia resultado para o vetor result na devida posição
			int index = neuron * network->total_fields + field;
			att << <1, 1 >> >(result, aux_vector, index);
		}
	}
}

//---------------------------------------------------------------------
// Calcula o vetor T final de todos os neuronios
// float* division: vetor contendo a divisão da equação de T
// float* T_vector: vetor de saida com os valores T
// int total_neurons: total de neuronios a serem calculados
// int total_fields: total de campos por neuronio
//---------------------------------------------------------------------

__global__
void block_Calculate_final_T_vector(float* division, float* T_vector, float* fields_gamma, int total_neurons, int total_fields){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES

	// shared int;
	// extern shared int;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int field_index = index * total_fields;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO


	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	if (index < total_neurons){
		float sum = 0.0;
		for (int i = 0; i < total_fields; i++){
			int field_to_sum_index = field_index + i;
			sum += division[field_to_sum_index] * fields_gamma[i];
		}

		sum /= (float)total_fields;

		T_vector[index] = sum;

		if (index == total_neurons - 1)
			T_vector[index] = 1.5; // coloca 2 para facilitar as contas no final
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO
#if defined __T_VECTOR_DEBUG__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("T_vector_final: ");
		for (int i = 0; i < total_neurons; i++){
			printf("%f ", T_vector[i]);
		}
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// Calcula o fuzzy and de dois vetores
// float* neurons: valores de peso de todos os neuronios
// float* input: vetor com valores a serem comparados com float* neurons
// float* output: vetor de saida com valores de pesos
// int total_size: tamanho total a ser calculado
//---------------------------------------------------------------------

__global__
void block_calculate_fuzzy_and(float* neurons, float* input, float* output, int total_size){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES
	// Variaveis locais
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO


	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	if (index < total_size){
		if (input[index] < neurons[index])
			output[index] = input[index];
		else
			output[index] = neurons[index];
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO

#if defined __VERBOSE__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("Fuzzy and: ");
		for (int i = 0; i < total_size; i++)
			printf("%f ", output[i]);
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// Calcula a soma simples, 1 a 1
// float* input: vetor de entrada dos neuronios
// float* increment: variavel especificando quanto deve ser somado
// int total_size: tamanho total a ser somado
//---------------------------------------------------------------------

__global__
void block_simple_add(float* input, float increment, int total_size){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES
	// Variaveis locais
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	if (index < total_size){
		input[index] = input[index] + increment;
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO

#if defined __VERBOSE__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("add: ");
		for (int i = 0; i < total_size; i++)
			printf("%f ", input[i]);
		printf("\n");
	}
#endif
}

__global__
void block_simple_vecadd(float* A, float* B, float* result, int total_size){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES
	// Variaveis locais
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	if (index < total_size){
		result[index] = A[index] + B[index];
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO

#if defined __VERBOSE__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("add: ");
		for (int i = 0; i < total_size; i++)
			printf("%f ", input[i]);
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// Calcula a multiplicação simples
// float* input: vetor de entrada
// float value: valor que cada elemento de input será multiplicado
// int total_size: tamanho total a ser calculado
//---------------------------------------------------------------------

__global__
void block_simple_mul(float* input, float value, int total_size){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES
	// Variaveis locais
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	if (index < total_size){
		input[index] = input[index] * value;
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO

#if defined __VERBOSE__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("mul: ");
		for (int i = 0; i < total_size; i++)
			printf("%f ", input[i]);
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// Efetua a divisão do indice de dois vetores, variaveis com nome especial por conveniencia da aplicação
// float* fuzzy_and: vetor de entrada
// float* norm: vetor com denominadores
// float* division: vetor de saida com as divisões
// int total_size: total de elementos a serem calculados
//---------------------------------------------------------------------

__global__
void block_fuzzy_art_division(float* fuzzy_and, float* norm, float* division, int total_size){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES
	// Variaveis locais
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO

	// Sincronizar aqui para garantir que todas as threads terão os dados inicializados
	//__syncthreads();

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	if (index < total_size){
		if (norm[index] <= 0.0)
			norm[index] = 0.0001;

		division[index] = fuzzy_and[index] / norm[index];
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO


#if defined __VERBOSE__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (threadIdx.x == 0 && blockIdx.x == 0){
		printf("division fuzzy art: ");
		for (int i = 0; i < total_size; i++)
			printf("%f ", division[i]);
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// Calcula vetor de resonancia dado o M de cada campo de cada neuronio
// float* division: vetor de entrada com os m_js
// float* resonated_vetor: vetor de saida com os neuronios que resonaram ou não
// float precision: precisão de resonancia
// int total_neurons: total de neuronios a serem verificados
// int total_fields: total de campos por neuronio
//---------------------------------------------------------------------

__global__
void block_Calculate_resonated_vector(float* division, float* resonated_vector, float precision, int total_neurons, int total_fields){
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							DECLARAÇÕES

	// shared int;
	// extern shared int;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int field_index = index * total_fields;

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//							INICIALIZAÇÃO

	resonated_vector[index] = 0.0;

	__syncthreads();

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//						OPERAÇÕES PARALELAS

	if (index < total_neurons){

		int resonated = 2.0;
		for (int i = 0; i < total_fields; i++){
			int field = field_index + i;
			if (division[field] < precision){
				resonated = 0.0;
				i = total_fields;
			}
		}

		resonated_vector[index] = resonated;

		// Garantir que o ultimo nunca resonara para poder escolher baseando-se em T
		if (index == total_neurons - 1)
			resonated_vector[index] = 0.0; // coloca 2 para facilitar as contas no final
	}

	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//					FIM PROCESSAMENTO PARALELO
#if defined __RESONANCE_VECTOR_DEBUG__
	// Garantir que todos as threads tenham executado suas operações
	__syncthreads();

	// Imprimir resultados apenas uma vez utilizando a thread 0 do processador 0
	if (index == 0){
		printf("Resonated vector: ");
		for (int i = 0; i < total_neurons; i++){
			printf("%f ", resonated_vector[i]);
		}
		printf("\n");
	}
#endif
}

//---------------------------------------------------------------------
// Efetua o aprendizado para os neuronios dado uma entrada
// float* neurons: vetor de pesos com todos os neuronios
// float* fuzzy_and: vetor de entrada para aprendizado
// float beta: taxa de aprendizado
// int learning_neuron: indice do neuronio a utilizar para aprender
// int field_reserved_space: espaço reservado de todos os indices para indexar corretamente
//---------------------------------------------------------------------

__global__ void learn(float* neurons, float* fuzzy_and, float beta, int* learning_neuron, int field_reserved_space){
	// Definição das variáveis --------------------------------------------------
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int start_position = *learning_neuron * field_reserved_space;
	int learning_index = start_position + index;

	// Aprender apenas se estiver dentro do intervalo de computação do aprendizado
	neurons[learning_index] = (1.0 - beta) * neurons[learning_index] + beta * fuzzy_and[learning_index];

#if defined __VERBOSE__
	// Imprimir dados -----------------------------------------------------------
	__syncthreads();

	// Imprimir apenas se estiver na thread 0 e bloco 0 -------------------------
	if (threadIdx.x == 0 && blockIdx.x == 0){

	}
#endif
}

//---------------------------------------------------------------------
// Função na CPU para calcular o T
// float* neurons: todos os neuronios da rede
// float* activity: vetor de ativida da rede
// float* fuzzyAnd: vetor a ser utilizado para armazenar o fuzzy and
// float* neurons_norm_field: vetor para guardar a norma dos neuronios
// float* fuzzy_and_norm_field: vetor para guardar norma fuzzy
// float* t_vector_division: vetor para guardar divisão das normas
// float* t_vector: vetor para guardar o T_vector final
// int* field_sizes: tamanho de todos os campos
// int total_neurons: total de neuronios a serem processador
// int total_fields: total de campos por neuronio
// int field_reserved_space: espaço total reservado para todos os campos
//---------------------------------------------------------------------

void calculate_T_vector(
	FUSION_ART* network,
	float* neurons,
	float* activity,
	float* fuzzy_and,
	float* neurons_norm_field,
	float* fuzzy_and_norm_field,
	float* t_vector_division,
	float* t_vector,
	float* field_gammas,
	int* field_sizes,
	int total_neurons,
	int total_fields,
	int field_reserved_space){

	// Constantes
	float alpha = 0.01f;

	// Guarda numero de threads e blocos a utilizar
	int threads = 0;
	int blocks = 0;

	// Calcula o fuzzyand, deve calcular por celula no vetor neurons. Para isso pegar tamanho total do vetor
	int neurons_size = total_neurons * field_reserved_space;

	{// CALCULO DO FUZZY AND ---------------------------------------------------------------------------------
		// Calcula blocos e threads de acordo com cada celula do vetor neurons
		calculateBlocks(&neurons_size, &threads, &blocks);

		// Calcula fuzzy and para o vetor neurons e a atividade
		block_calculate_fuzzy_and << <blocks, threads >> >(neurons, activity, fuzzy_and, neurons_size);
	}//-------------------------------------------------------------------------------------------------------



	{// CALCULO DAS NORMAS -----------------------------------------------------------------------------------
		// Calcula o total de blocos para efetuar a soma dos campos, utilizar quantidade de neuronios a serem calculados: total_neurons
		calculateBlocks(&total_neurons, &threads, &blocks);

		// Calcula a norma dos pesos
#if defined __FIELD_REDUCTION__
		block_Calculate_fields_sumB(network, neurons, network->sum_reduction_aux, neurons_norm_field);
#elif defined __PER_FIELD_SUM__
		neurons_size = total_neurons * total_fields;
		calculateBlocks(&neurons_size, &threads, &blocks);
		block_Calculate_fields_sumC<<<blocks, threads>>>(neurons, neurons_norm_field, field_sizes, network->network_field_thread_index_aux, total_fields, total_neurons * total_fields);
#else
		block_Calculate_fields_sum << <blocks, threads >> >(neurons, neurons_norm_field, field_sizes, total_neurons, total_fields, field_reserved_space);
#endif

		// Calcula a norma do fuzzy and
#if defined __FIELD_REDUCTION__
		block_Calculate_fields_sumB(network, fuzzy_and, network->sum_reduction_aux, fuzzy_and_norm_field);
#elif defined __PER_FIELD_SUM__
		neurons_size = total_neurons * total_fields;
		calculateBlocks(&neurons_size, &threads, &blocks);
		block_Calculate_fields_sumC << <blocks, threads >> >(fuzzy_and, fuzzy_and_norm_field, field_sizes, network->network_field_thread_index_aux, total_fields, total_neurons * total_fields);
#else
		block_Calculate_fields_sum << <blocks, threads >> >(fuzzy_and, fuzzy_and_norm_field, field_sizes, total_neurons, total_fields, field_reserved_space);
#endif
	}//-------------------------------------------------------------------------------------------------------



	{// ADICIONAR ALPHA A CADA NORMA DOS NEURONIOS e EFETUA A DIVISÃO das normas -----------------------------
		// Calcula numero de campos total considerando cada neuronio ativo
		int total_fields_per_neuron = total_fields * total_neurons;

		// Calcula blocos e threads para numero 
		calculateBlocks(&total_fields_per_neuron, &threads, &blocks);
		block_simple_add << <blocks, threads >> > (neurons_norm_field, alpha, total_fields_per_neuron);

		// Efetua divisão dos elementos da norma do peso e norma do fuzzy and
		block_fuzzy_art_division << <blocks, threads >> >(fuzzy_and_norm_field, neurons_norm_field, t_vector_division, total_fields_per_neuron);
	}//-------------------------------------------------------------------------------------------------------



	{// CALCULO FINAL DE T COM SOMATORIO DOS CAMPOS DEPOIS DA DIVISÃO ----------------------------------------
		// Efetua a soma dos campos
		calculateBlocks(&total_neurons, &threads, &blocks);
		block_Calculate_final_T_vector << <blocks, threads >> >(t_vector_division, t_vector, field_gammas, total_neurons, total_fields);

		// Multiplica pelo gamma cada valor do T_vetor
		//block_simple_mul << <blocks, threads >> >(t_vector, gamma, total_neurons);
	}//-------------------------------------------------------------------------------------------------------

#if defined __T_VECTOR_DEBUG__
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != CUDA_SUCCESS)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
#endif
}

//---------------------------------------------------------------------
// Função na CPU para calcular o T
// float* neurons: todos os neuronios da rede
// float* activity: vetor de ativida da rede
// float* fuzzyAnd: vetor a ser utilizado para armazenar o fuzzy and
// float* neurons_norm_field: vetor para guardar a norma dos neuronios
// float* fuzzy_and_norm_field: vetor para guardar norma fuzzy
// float* resonance_division: vetor para guardar divisão das normas
// float* ressonating_neurons: vetor para guardar o T_vector final
// float precision: precisão a ser utilizada para verificar resonancia
// int* field_sizes: tamanho de todos os campos
// int total_neurons: total de neuronios a serem processador
// int total_fields: total de campos por neuronio
// int field_reserved_space: espaço total reservado para todos os campos
//---------------------------------------------------------------------

void calculate_resonance(
	FUSION_ART* network,
	float* neurons,
	float* activity,
	float* neurons_norm_field,
	float* fuzzy_and_norm_field,
	float* resonance_division,
	float* ressonating_neurons,
	float precision,
	int* field_sizes,
	int total_neurons,
	int total_fields,
	int field_reserved_space){

	// Guarda numero de threads e blocos a utilizar
	int threads = 0;
	int blocks = 0;

	{// CALCULA NORMA DA ATIVIDADE COM OS PESOS -------------------------------------------------------------
		// Calcula blocos e threads de acordo com total de neuronios para efetuar aredução da atividade
		calculateBlocks(&total_neurons, &threads, &blocks);

		// Calcula a normal da atividade
#if defined __FIELD_REDUCTION__
		block_Calculate_fields_sumB(network, activity, network->sum_reduction_aux, neurons_norm_field);
#elif defined __PER_FIELD_SUM__
		int neurons_size = total_neurons * total_fields;
		calculateBlocks(&neurons_size, &threads, &blocks);
		block_Calculate_fields_sumC << <blocks, threads >> >(activity, neurons_norm_field, field_sizes, network->network_field_thread_index_aux, total_fields, total_neurons * total_fields);
#else
		block_Calculate_fields_sum << <blocks, threads >> >(activity, neurons_norm_field, field_sizes, total_neurons, total_fields, field_reserved_space);
#endif
	}//-------------------------------------------------------------------------------------------------------



	{// CALCULA DIVISÃO DA NORMA DO FUZZY AND e NORMA DA ATIVIDADE -------------------------------------------
		// Calcular a divisão para cada campo de cado neuronio
		int total_fields_per_neuron = total_fields * total_neurons;

		// Calcular blocos e threads de acordo com total de campos por neuronio para cada neuronio
		calculateBlocks(&total_fields_per_neuron, &threads, &blocks);

		// Efetua divisão dos elementos da norma do peso e norma do fuzzy and
		block_fuzzy_art_division << <blocks, threads >> >(fuzzy_and_norm_field, neurons_norm_field, resonance_division, total_fields_per_neuron);
	}//-------------------------------------------------------------------------------------------------------



	{// CALCULA VETOR DE RESONANCIA FINAL --------------------------------------------------------------------
		// Calcula blocos e threads para o total de neuronios a serem verificados
		calculateBlocks(&total_neurons, &threads, &blocks);

		// Efetua o calculo da resonancia por neuronio
		block_Calculate_resonated_vector << <blocks, threads >> >(resonance_division, ressonating_neurons, precision, total_neurons, total_fields);
	}//-------------------------------------------------------------------------------------------------------

#if defined __RESONANCE_VECTOR_DEBUG__
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != CUDA_SUCCESS)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
#endif
}

//---------------------------------------------------------------------
// Efetua a leitura da atividade do neuronio selecionado na rede
// FUSION_ART* network: rede apos efetuar todos os calculos
// int selected_neuron: neuronio selecionado para efetuar o readout
//---------------------------------------------------------------------

void activity_read_out(FUSION_ART* network, int selected_neuron){
	// Indice que será efetuado o readout para a CPU
	int reading_index = network->all_fields_reserved_space * selected_neuron;

	// Efetua copia de espaço de memoria do neuronio que dara o readout
	cudaMemcpy(network->readout_CPU, network->fuzzy_and + reading_index, sizeof(float) * network->all_fields_reserved_space, cudaMemcpyDeviceToHost);
}

//---------------------------------------------------------------------
// Copia vetores de atividade para a rede
// FUSION_ART* network: rede onde vai copiar a atividade para a atividade na GPU
//---------------------------------------------------------------------

void inputToGPUActivity(FUSION_ART* network, float* super_input){
	int field_start_index = 0;

	// Para cada campo inicializa campo no vetor super_input
	for (unsigned int field = 0; field < network->fields_input.size(); field++){
		// Pega tamanho do campo para calcular largura do passo na memoria
		int field_size = network->field_sizes_CPU[field];

		// Copia campo para a memoria no super_input utilizando tamanho do campo e posição inicial dada pelo field_start_index
		cudaMemcpy(&super_input[field_start_index], network->fields_input[field], sizeof(float) * field_size, cudaMemcpyHostToHost);

		// Atualiza posição inicial para copiar o proximo campo
		field_start_index += field_size;
	}

	// Copia vetor de forma distribuida para vetor na rede para enviar para a GPU
	for (int neuron = 0; neuron < network->available_neurons; neuron++){
		int neuron_index = neuron * network->all_fields_reserved_space;
		cudaMemcpy(&network->activity_CPU[neuron_index], super_input, sizeof(float) * network->all_fields_reserved_space, cudaMemcpyHostToHost);
	}

	// Copia super entrada para a GPU
	cudaMemcpy(network->activity, network->activity_CPU, sizeof(float) * network->available_neurons * network->all_fields_reserved_space, cudaMemcpyHostToDevice);
}

//---------------------------------------------------------------------
// Apenas imprime a memoria livre e utilizada na GPU
//---------------------------------------------------------------------

void printFreeMem(){
	size_t free, total;
	cudaMemGetInfo(&free, &total);
	free /= 1000000;
	total /= 1000000;
	printf("Total de memoria utilizado: %d MegaBytes\n", total - free);
	printf("Total de memoria liberado: %d MegaBytes\n", free);
	printf("Total de memoria: %d MegaBytes\n", total);
}

//---------------------------------------------------------------------
// Executa uma iteração de consulta a rede, pode ser em modo aprendizado ou não
// FUSION_ART* network: rede a ser utilizada
// float learning_rate: taxa de aprendizado
// float resonance_precision: fator de resonancia a ser utilizado
// bool learning: flag para saber se deve aprender ou não
// bool debug: flag para saber se deve imprimir variaveis de depuração
//---------------------------------------------------------------------

int run_network(FUSION_ART* network, float learning_rate, float resonance_precision, bool learning, bool debug){
	// Todas as operações são feitas em paralelo, então executa tudo de uma vez mesmo que não seja necessário
	int threads = 1; // valor padrão
	int blocks = 1; // valor padrão

	// Neuronio a retornar caso faça alguma checagem
	int selected_neuron = 0;

	// Garantir que nunca fara uma busca em neuronios fora do total armazenado
	if (network->available_neurons > network->neuron_count)
		network->available_neurons = network->neuron_count;

	// Calcula vetor T
	calculate_T_vector(
		network,
		network->neurons,
		network->activity,
		network->fuzzy_and,
		network->neurons_norm_field,
		network->fuzzy_and_norm_field,
		network->t_vector_division,
		network->t_vector,
		network->fields_gammas,
		network->field_sizes,
		network->available_neurons, // não usar neuron_count, caso contrario irá calcular um vetor imenso de cara
		network->total_fields,
		network->all_fields_reserved_space);

	// Aprender apenas se necessário
	if (!learning){
		calculateBlocks(&network->available_neurons, &threads, &blocks);
		callReductionMax(network->t_vector, network->network_reduction_aux_T_vector, network->network_reduction_aux_T_vector_index, network->last_selected_max_index, network->available_neurons, false);
	}
	else{
		// Verifica resonancia
		calculate_resonance(
			network,
			network->neurons,
			network->activity,
			network->neurons_norm_field,
			network->fuzzy_and_norm_field,
			network->resonance_division,
			network->ressonating_neurons,
			resonance_precision,
			network->field_sizes,
			network->available_neurons, // não usar neuron_count, caso contrario irá calcular um vetor imenso de cara
			network->total_fields,
			network->all_fields_reserved_space);

		// Soma para calcular resonancia
		calculateBlocks(&network->available_neurons, &threads, &blocks);
		block_simple_vecadd<<<blocks, threads>>>(network->t_vector, network->ressonating_neurons, network->t_vecPlusResonating_neurons, network->available_neurons);

		// Pega o max como neuronio selecionado
		callReductionMax(network->t_vecPlusResonating_neurons, network->network_reduction_aux_T_vector, network->network_reduction_aux_T_vector_index, network->last_selected_max_index, network->available_neurons, false);

		if (debug)
			printf("Execution end.\n");

		// Calcula blocos e threads para uma thread por variavel no vetor neurons
		calculateBlocks(&network->all_fields_reserved_space, &threads, &blocks);

		// Efetua aprendizado
		learn << <blocks, threads >> >(
			network->neurons,
			network->fuzzy_and,
			learning_rate,
			network->last_selected_max_index,
			network->all_fields_reserved_space);

#if defined __DEBUG__
		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != CUDA_SUCCESS)
			printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
#endif
	}

	// Efetua copia do neuronio selecionado
	cudaMemcpy(&selected_neuron, network->last_selected_max_index, sizeof(int), cudaMemcpyDeviceToHost);
	selected_neuron = network->available_neurons - 1;
	// Se selecionado neurônio não comitado comitar ele e criar um novo
	if (selected_neuron == network->available_neurons - 1)
		network->available_neurons++;

	// Copia resultado da resonancia para o vetor de readout
	//activity_read_out(network, selected_neuron);

	// retorna neuronio selecionado
	return selected_neuron;
}

//---------------------------------------------------------------------
// Função de teste das operações na rede.
// Dede criar 4 neuronios e nunca resonar.
//---------------------------------------------------------------------

void test_function(){
	FUSION_ART network;
	NET_CONFIG config;

	// Cria objeto de configuração da rede
	config.total_fields = 2;
	int* field_sizes = new int[config.total_fields];
	field_sizes[0] = 4;
	field_sizes[1] = 1;
	config.fields_sizes = field_sizes;

	float* field_gammas = new float[config.total_fields];
	field_gammas[0] = 1.0f;
	field_gammas[1] = 1.0f;
	config.fields_gammas = field_gammas;

	// Cria rede na GPU
	createNetwork(&config, &network);

	// Aglomera todos os fields em um super vetor para mandar para a GPU
	float* super_input; 
	cudaMallocHost((void**)&super_input, sizeof(float)*network.all_fields_reserved_space);

	// Referencia das variaveis de entrada para manipulação
	float* field_input = network.fields_input[0];
	float* reward_field = network.fields_input[1];

	// Configura inputs para teste
	field_input[0] = 0.1f;
	field_input[1] = 0.1f;
	field_input[2] = 0.1f;
	field_input[3] = 0.1f;
	reward_field[0] = 1.0f;

	// Copia fields para GPU
	//inputToGPUActivity(&network, super_input);

	// Roda rede
	float learning_rate = 1.0f;
	float resonance_precision = 0.99f;
	bool debug = true;
	bool learning = true;

	printf("Press any key to start the tests..."); getchar();

	int selected_neuron = run_network(&network, learning_rate, resonance_precision, learning, debug);

	// Configura inputs para teste
	field_input[0] = 1.0f;
	field_input[1] = 0.0f;
	field_input[2] = 0.0f;
	field_input[3] = 0.0f;
	reward_field[0] = 0.0f;

	// Copia fields para GPU
	inputToGPUActivity(&network, super_input);

	selected_neuron = run_network(&network, learning_rate, resonance_precision, learning, debug);

	// Configura inputs para teste
	field_input[0] = 0.0f;
	field_input[1] = 0.5f;
	field_input[2] = 0.5f;
	field_input[3] = 0.5f;
	reward_field[0] = 0.5f;

	// Copia fields para GPU
	inputToGPUActivity(&network, super_input);

	selected_neuron = run_network(&network, learning_rate, resonance_precision, learning, debug);

	// Configura inputs para teste
	field_input[0] = 0.0f;
	field_input[1] = 1.0f;
	field_input[2] = 0.8f;
	field_input[3] = 0.2f;
	reward_field[0] = 0.7f;

	// Copia fields para GPU
	inputToGPUActivity(&network, super_input);

	selected_neuron = run_network(&network, learning_rate, resonance_precision, learning, debug);

	// Configura inputs para teste
	field_input[0] = 1.0f;
	field_input[1] = 1.0f;
	field_input[2] = 0.8f;
	field_input[3] = 0.2f;
	reward_field[0] = 0.2f;

	// Copia fields para GPU
	inputToGPUActivity(&network, super_input);

	selected_neuron = run_network(&network, learning_rate, resonance_precision, learning, debug);

	block_Calculate_fields_sumB(&network, network.neurons, network.sum_reduction_aux, network.neurons_norm_field);
}

//---------------------------------------------------------------------
// Função de teste das rotinar de redução e max
// Deve imprimir valores consistentes em todo o teste, 
// caso contrario tem algum problema
//---------------------------------------------------------------------

void reduction_test(){
	int arrSize = 100000;

	float* A; cudaMalloc((void**)&A, sizeof(float) * arrSize);
	float* C; cudaMalloc((void**)&C, sizeof(float) * arrSize);
	int*  max; cudaMalloc((void**)&max, sizeof(int) * arrSize);

	float* a = new float[arrSize];
	for (int i = 0; i < arrSize; i++)
		a[i] = (float)i;

	cudaMemcpy(A, a, sizeof(float)*arrSize, cudaMemcpyHostToDevice);

	for (int i = 0; i <= 100000; i++){
	//	callReduction(A, C, i, true);
	}

	for (int i = 1; i < 100000; i++){
//		callReductionMax(A, C, max, i, true);
	}

#if defined __DEBUG__
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != CUDA_SUCCESS)
		printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
#endif
}

//---------------------------------------------------------------------
// Função de execução da rede
//---------------------------------------------------------------------

void netfunc(){
	printf("Criando FUSION ART...\n");

	FUSION_ART network;
	NET_CONFIG config;

	// Cria objeto de configuração da rede
	config.total_fields = 9;
	int* field_sizes; cudaMallocHost((void**)&field_sizes, sizeof(int) * config.total_fields);// new int[config.total_fields];

	// Environment
	field_sizes[0] = 11; //ambiente
	field_sizes[1] = 11; //ambiente
	field_sizes[2] = 11; //ambiente
	field_sizes[3] = 11; //ambiente
	field_sizes[4] = 11; //ambiente
	field_sizes[5] = 11; //ambiente
	field_sizes[6] = 10; //ambiente
	field_sizes[7] = 20; //ações
	field_sizes[8] = 1;  //reward

	float* field_gammas; cudaMallocHost((void**)&field_gammas, sizeof(float) * config.total_fields); //new float[config.total_fields];
	field_gammas[0] = 1.0f; //ambiente
	field_gammas[1] = 1.0f; //ambiente
	field_gammas[2] = 1.0f; //ambiente
	field_gammas[3] = 1.0f; //ambiente
	field_gammas[4] = 1.0f; //ambiente
	field_gammas[5] = 1.0f; //ambiente
	field_gammas[6] = 1.0f; //ambiente
	field_gammas[7] = 1.0f; //ações
	field_gammas[8] = 1.0f; //reward

	// Configura fields
	config.fields_sizes = field_sizes;

	// Configura gamas
	config.fields_gammas = field_gammas;

	// Aglomera todos os fields em um super vetor para mandar para a GPU
	float* super_input; cudaMallocHost((void**)&super_input, sizeof(float)*network.all_fields_reserved_space);

	printf("Inicializando cerebro e campos sensoriais...\n");

	// Cria rede na GPU
	createNetwork(&config, &network);

	// Configurações
	float learning_rate = 0.8f;
	float resonance_precision = 0.50f;
	bool debug = false;
	bool learning = true;

	// Contagem de execução
	int iteractions = 0;
	int total_iteractions = 0;
	float execution_time_sum = 0.0f;

	printf("\n---------Parametros---------\nTaxa de aprendizado: %f\nPrecisao da resonancia: %f\nModo de depuracao: %d\nModo de aprendizado: %d\n---------Parametros---------\n\n", learning_rate, resonance_precision, debug, learning);

	// -----------------------------------------------------------------------------------------
	// Hearthstone network ---------------------------------------------------------------------
	// -----------------------------------------------------------------------------------------

	const int pipe_buffer_size = 2048;

	// Cria um pipe padrão para operações
	HANDLE _named_pipe;
	DWORD  cbRead, cbWritten;
	TCHAR  inputBuffer[pipe_buffer_size];
	TCHAR  outputBuffer[pipe_buffer_size];

	printf("Configurando pipe de comunicacao...\n");

	// Cria o pipe com a chamada de sistema
	_named_pipe = CreateNamedPipe(
		"\\\\.\\pipe\\fusionartpipe", // nome do pipe
		PIPE_ACCESS_DUPLEX, // tipo do pipe, mão única dupla, duplex...
		PIPE_TYPE_BYTE, // modo de armazenamento do pipe
		1, // quantidade de pipes que podem ser criados
		pipe_buffer_size, // tamanho do buffer de saida
		pipe_buffer_size, // tamanho do buffer de entrada
		0, // Tempo de espera para criação, 0 resulta em tempo padrão de 50ms
		PIPE_ACCEPT_REMOTE_CLIENTS); // atributos de segurança

	int last_err = GetLastError();

	printf("Cerebro pronto para ser utilizado.\n");
	while (true){
		// Tenta ler do pipe com chamada de sistema
		BOOL _opState = ReadFile(
			_named_pipe,
			inputBuffer,
			sizeof(TCHAR) * pipe_buffer_size,
			&cbRead,
			NULL);

		// Faz as operações com os valores lidos
		if (_opState == TRUE){

#if defined __VERBOSE__
			printf("Requisicao recebida, processando...\n");
#endif

			// Tratar a mensagem aqui para poder enviar uma mensagem
			std::string message(inputBuffer, cbRead);
			//printf("Message received from client: \n%s \n", message.c_str());
			//printf("Sending results to client...\n");

			// Organizar entradas
			std::istringstream iss(message);
			std::vector<std::string> input{ std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{} };

			// Preparar todos os campos de entrada
			int j = 0;
			for (unsigned int i = 0; i < network.fields_input.size(); i++){
				float* field_i = network.fields_input[i];
				int field_i_size = network.field_sizes_CPU[i];

				for (int k = 0; k < field_i_size; k++, j++){
					float value = (float)atof(input[j].c_str());
					field_i[k] = value;
				}
			}

			// Configura modo de operação
			learning = (atoi(input[input.size() - 1].c_str()) == 1);

			// Executar FUSION
			//inputToGPUActivity(&network, super_input);

			// Medir tempo de execução
			clock_t tStart = clock();

			int	selected_neuron = run_network(&network, learning_rate, resonance_precision, learning, debug);

			// Medir tempo de execução
			clock_t tEnd = clock();

			// Soma tempo de execução para pegar média
			execution_time_sum += ((float)(tEnd - tStart) / CLOCKS_PER_SEC);

			// Total de iterações do algoritmo
			iteractions++;
			total_iteractions++;

			// Imprimir apenas após mil execuções
			if (iteractions % 500 == 0){
				printf("\nMedia do tempo de execucao das rotinas da rede: %fs\n", execution_time_sum / (float)iteractions);
				printFreeMem();
				printf("Total de neuronios utilizados ate o momento: %d\n", network.available_neurons);
				printf("Total de iteracoes ate o momento: %d\n", total_iteractions);
				printf("----------------------------------------------------------\n");

				// Reseta variaveis para nao poluir avaliação
				iteractions = 0;
				execution_time_sum = 0.0f;
			}

			// Prepara mensagem de saida da rede
			std::string sending_message;
			for (int i = 0; i < network.all_fields_reserved_space; i++){
				std::string number = std::to_string(network.readout_CPU[i]);
				sending_message.append(number);
				sending_message.append(" ");
			}
			sending_message.append("\n");
			strcpy(outputBuffer, sending_message.c_str());

			// Enviar resposta
			BOOL _sendState = WriteFile(
				_named_pipe,
				outputBuffer,
				strlen(outputBuffer) * sizeof(TCHAR),
				&cbWritten,
				NULL);

#if defined __VERBOSE__
			if (_sendState){
				printf("Resultados enviados com sucesso...\n");
			}
#endif
		}
	}
}

/**
 * Main
 */
int main(int argc, char** argv)
{
	// Função de execução principal da rede
	netfunc();
	
	// Testes de redução
	//reduction_test();

	// Testes de funcionamento da rede
	//test_function();
	getchar();
	return 0;
}

