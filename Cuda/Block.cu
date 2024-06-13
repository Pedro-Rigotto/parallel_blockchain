#define NUM_THREADS 384
#define MAX_NONCE 100000000

#include "Block.cuh"
#include "sha256.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <sstream>

Block::Block(uint32_t nIndexIn, const std::string &sDataIn) : _nIndex(nIndexIn), _sData(sDataIn)
{
    _nNonce = 0;
    _tTime = time(nullptr);

    sHash = _CalculateHash();
}

__device__ unsigned int validHashFound = 0;

__device__ void uint32ToString(uint32_t value, char* buffer) {
    int i = 0;
    do {
        buffer[i++] = (value % 10) + '0';
        value /= 10;
    } while (value != 0);
    buffer[i] = '\0';
    
    // Reverter a string
    for (int j = 0; j < i / 2; ++j) {
        char temp = buffer[j];
        buffer[j] = buffer[i - j - 1];
        buffer[i - j - 1] = temp;
    }
}

__device__ int stringLength(const char* str) {
    int length = 0;
    while (str[length] != '\0') {
        length++;
    }
    return length;
}

__device__ void stringConcat(char* dest, const char* src) {
    int destLen = stringLength(dest);
    int i = 0;
    while (src[i] != '\0') {
        dest[destLen + i] = src[i];
        i++;
    }
    dest[destLen + i] = '\0';
}

__device__ void timeToString(time_t value, char* buffer) {
    uint32ToString(static_cast<uint32_t>(value), buffer);
}

__device__ bool compareStrings(const char* str1, const char* str2) {
    while (*str1 && (*str1 == *str2)) {
        ++str1;
        ++str2;
    }
    return (*str1 - *str2) == 0;
}

__device__ void myStrcpy(char* dest, const char* src) {
    while (*src) {
        *dest++ = *src++;
    }
    *dest = '\0';
}

__device__ void mySubstr(const char* input, char* output, int start, int length) {
    int i;
    for (i = 0; i < length && input[start + i] != '\0'; ++i) {
        output[i] = input[start + i];
    }
    output[i] = '\0';
}

__global__ void MineKernel(uint32_t nDifficulty, const char* str, uint32_t _nIndex, const char* sPrevHash, time_t _tTime, const char* _sData, uint32_t* resultNonce, char* resultHash) {
    __shared__ int found;
    if (threadIdx.x == 0) {
        found = 0;
    }
    __syncthreads();
    
    if(atomicAdd(&found, 0) == 1) return; // Se um hash válido foi encontrado, retorne (saia do kernel)

    // Calculate the thread's unique nonce
    uint32_t _nNonce = blockIdx.x * blockDim.x + threadIdx.x;

    // Prepare the string to be hashed
    char indexBuffer[11]; // buffer para _nIndex (máximo 10 dígitos + \0)
    char timeBuffer[21];  // buffer para _tTime (máximo 20 dígitos + \0)
    char nonceBuffer[11]; // buffer para _nNonce (máximo 10 dígitos + \0)
    uint32ToString(_nIndex, indexBuffer);
    timeToString(_tTime, timeBuffer);
    uint32ToString(_nNonce, nonceBuffer);

    int bufferSize = stringLength(indexBuffer) + stringLength(sPrevHash) + stringLength(timeBuffer) + stringLength(_sData) + stringLength(nonceBuffer);

    // Use um buffer local em vez de alocação dinâmica
    char outputBuffer[256];
    if (bufferSize + 1 > sizeof(outputBuffer)) {
        return; // Evita estouro de buffer
    }
    outputBuffer[0] = '\0';

    stringConcat(outputBuffer, indexBuffer);
    stringConcat(outputBuffer, sPrevHash);
    stringConcat(outputBuffer, timeBuffer);
    stringConcat(outputBuffer, _sData);
    stringConcat(outputBuffer, nonceBuffer);

    // Perform the hash calculation
    char* sHash = sha256(outputBuffer);

    // Check if the hash meets the difficulty criteria
    char tempStr[65];
    mySubstr(sHash, tempStr, 0, nDifficulty);
    if (compareStrings(tempStr, str)) {
        atomicExch(&validHashFound, 1); // Set the flag using an atomic exchange
        if (atomicExch(&found, 1) == 0) {
            *resultNonce = _nNonce;
            myStrcpy(resultHash, sHash);
        }
    }
}

void Block::MineBlock(uint32_t nDifficulty)
{
    // Prepare the difficulty string
    char cstr[nDifficulty + 1];
    for (uint32_t i = 0; i < nDifficulty; ++i)
    {
        cstr[i] = '0';
    }
    cstr[nDifficulty] = '\0';

    // Allocate memory on the GPU for the result nonce and hash
    uint32_t* d_resultNonce;
    char* d_resultHash;
    cudaMalloc(&d_resultNonce, sizeof(uint32_t));
    cudaMalloc(&d_resultHash, 65); // SHA256 hash is 64 characters + null terminator

    // Reset the validHashFound flag to false
    unsigned int flag = 0;
    cudaMemcpyToSymbol(validHashFound, &flag, sizeof(unsigned int));

    // Copy the block data to the GPU
    uint32_t h_nIndex = _nIndex;
    time_t h_tTime = _tTime;
    char* d_sPrevHash;
    char* d_sData;
    cudaMalloc(&d_sPrevHash, sPrevHash.length() + 1);
    cudaMalloc(&d_sData, _sData.length() + 1);

    cudaMemcpy(d_sPrevHash, sPrevHash.c_str(), sPrevHash.length() + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sData, _sData.c_str(), _sData.length() + 1, cudaMemcpyHostToDevice);

    // Calculate the number of blocks and threads
    uint32_t numThreads = NUM_THREADS;
    uint32_t numBlocks = (MAX_NONCE + numThreads - 1) / numThreads;

    // Launch the kernel
    MineKernel<<<numBlocks, numThreads>>>(nDifficulty, cstr, h_nIndex, d_sPrevHash, h_tTime, d_sData, d_resultNonce, d_resultHash);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Check for any errors during kernel execution
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        // Print the CUDA error message and exit
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Check if a valid hash was found
    uint32_t resultNonce;
    char resultHash[65];
    cudaMemcpy(&resultNonce, d_resultNonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(resultHash, d_resultHash, 65, cudaMemcpyDeviceToHost);

    if (resultNonce != 0) // Assuming 0 is the initial value and means no valid nonce was found
    {
        std::cout << "Block mined: " << resultHash << std::endl;
    }
    else
    {
        std::cout << "No valid hash found within the nonce range." << std::endl;
    }

    // Free the GPU memory
    cudaFree(d_resultNonce);
    cudaFree(d_resultHash);
    cudaFree(d_sPrevHash);
    cudaFree(d_sData);
}

inline std::string Block::_CalculateHash() const
{
    std::stringstream ss;
    ss << _nIndex << sPrevHash << _tTime << _sData << _nNonce;

    return sha256host(ss.str());
}
