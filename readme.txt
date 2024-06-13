Tempos de execução no parcode

- Versão sequencial: 16m11.073s
- Versão OpenMP Multicore: 4m46.945s
- Versão OpenMP GPU: 4m11.007s
- Versão CUDA: segmentation fault

Speedups

- Versão sequencial para versão OpenMP multicore: 3.38
- Versão sequencial para versão OpenMP GPU: 3.87

Para compilar

- OpenMP: gcc8 -lstdc++ -o TestChain -std=c++11 -x c++ main.cpp -fopenmp Block.cpp Blockchain.cpp sha256.cpp
- CUDA: nvcc -o blockchain -std=c++11 main.cpp Block.cu Blockchain.cpp sha256.cu -lcudart -rdc=true