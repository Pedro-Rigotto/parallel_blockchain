#include <omp.h>
#include <atomic>
#include "Block.h"
#include "sha256.h"

Block::Block(uint32_t nIndexIn, const string &sDataIn) : _nIndex(nIndexIn), _sData(sDataIn)
{
    _nNonce = 0;
    _tTime = time(nullptr);

    sHash = _CalculateHash();
}

void Block::MineBlock(uint32_t nDifficulty)
{
    char cstr[nDifficulty + 1];
    for (uint32_t i = 0; i < nDifficulty; ++i)
    {
        cstr[i] = '0';
    }
    cstr[nDifficulty] = '\0';

    string str(cstr);

    std::atomic<bool> found(false); // Variavel para sinalizar que um hash válido foi encontrado
    uint32_t localNonce = 0;
    string localHash;

    #pragma omp parallel private(localNonce, localHash) // Cada thread trabalha em um while, que continua a executar enquanto nenhum thread encontrar um hash válido
    {
        while (!found.load()) {
            #pragma omp critical
            {
                if (!found.load()) {
                    localNonce = ++_nNonce;
                    localHash = _CalculateHash();
                    if (localHash.substr(0, nDifficulty) == str) {
                        found.store(true);
                        sHash = localHash;
                    }
                }
            }
        }
    }

    cout << "Block mined: " << sHash << endl;
}

inline string Block::_CalculateHash() const
{
    stringstream ss;
    ss << _nIndex << sPrevHash << _tTime << _sData << _nNonce;

    return sha256(ss.str());
}
