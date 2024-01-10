#pragma once

#include <chrono>
#include <random>
#include <complex>

// Ripped from the below link for using the fancy new C++ random library
// https://www.learncpp.com/cpp-tutorial/generating-random-numbers-using-mersenne-twister/
template <typename T>
class Random
{
public:
    Random() : mt{ CreateSeedSeq() } { }
    Random(long unsigned int seed) : mt{ seed } { }

    T Generate() { return std::uniform_real_distribution<T>{0, 1}(mt); }
    T operator()() { return Generate(); }

private:
    std::mt19937 CreateSeedSeq()
    {
        std::random_device rd{};
        std::seed_seq ss { 
            static_cast<std::seed_seq::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()),
            rd(), rd(), rd(), rd(), rd(), rd(), rd()
        };
        
        return std::mt19937{ ss };
    }

private:
    std::mt19937 mt;
};

template <>
class Random<std::complex<double>>
{
    using T = std::complex<double>;
public:
    Random() : mt{ CreateSeedSeq() } { }
    Random(long unsigned int seed) : mt{ seed } { }

    T Generate() 
    {
         return T(
            std::uniform_real_distribution<T::value_type>{0, 1}(mt), 
            std::uniform_real_distribution<T::value_type>{0, 1}(mt)
        );
    }

    T GenerateReal()
    {
        std::uniform_real_distribution<T::value_type>{0, 1}(mt); // Discard one
         return T(
            std::uniform_real_distribution<T::value_type>{0, 1}(mt), 
            0
        );
    }

    T GenerateImaginary()
    {
        std::uniform_real_distribution<T::value_type>{0, 1}(mt); // Discard one
         return T(
            0, 
            std::uniform_real_distribution<T::value_type>{0, 1}(mt)
        );
    }
    
    T operator()() { return Generate(); }

private:
    std::mt19937 CreateSeedSeq()
    {
        std::random_device rd{};
        std::seed_seq ss { 
            static_cast<std::seed_seq::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()),
            rd(), rd(), rd(), rd(), rd(), rd(), rd()
        };
        
        return std::mt19937{ ss };
    }

private:
    std::mt19937 mt;
};