#pragma once

#include <cstdint>
#include <cstdlib>
#include <assert.h>
#include <utility>

class MemoryArenaPool;

class MemoryArena
{
public:
    MemoryArena(MemoryArenaPool& pool, uint32_t maxSize);
    ~MemoryArena(void);

    MemoryArena(const MemoryArena&) = delete;
    MemoryArena& operator=(const MemoryArena&) = delete;


    void* Allocate(int size)
    {
        assert(m_offset + size <= m_size);

        auto mem = static_cast<uint8_t*>(m_memory) + m_offset;
        m_offset += size;

        return mem;
    }

    template <typename T>
    T* Allocate(int count)
    {
        return static_cast<T*>(Allocate(count * sizeof(T)));
    }

private:
    MemoryArenaPool* m_pool;
    void* m_memory;
    uint32_t m_size;
    uint32_t m_offset;
};

class MemoryArenaPool
{
public:
    MemoryArenaPool(void)
        : m_free(NULL)
        , m_allocd(NULL)
    { }

    ~MemoryArenaPool(void);

    static MemoryArena GetArena(uint32_t maxSize)
    {
        return MemoryArena(s_globalPool, maxSize);
    }

    template <typename T>
    static MemoryArena GetArena(uint32_t count)
    {
        return GetArena(count * sizeof(T));
    }

private:
    // Private interface for MemoryArena
    void* GetMemory(uint32_t maxSize);
    void FreeMemory(void* mem);

private:
    // Memory chunk holder with singly-linked in-list
    struct Chunk
    {
        void*    memory;
        uint32_t size;
        Chunk*   next;
    };

    // Free & allocated memory chunk list
    Chunk* m_free;
    Chunk* m_allocd;

    static MemoryArenaPool s_globalPool;

    friend class MemoryArena;
};