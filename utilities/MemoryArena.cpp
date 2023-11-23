#include "MemoryArena.h"

#include <algorithm>
#include <assert.h>

MemoryArenaPool MemoryArenaPool::s_globalPool;

MemoryArenaPool::~MemoryArenaPool(void)
{
    Chunk* curr;
    
    curr = m_free;
    while (curr)
    {
        free(curr->memory);
        curr = curr->next;
    }

    curr = m_allocd;
    while (curr)
    {
        free(curr->memory);
        curr = curr->next;
    }
}

void* MemoryArenaPool::GetMemory(uint32_t size)
{
    // Dummy head to avoid edge cases
    Chunk dummy, *prev, *curr;

    // Find first chunk in free list which can fulfill memory request
    dummy.next = m_free;
    prev = &dummy;
    curr = dummy.next;
    
    while (curr != NULL && curr->size < size)
    {
        prev = curr;
        curr = curr->next;
    }

    Chunk* c = NULL;
    if (curr)
    {   
        // Remove found chunk from free list
        prev->next = curr->next;
        c = curr;
    }
    else
    {
        // If no suitable chunks found, add one at the insertion location (to keep cache ordered by size)
        c = new Chunk{ malloc(size), size, NULL };
    }

    // Link suitable chunk to the alloc'ed list
    c->next = m_allocd;
    m_allocd = c;

    // Fulfill request
    return c->memory;
} 

void MemoryArenaPool::FreeMemory(void* mem)
{
    // Dummy head to avoid edge cases
    Chunk dummy, *prev, *curr;
    
    // Find owning chunk in allocated list
    dummy.next = m_allocd;
    prev = &dummy;
    curr = dummy.next;

    while (curr != NULL && curr->memory != mem)
    {
        prev = curr;
        curr = curr->next;
    }

    // Bail if chunk not found
    if (curr == NULL)
        return;

    // Fix-up alloc list; cache found chunk
    prev->next = curr->next;
    Chunk* c = curr;

    // Find position in free list to insert freed chunk (ordered by chunk size)
    dummy.next = m_free;
    prev = &dummy;
    curr = dummy.next;

    while (curr != NULL && curr->size < c->size)
    {
        prev = curr;
        curr = curr->next;
    }

    // Insert into chunk list
    prev->next = c;
    c->next = curr;
}

MemoryArena::MemoryArena(MemoryArenaPool& pool, uint32_t maxSize)
    : m_pool(&pool)
    , m_memory(pool.GetMemory(maxSize))
    , m_size(maxSize)
    , m_offset(0)
{ }

MemoryArena::~MemoryArena()
{
    m_pool->FreeMemory(m_memory);
    m_memory = NULL;
    m_size = 0;
}