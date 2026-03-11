#ifndef MM_THREAD_H
#define MM_THREAD_H

#include <stdint.h>
#include <stdlib.h>

/* ============================================================
   Windows implementation
   ============================================================ */

#if defined(_WIN32)

#include <windows.h>

typedef HANDLE mm_thread_t;
typedef CRITICAL_SECTION mm_mutex_t;

struct mm_thread_start {
    int (*fn)(void*);
    void *arg;
};

static DWORD WINAPI mm_thread_trampoline(LPVOID p)
{
    struct mm_thread_start s = *(struct mm_thread_start*)p;
    free(p);
    return (DWORD)s.fn(s.arg);
}

static inline int mm_thread_create(mm_thread_t *t, int (*fn)(void*), void *arg)
{
    struct mm_thread_start *s = malloc(sizeof(*s));
    if (!s) return -1;

    s->fn = fn;
    s->arg = arg;

    *t = CreateThread(NULL,0,mm_thread_trampoline,s,0,NULL);
    return *t ? 0 : -1;
}

static inline int mm_thread_join(mm_thread_t t, int *result)
{
    WaitForSingleObject(t, INFINITE);

    if (result) {
        DWORD code;
        GetExitCodeThread(t, &code);
        *result = (int)code;
    }

    CloseHandle(t);
    return 0;
}

static inline void mm_mutex_init(mm_mutex_t *m)
{
    InitializeCriticalSection(m);
}

static inline void mm_mutex_lock(mm_mutex_t *m)
{
    EnterCriticalSection(m);
}

static inline void mm_mutex_unlock(mm_mutex_t *m)
{
    LeaveCriticalSection(m);
}

static inline void mm_mutex_destroy(mm_mutex_t *m)
{
    DeleteCriticalSection(m);
}

static inline void mm_sleep_ms(uint32_t ms)
{
    Sleep(ms);
}

/* ============================================================
   POSIX implementation
   ============================================================ */

#elif defined(__unix__) || defined(__APPLE__) || defined(__linux__)

#include <pthread.h>
#include <time.h>

typedef pthread_t mm_thread_t;
typedef pthread_mutex_t mm_mutex_t;

struct mm_thread_start {
    int (*fn)(void*);
    void *arg;
};

static void* mm_thread_trampoline(void *p)
{
    struct mm_thread_start s = *(struct mm_thread_start*)p;
    free(p);
    int r = s.fn(s.arg);
    return (void*)(intptr_t)r;
}

static inline int mm_thread_create(mm_thread_t *t, int (*fn)(void*), void *arg)
{
    struct mm_thread_start *s = malloc(sizeof(*s));
    if (!s) return -1;

    s->fn = fn;
    s->arg = arg;

    return pthread_create(t,NULL,mm_thread_trampoline,s);
}

static inline int mm_thread_join(mm_thread_t t, int *result)
{
    void *ret;
    int r = pthread_join(t,&ret);

    if (result)
        *result = (int)(intptr_t)ret;

    return r;
}

static inline void mm_mutex_init(mm_mutex_t *m)
{
    pthread_mutex_init(m,NULL);
}

static inline void mm_mutex_lock(mm_mutex_t *m)
{
    pthread_mutex_lock(m);
}

static inline void mm_mutex_unlock(mm_mutex_t *m)
{
    pthread_mutex_unlock(m);
}

static inline void mm_mutex_destroy(mm_mutex_t *m)
{
    pthread_mutex_destroy(m);
}

static inline void mm_sleep_ms(uint32_t ms)
{
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts,NULL);
}

/* ============================================================
   C11 threads fallback
   ============================================================ */

#else

#include <threads.h>

typedef thrd_t mm_thread_t;
typedef mtx_t  mm_mutex_t;

static inline int mm_thread_create(mm_thread_t *t, int (*fn)(void*), void *arg)
{
    return thrd_create(t, fn, arg);
}

static inline int mm_thread_join(mm_thread_t t, int *result)
{
    return thrd_join(t, result);
}

static inline void mm_mutex_init(mm_mutex_t *m)
{
    mtx_init(m, mtx_plain);
}

static inline void mm_mutex_lock(mm_mutex_t *m)
{
    mtx_lock(m);
}

static inline void mm_mutex_unlock(mm_mutex_t *m)
{
    mtx_unlock(m);
}

static inline void mm_mutex_destroy(mm_mutex_t *m)
{
    mtx_destroy(m);
}

static inline void mm_sleep_ms(uint32_t ms)
{
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    thrd_sleep(&ts,NULL);
}

#endif

#endif
