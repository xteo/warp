/*
 * CPython C extension for fast CUDA kernel launch.
 * Eliminates ctypes overhead in the hot path by calling wp_cuda_launch_kernel
 * directly through a resolved function pointer.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <dlfcn.h>

/* Function pointer type matching wp_cuda_launch_kernel signature */
typedef size_t (*wp_cuda_launch_kernel_fn)(
    void* context,
    void* kernel,
    size_t dim,
    int max_blocks,
    int block_dim,
    int shared_memory_bytes,
    void** args,
    void* stream
);

typedef int (*wp_cuda_stream_is_capturing_fn)(void* stream);
typedef unsigned long long (*wp_cuda_stream_get_capture_id_fn)(void* stream);

/* Module state */
static wp_cuda_launch_kernel_fn g_launch_fn = NULL;
static wp_cuda_stream_is_capturing_fn g_is_capturing_fn = NULL;
static wp_cuda_stream_get_capture_id_fn g_get_capture_id_fn = NULL;

/*
 * init_from_dll(dll_path: str) -> None
 * Resolve wp_cuda_launch_kernel from the warp shared library.
 */
static PyObject*
fast_launch_init(PyObject* self, PyObject* args)
{
    const char* dll_path;
    if (!PyArg_ParseTuple(args, "s", &dll_path))
        return NULL;

    void* handle = dlopen(dll_path, RTLD_NOW | RTLD_NOLOAD);
    if (!handle) {
        /* Try loading it fresh */
        handle = dlopen(dll_path, RTLD_NOW);
    }
    if (!handle) {
        PyErr_Format(PyExc_RuntimeError, "Cannot open %s: %s", dll_path, dlerror());
        return NULL;
    }

    g_launch_fn = (wp_cuda_launch_kernel_fn)dlsym(handle, "wp_cuda_launch_kernel");
    if (!g_launch_fn) {
        PyErr_Format(PyExc_RuntimeError, "Cannot find wp_cuda_launch_kernel in %s", dll_path);
        dlclose(handle);
        return NULL;
    }

    g_is_capturing_fn = (wp_cuda_stream_is_capturing_fn)dlsym(handle, "wp_cuda_stream_is_capturing");
    g_get_capture_id_fn = (wp_cuda_stream_get_capture_id_fn)dlsym(handle, "wp_cuda_stream_get_capture_id");

    /* Don't dlclose — we need the handle to remain valid */
    Py_RETURN_NONE;
}

/*
 * launch(context: int, kernel: int, dim: int, max_blocks: int, block_dim: int,
 *        smem_bytes: int, kparams: int, stream: int) -> int
 *
 * All arguments are raw pointer addresses passed as Python ints.
 * This is the absolute minimum overhead path: parse 8 ints, call the C function.
 */
static PyObject*
fast_launch_launch(PyObject* self, PyObject* const* fastargs, Py_ssize_t nargs)
{
    if (nargs != 8) {
        PyErr_SetString(PyExc_TypeError, "launch() requires exactly 8 arguments");
        return NULL;
    }

    void* context = PyLong_AsVoidPtr(fastargs[0]);
    void* kernel = PyLong_AsVoidPtr(fastargs[1]);
    size_t dim = (size_t)PyLong_AsUnsignedLongLong(fastargs[2]);
    int max_blocks = (int)PyLong_AsLong(fastargs[3]);
    int block_dim = (int)PyLong_AsLong(fastargs[4]);
    int smem_bytes = (int)PyLong_AsLong(fastargs[5]);
    void** kparams = (void**)PyLong_AsVoidPtr(fastargs[6]);
    void* stream = PyLong_AsVoidPtr(fastargs[7]);

    if (PyErr_Occurred())
        return NULL;

    size_t result = g_launch_fn(context, kernel, dim, max_blocks, block_dim, smem_bytes, kparams, stream);

    return PyLong_FromSize_t(result);
}

static PyMethodDef FastLaunchMethods[] = {
    {"init_from_dll", fast_launch_init, METH_VARARGS,
     "Initialize the fast launch extension from a warp shared library path."},
    {"launch", (PyCFunction)fast_launch_launch, METH_FASTCALL,
     "Launch a CUDA kernel with minimal overhead."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastlaunchmodule = {
    PyModuleDef_HEAD_INIT,
    "_fast_launch",
    "Fast CUDA kernel launch extension for Warp",
    -1,
    FastLaunchMethods
};

PyMODINIT_FUNC
PyInit__fast_launch(void)
{
    return PyModule_Create(&fastlaunchmodule);
}
