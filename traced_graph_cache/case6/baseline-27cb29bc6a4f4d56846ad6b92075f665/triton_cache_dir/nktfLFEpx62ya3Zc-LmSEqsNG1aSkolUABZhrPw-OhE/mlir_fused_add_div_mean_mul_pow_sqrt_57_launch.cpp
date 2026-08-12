
#include <cpp_common.h>
#include <stdbool.h>
#include <string>

typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(obj));
    return ptr_info;
  }
  if (obj == Py_None) {
    // valid nullptr
    return ptr_info;
  }
  PyObject *ptr = PyObject_GetAttrString(obj, "data_ptr");
  if(ptr){
    PyObject *empty_tuple = PyTuple_New(0);
    PyObject *ret = PyObject_Call(ptr, empty_tuple, NULL);
    Py_DECREF(empty_tuple);
    Py_DECREF(ptr);
    if (!PyLong_Check(ret)) {
      PyErr_SetString(PyExc_TypeError, "data_ptr method of Pointer object must return 64-bit int");
      ptr_info.valid = false;
      return ptr_info;
    }
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsUnsignedLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either uint64 or have data_ptr method");
  return ptr_info;
}

static void _launch(const void* func, rtStream_t stream, int gridX, void* arg0, void* arg1, void* arg2) {
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  auto launch_call = [func, gridX, stream,  arg0,  arg1,  arg2]() {
    struct __attribute__((packed)) {
      void* arg0 __attribute__((aligned(8))); void* arg1 __attribute__((aligned(8))); void* arg2 __attribute__((aligned(8)));
    } args = {
      static_cast<void*>(arg0), static_cast<void*>(arg1), static_cast<void*>(arg2)
    };

    rtError_t ret = common_launch(const_cast<char*>("mlir_fused_add_div_mean_mul_pow_sqrt_57"), func, gridX, static_cast<void *>(&args), sizeof(args), stream);
    return ret;
  };
  opcommand_call("mlir_fused_add_div_mean_mul_pow_sqrt_57", launch_call);
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX;
  rtStream_t stream;
  const void *function;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *metadata = NULL;
  PyObject* _arg0;  PyObject* _arg1;  PyObject* _arg2; 
  if(!PyArg_ParseTuple(
      args, "iKKOOOOOO",
      &gridX, &stream, &function,
      &launch_enter_hook, &launch_exit_hook, &metadata
      , &_arg0, &_arg1, &_arg2
      )
    ) {
    return NULL;
  }

  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {
    return NULL;
  }

  // raise exception asap
  DevicePtrInfo ptr_info0 = getPointer(_arg0, 0); if (!ptr_info0.valid) return NULL;; DevicePtrInfo ptr_info1 = getPointer(_arg1, 1); if (!ptr_info1.valid) return NULL;; DevicePtrInfo ptr_info2 = getPointer(_arg2, 2); if (!ptr_info2.valid) return NULL;;

  _launch(function, stream, gridX, ptr_info0.dev_ptr, ptr_info1.dev_ptr, ptr_info2.dev_ptr);

  if (PyErr_Occurred()) {
    return NULL;
  }
  if (launch_exit_hook != Py_None && !PyObject_CallObject(launch_exit_hook, args)) {
    return NULL;
  }

  // return None
  Py_INCREF(Py_None);
  return Py_None;
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "__launcher",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit___launcher(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
