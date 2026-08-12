
#include <cpp_common.h>
#include <stdbool.h>
#include <string>
#include <dlfcn.h>
#include <iostream>

typedef struct _DevicePtrInfo {
  void *dev_ptr;
  bool valid;
} DevicePtrInfo;

static inline DevicePtrInfo getPointer(PyObject *obj, int idx) {
  DevicePtrInfo ptr_info;
  ptr_info.dev_ptr = 0;
  ptr_info.valid = true;
  if (PyLong_Check(obj)) {
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsLongLong(obj));
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
    ptr_info.dev_ptr = reinterpret_cast<void *>(PyLong_AsLongLong(ret));
    if(!ptr_info.dev_ptr)
      return ptr_info;
    Py_DECREF(ret);
    return ptr_info;
  }
  PyErr_SetString(PyExc_TypeError, "Pointer argument must be either int64 or have data_ptr method");
  return ptr_info;
}

static void _launch(void* func, void* tiling_func, int64_t tiling_size, void* arg_tiling_host, void* arg_tiling_device, rtStream_t stream, int gridX, void* arg0, void* arg_allocate0, void* offset0, void* sizes0_0, void* sizes0_1, void* sizes0_2, void* strides0_0, void* strides0_1, void* strides0_2, void* arg1, void* arg_allocate1, void* offset1, void* sizes1_0, void* sizes1_1, void* sizes1_2, void* strides1_0, void* strides1_1, void* strides1_2, void* arg2, void* arg3, void* arg4, void* arg5, void* arg_allocate5, void* offset5, void* sizes5_0, void* strides5_0, void* arg6, void* arg_allocate6, void* offset6, void* sizes6_0, void* strides6_0, void* arg7, void* arg_allocate7, void* offset7, void* sizes7_0, void* sizes7_1, void* sizes7_2, void* strides7_0, void* strides7_1, void* strides7_2) {
  // only 1D parallelization is supported for NPU
  // Pointer type becomes flattend 1-D Memref tuple: base_ptr, data_ptr, offset, shape, stride
  // base_ptr offset shape and stride are not used, arbitrarily set for now
  
  if (tiling_size == 0) {
    auto launch_call = [func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, stream, arg0, arg_allocate0, offset0, sizes0_0, sizes0_1, sizes0_2, strides0_0, strides0_1, strides0_2, arg1, arg_allocate1, offset1, sizes1_0, sizes1_1, sizes1_2, strides1_0, strides1_1, strides1_2, arg2, arg3, arg4, arg5, arg_allocate5, offset5, sizes5_0, strides5_0, arg6, arg_allocate6, offset6, sizes6_0, strides6_0, arg7, arg_allocate7, offset7, sizes7_0, sizes7_1, sizes7_2, strides7_0, strides7_1, strides7_2]() {
      struct __attribute__((packed)) {
      
      void* arg0 __attribute__((aligned(8))); void* arg_allocate0 __attribute__((aligned(8))); void* offset0 __attribute__((aligned(8))); void* sizes0_0 __attribute__((aligned(8))); void* sizes0_1 __attribute__((aligned(8))); void* sizes0_2 __attribute__((aligned(8))); void* strides0_0 __attribute__((aligned(8))); void* strides0_1 __attribute__((aligned(8))); void* strides0_2 __attribute__((aligned(8))); void* arg1 __attribute__((aligned(8))); void* arg_allocate1 __attribute__((aligned(8))); void* offset1 __attribute__((aligned(8))); void* sizes1_0 __attribute__((aligned(8))); void* sizes1_1 __attribute__((aligned(8))); void* sizes1_2 __attribute__((aligned(8))); void* strides1_0 __attribute__((aligned(8))); void* strides1_1 __attribute__((aligned(8))); void* strides1_2 __attribute__((aligned(8))); void* arg2 __attribute__((aligned(8)));  void* arg3 __attribute__((aligned(8)));  void* arg4 __attribute__((aligned(8)));  void* arg5 __attribute__((aligned(8))); void* arg_allocate5 __attribute__((aligned(8))); void* offset5 __attribute__((aligned(8))); void* sizes5_0 __attribute__((aligned(8))); void* strides5_0 __attribute__((aligned(8))); void* arg6 __attribute__((aligned(8))); void* arg_allocate6 __attribute__((aligned(8))); void* offset6 __attribute__((aligned(8))); void* sizes6_0 __attribute__((aligned(8))); void* strides6_0 __attribute__((aligned(8))); void* arg7 __attribute__((aligned(8))); void* arg_allocate7 __attribute__((aligned(8))); void* offset7 __attribute__((aligned(8))); void* sizes7_0 __attribute__((aligned(8))); void* sizes7_1 __attribute__((aligned(8))); void* sizes7_2 __attribute__((aligned(8))); void* strides7_0 __attribute__((aligned(8))); void* strides7_1 __attribute__((aligned(8))); void* strides7_2 __attribute__((aligned(8)));

      } args = {
      static_cast<void*>(arg0), static_cast<void*>(arg_allocate0), static_cast<void*>(offset0), static_cast<void*>(sizes0_0), static_cast<void*>(sizes0_1), static_cast<void*>(sizes0_2), static_cast<void*>(strides0_0), static_cast<void*>(strides0_1), static_cast<void*>(strides0_2), static_cast<void*>(arg1), static_cast<void*>(arg_allocate1), static_cast<void*>(offset1), static_cast<void*>(sizes1_0), static_cast<void*>(sizes1_1), static_cast<void*>(sizes1_2), static_cast<void*>(strides1_0), static_cast<void*>(strides1_1), static_cast<void*>(strides1_2), static_cast<void*>(arg2), static_cast<void*>(arg3), static_cast<void*>(arg4), static_cast<void*>(arg5), static_cast<void*>(arg_allocate5), static_cast<void*>(offset5), static_cast<void*>(sizes5_0), static_cast<void*>(strides5_0), static_cast<void*>(arg6), static_cast<void*>(arg_allocate6), static_cast<void*>(offset6), static_cast<void*>(sizes6_0), static_cast<void*>(strides6_0), static_cast<void*>(arg7), static_cast<void*>(arg_allocate7), static_cast<void*>(offset7), static_cast<void*>(sizes7_0), static_cast<void*>(sizes7_1), static_cast<void*>(sizes7_2), static_cast<void*>(strides7_0), static_cast<void*>(strides7_1), static_cast<void*>(strides7_2)

      };
      
      rtError_t ret = common_launch_dyn(const_cast<char*>("mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_26"), func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, static_cast<void *>(&args), sizeof(args), stream);
      return ret;
    };
    opcommand_call("mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_26", launch_call);
  } else {
    int64_t __attribute__((aligned(8))) key_tiling;
    // void* arg_tiling_host = nullptr;
    void* offset_tiling = 0;
    void* sizes_tiling = (void*)(tiling_size / sizeof(int64_t));
    void* strides_tiling = (void*)1;
    auto launch_call = [func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, stream, arg0, arg_allocate0, offset0, sizes0_0, sizes0_1, sizes0_2, strides0_0, strides0_1, strides0_2, arg1, arg_allocate1, offset1, sizes1_0, sizes1_1, sizes1_2, strides1_0, strides1_1, strides1_2, arg2, arg3, arg4, arg5, arg_allocate5, offset5, sizes5_0, strides5_0, arg6, arg_allocate6, offset6, sizes6_0, strides6_0, arg7, arg_allocate7, offset7, sizes7_0, sizes7_1, sizes7_2, strides7_0, strides7_1, strides7_2, key_tiling, offset_tiling, sizes_tiling, strides_tiling]() {
      struct __attribute__((packed)) {
      
      void* arg0 __attribute__((aligned(8))); void* arg_allocate0 __attribute__((aligned(8))); void* offset0 __attribute__((aligned(8))); void* sizes0_0 __attribute__((aligned(8))); void* sizes0_1 __attribute__((aligned(8))); void* sizes0_2 __attribute__((aligned(8))); void* strides0_0 __attribute__((aligned(8))); void* strides0_1 __attribute__((aligned(8))); void* strides0_2 __attribute__((aligned(8))); void* arg1 __attribute__((aligned(8))); void* arg_allocate1 __attribute__((aligned(8))); void* offset1 __attribute__((aligned(8))); void* sizes1_0 __attribute__((aligned(8))); void* sizes1_1 __attribute__((aligned(8))); void* sizes1_2 __attribute__((aligned(8))); void* strides1_0 __attribute__((aligned(8))); void* strides1_1 __attribute__((aligned(8))); void* strides1_2 __attribute__((aligned(8))); void* arg2 __attribute__((aligned(8)));  void* arg3 __attribute__((aligned(8)));  void* arg4 __attribute__((aligned(8)));  void* arg5 __attribute__((aligned(8))); void* arg_allocate5 __attribute__((aligned(8))); void* offset5 __attribute__((aligned(8))); void* sizes5_0 __attribute__((aligned(8))); void* strides5_0 __attribute__((aligned(8))); void* arg6 __attribute__((aligned(8))); void* arg_allocate6 __attribute__((aligned(8))); void* offset6 __attribute__((aligned(8))); void* sizes6_0 __attribute__((aligned(8))); void* strides6_0 __attribute__((aligned(8))); void* arg7 __attribute__((aligned(8))); void* arg_allocate7 __attribute__((aligned(8))); void* offset7 __attribute__((aligned(8))); void* sizes7_0 __attribute__((aligned(8))); void* sizes7_1 __attribute__((aligned(8))); void* sizes7_2 __attribute__((aligned(8))); void* strides7_0 __attribute__((aligned(8))); void* strides7_1 __attribute__((aligned(8))); void* strides7_2 __attribute__((aligned(8)));

      void* key_tiling __attribute__((aligned(8)));
      void* arg_tiling_host __attribute__((aligned(8)));
      void* arg_tiling_device __attribute__((aligned(8)));
      void* offset_tiling __attribute__((aligned(8)));
      void* sizes_tiling __attribute__((aligned(8)));
      void* strides_tiling __attribute__((aligned(8)));

      } args = {
      static_cast<void*>(arg0), static_cast<void*>(arg_allocate0), static_cast<void*>(offset0), static_cast<void*>(sizes0_0), static_cast<void*>(sizes0_1), static_cast<void*>(sizes0_2), static_cast<void*>(strides0_0), static_cast<void*>(strides0_1), static_cast<void*>(strides0_2), static_cast<void*>(arg1), static_cast<void*>(arg_allocate1), static_cast<void*>(offset1), static_cast<void*>(sizes1_0), static_cast<void*>(sizes1_1), static_cast<void*>(sizes1_2), static_cast<void*>(strides1_0), static_cast<void*>(strides1_1), static_cast<void*>(strides1_2), static_cast<void*>(arg2), static_cast<void*>(arg3), static_cast<void*>(arg4), static_cast<void*>(arg5), static_cast<void*>(arg_allocate5), static_cast<void*>(offset5), static_cast<void*>(sizes5_0), static_cast<void*>(strides5_0), static_cast<void*>(arg6), static_cast<void*>(arg_allocate6), static_cast<void*>(offset6), static_cast<void*>(sizes6_0), static_cast<void*>(strides6_0), static_cast<void*>(arg7), static_cast<void*>(arg_allocate7), static_cast<void*>(offset7), static_cast<void*>(sizes7_0), static_cast<void*>(sizes7_1), static_cast<void*>(sizes7_2), static_cast<void*>(strides7_0), static_cast<void*>(strides7_1), static_cast<void*>(strides7_2), 

      (void*)(&key_tiling), arg_tiling_host, arg_tiling_device, static_cast<void*>(offset_tiling), static_cast<void*>(sizes_tiling), static_cast<void*>(strides_tiling)
      };
      
      rtError_t ret = common_launch_dyn(const_cast<char*>("mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_26"), func, tiling_func, tiling_size, arg_tiling_host, arg_tiling_device, gridX, static_cast<void *>(&args), sizeof(args), stream);
      return ret;
    };
    opcommand_call("mlir_fused__to_copy_add_mean_mul_pow_rsqrt_sub_26", launch_call);
  }
}

static PyObject* launch(PyObject* self, PyObject* args) {
  int gridX;
  rtStream_t stream;
  PyObject *func;
  PyObject *tiling_func;
  int64_t tiling_size;
  PyObject *arg_tiling_host;
  PyObject *arg_tiling_device;
  PyObject *launch_enter_hook = NULL;
  PyObject *launch_exit_hook = NULL;
  PyObject *metadata = NULL;
  PyObject* _arg0; PyObject* _arg_allocate0; int64_t offset0; int64_t sizes0_0; int64_t sizes0_1; int64_t sizes0_2; int64_t strides0_0; int64_t strides0_1; int64_t strides0_2; PyObject* _arg1; PyObject* _arg_allocate1; int64_t offset1; int64_t sizes1_0; int64_t sizes1_1; int64_t sizes1_2; int64_t strides1_0; int64_t strides1_1; int64_t strides1_2; PyObject* _arg2; PyObject* _arg3; PyObject* _arg4; PyObject* _arg5; PyObject* _arg_allocate5; int64_t offset5; int64_t sizes5_0; int64_t strides5_0; PyObject* _arg6; PyObject* _arg_allocate6; int64_t offset6; int64_t sizes6_0; int64_t strides6_0; PyObject* _arg7; PyObject* _arg_allocate7; int64_t offset7; int64_t sizes7_0; int64_t sizes7_1; int64_t sizes7_2; int64_t strides7_0; int64_t strides7_1; int64_t strides7_2; 
  if(!PyArg_ParseTuple(
      args, "iKkkLOOOOOOOLLLLLLLOOLLLLLLLOOOOOLLLOOLLLOOLLLLLLL",
      &gridX, &stream, &func, &tiling_func, &tiling_size, &arg_tiling_host, &arg_tiling_device,
      &launch_enter_hook, &launch_exit_hook, &metadata
      , &_arg0, &_arg_allocate0, &offset0, &sizes0_0, &sizes0_1, &sizes0_2, &strides0_0, &strides0_1, &strides0_2, &_arg1, &_arg_allocate1, &offset1, &sizes1_0, &sizes1_1, &sizes1_2, &strides1_0, &strides1_1, &strides1_2, &_arg2, &_arg3, &_arg4, &_arg5, &_arg_allocate5, &offset5, &sizes5_0, &strides5_0, &_arg6, &_arg_allocate6, &offset6, &sizes6_0, &strides6_0, &_arg7, &_arg_allocate7, &offset7, &sizes7_0, &sizes7_1, &sizes7_2, &strides7_0, &strides7_1, &strides7_2
      )
    ) {
    return NULL;
  }


  if (launch_enter_hook != Py_None && !PyObject_CallObject(launch_enter_hook, args)) {
    return NULL;
  }

  // raise exception asap
  DevicePtrInfo ptr_info0 = getPointer(_arg0, 0); if (!ptr_info0.valid) return NULL; DevicePtrInfo ptr_info1 = getPointer(_arg1, 1); if (!ptr_info1.valid) return NULL; DevicePtrInfo ptr_info2 = getPointer(_arg2, 2); if (!ptr_info2.valid) return NULL; DevicePtrInfo ptr_info3 = getPointer(_arg3, 3); if (!ptr_info3.valid) return NULL; DevicePtrInfo ptr_info4 = getPointer(_arg4, 4); if (!ptr_info4.valid) return NULL; DevicePtrInfo ptr_info5 = getPointer(_arg5, 5); if (!ptr_info5.valid) return NULL; DevicePtrInfo ptr_info6 = getPointer(_arg6, 6); if (!ptr_info6.valid) return NULL; DevicePtrInfo ptr_info7 = getPointer(_arg7, 7); if (!ptr_info7.valid) return NULL; 
  DevicePtrInfo ptr_allocate_info0 = getPointer(_arg_allocate0, 0); if (!ptr_allocate_info0.valid) return NULL; DevicePtrInfo ptr_allocate_info1 = getPointer(_arg_allocate1, 1); if (!ptr_allocate_info1.valid) return NULL; ; ; ; DevicePtrInfo ptr_allocate_info5 = getPointer(_arg_allocate5, 5); if (!ptr_allocate_info5.valid) return NULL; DevicePtrInfo ptr_allocate_info6 = getPointer(_arg_allocate6, 6); if (!ptr_allocate_info6.valid) return NULL; DevicePtrInfo ptr_allocate_info7 = getPointer(_arg_allocate7, 7); if (!ptr_allocate_info7.valid) return NULL; 

  DevicePtrInfo arg_tiling_host_ptr = getPointer(arg_tiling_host, 0);
  DevicePtrInfo arg_tiling_device_ptr = getPointer(arg_tiling_device, 0);

  _launch(reinterpret_cast<void *>(func), reinterpret_cast<void *>(tiling_func), tiling_size, arg_tiling_host_ptr.dev_ptr, arg_tiling_device_ptr.dev_ptr, stream, gridX, ptr_info0.dev_ptr, ptr_allocate_info0.dev_ptr, reinterpret_cast<void *>(offset0), reinterpret_cast<void *>(sizes0_0), reinterpret_cast<void *>(sizes0_1), reinterpret_cast<void *>(sizes0_2), reinterpret_cast<void *>(strides0_0), reinterpret_cast<void *>(strides0_1), reinterpret_cast<void *>(strides0_2), ptr_info1.dev_ptr, ptr_allocate_info1.dev_ptr, reinterpret_cast<void *>(offset1), reinterpret_cast<void *>(sizes1_0), reinterpret_cast<void *>(sizes1_1), reinterpret_cast<void *>(sizes1_2), reinterpret_cast<void *>(strides1_0), reinterpret_cast<void *>(strides1_1), reinterpret_cast<void *>(strides1_2), ptr_info2.dev_ptr, ptr_info3.dev_ptr, ptr_info4.dev_ptr, ptr_info5.dev_ptr, ptr_allocate_info5.dev_ptr, reinterpret_cast<void *>(offset5), reinterpret_cast<void *>(sizes5_0), reinterpret_cast<void *>(strides5_0), ptr_info6.dev_ptr, ptr_allocate_info6.dev_ptr, reinterpret_cast<void *>(offset6), reinterpret_cast<void *>(sizes6_0), reinterpret_cast<void *>(strides6_0), ptr_info7.dev_ptr, ptr_allocate_info7.dev_ptr, reinterpret_cast<void *>(offset7), reinterpret_cast<void *>(sizes7_0), reinterpret_cast<void *>(sizes7_1), reinterpret_cast<void *>(sizes7_2), reinterpret_cast<void *>(strides7_0), reinterpret_cast<void *>(strides7_1), reinterpret_cast<void *>(strides7_2));

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

static PyObject* get_host_func_and_tiling_size(PyObject* self, PyObject* args) {
  const char *func_name;
  const char *tiling_func_name;
  const char *get_tiling_struct_size_func_name;
  const char *so_file;
  if(!PyArg_ParseTuple(
    args, "ssss", &func_name, &tiling_func_name, &get_tiling_struct_size_func_name, &so_file
    )
  ) {
    return NULL;
  }
  void *handle = dlopen(so_file, RTLD_LAZY);
  if (handle == NULL) {
      std::cout<<"handle == NULL!"<<std::endl;
      return Py_None;
  }

  typedef void (*mlir_func)(uint32_t, void*, void*, void*);
  mlir_func func = (mlir_func)dlsym(handle, func_name);
  if (func == NULL) {
      std::cout<<"Failed to load symbol for func: "<<dlerror()<<std::endl;
      dlclose(handle);
      return Py_None;
  }

  typedef int (*mlir_get_size_func)();
  mlir_get_size_func get_size_func = (mlir_get_size_func)dlsym(handle, get_tiling_struct_size_func_name);
  if (get_size_func == NULL) {
      std::cout<<"Failed to load symbol for get_size_func: "<<dlerror()<<std::endl;
      dlclose(handle);
      return Py_None;
  }

  int64_t tilingSize = get_size_func();
  tilingSize *= sizeof(int64_t);

  typedef int64_t (*mlir_tiling_func)(void*);
  mlir_tiling_func tiling_func = NULL;
  if (tilingSize != 0) {
    tiling_func = (mlir_tiling_func)dlsym(handle, tiling_func_name);
    if (tiling_func == NULL) {
      std::cout<<"Failed to load symbol for tiling_func: "<<dlerror()<<std::endl;
      dlclose(handle);
      return Py_None;
    }
  }

  return PyTuple_Pack(3, PyLong_FromUnsignedLong(reinterpret_cast<uintptr_t>(func)), PyLong_FromUnsignedLong(reinterpret_cast<uintptr_t>(tiling_func)), PyLong_FromLongLong(tilingSize));
}

static PyMethodDef ModuleMethods[] = {
  {"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"},
  {"get_host_func_and_tiling_size", get_host_func_and_tiling_size, METH_VARARGS, "Get host func from kernel.so"},
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
