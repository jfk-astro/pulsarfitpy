#include <python3.11/Python.h>
#include "pulsars.h"

static PyObject* py_add_numbers(PyObject* self, PyObject* args) {
    double a, b;

    if (!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }

    return Py_BuildValue("d", add_numbers(a, b));
}

static PyObject* py_multiply_numbers(PyObject* self, PyObject* args) {
    double a, b;

    if (!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }

    return Py_BuildValue("d", multiply_numbers(a, b));
}

static PyObject* py_divide_numbers(PyObject* self, PyObject* args) {
    double a, b;

    if (!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }

    double result = divide_numbers(a, b);

    if(result) 
        return Py_BuildValue("d", result);
    else
        return NULL;
}

static PyMethodDef PulsarLibMethods[] = {
    {"add_numbers", py_add_numbers, METH_VARARGS, "Add 2 numbers."},
    {"multiply_numbers", py_multiply_numbers, METH_VARARGS, "Multiply 2 numbers."},
    {"divide_numbers", py_divide_numbers, METH_VARARGS, "Divide 2 numbers."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef PulsarModule = {
    PyModuleDef_HEAD_INIT,
    "pulsar",
    "C library for approximations",
    -1,
    PulsarLibMethods
};

PyMODINIT_FUNC PyInit_pulsars(void) {
    return PyModule_Create(&PulsarModule);
}