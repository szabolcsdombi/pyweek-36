#include <Python.h>
#include <structmember.h>

extern int webapp_key_pressed(int key);
extern int webapp_key_down(int key);
extern void webapp_mouse(int * mouse);
extern int webapp_load_audio(short * data, int size);
extern void webapp_play_audio(int audio);
extern void webapp_reset_audio();
extern void webapp_save_score(int score);
extern int webapp_load_score();
extern void webapp_exit();

static PyObject * meth_key_pressed(PyObject * self, PyObject * arg) {
    PyObject * res = webapp_key_pressed(PyLong_AsLong(arg)) ? Py_True : Py_False;
    Py_INCREF(res);
    return res;
}

static PyObject * meth_key_down(PyObject * self, PyObject * arg) {
    PyObject * res = webapp_key_down(PyLong_AsLong(arg)) ? Py_True : Py_False;
    Py_INCREF(res);
    return res;
}

static PyObject * meth_mouse(PyObject * self, PyObject * args) {
    int mouse[2];
    webapp_mouse(mouse);
    return Py_BuildValue("(ii)", mouse[0], mouse[1]);
}

static PyObject * meth_load_audio(PyObject * self, PyObject * arg) {
    return PyLong_FromLong(webapp_load_audio((short *)PyBytes_AsString(arg), (int)PyBytes_Size(arg) / 2));
}

static PyObject * meth_play_audio(PyObject * self, PyObject * arg) {
    webapp_play_audio(PyLong_AsLong(arg));
    Py_RETURN_NONE;
}

static PyObject * meth_reset_audio(PyObject * self, PyObject * args) {
    webapp_reset_audio();
    Py_RETURN_NONE;
}

static PyObject * meth_save_score(PyObject * self, PyObject * arg) {
    webapp_save_score(PyLong_AsLong(arg));
    Py_RETURN_NONE;
}

static PyObject * meth_load_score(PyObject * self, PyObject * args) {
    return PyLong_FromLong(webapp_load_score());
}

static PyObject * meth_exit(PyObject * self, PyObject * args) {
    webapp_exit();
    Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"key_pressed", (PyCFunction)meth_key_pressed, METH_O, NULL},
    {"key_down", (PyCFunction)meth_key_down, METH_O, NULL},
    {"mouse", (PyCFunction)meth_mouse, METH_NOARGS, NULL},
    {"load_audio", (PyCFunction)meth_load_audio, METH_O, NULL},
    {"play_audio", (PyCFunction)meth_play_audio, METH_O, NULL},
    {"reset_audio", (PyCFunction)meth_reset_audio, METH_NOARGS, NULL},
    {"save_score", (PyCFunction)meth_save_score, METH_O, NULL},
    {"load_score", (PyCFunction)meth_load_score, METH_NOARGS, NULL},
    {"exit", (PyCFunction)meth_exit, METH_NOARGS, NULL},
    {NULL},
};

static PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "webapp", NULL, -1, module_methods};

extern PyObject * PyInit_webapp() {
    PyObject * module = PyModule_Create(&module_def);
    return module;
}
