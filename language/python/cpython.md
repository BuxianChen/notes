# cpython

基于 cpython==3.12.4, 主要结合 微信公众号古明地觉cpython系列 作为参考

```C
// Include/pytypedefs.h
typedef struct _object PyObject;

// Include/object.h

// 这个也就是 python 代码里的 object
// 简化后(主要是去掉了不必要的跨平台的宏, 仅考虑linux x86_64)
struct _object {
    union {
        Py_ssize_t ob_refcnt;  // Py_ssize_t 是指 64 位整数
        Py_UINT32_T ob_refcnt_split[2];
    };
    PyTypeObject *ob_type;
}

// 很多变长对象的实现会在 PyObject 的基础上增加一个属性
typedef struct {
    PyObject ob_base;
    Py_ssize_t ob_size;
} PyVarObject;

// 以下两个宏很常见
#define PyObject_HEAD PyObject ob_base;
#define PyObject_VAR_HEAD PyVarObject ob_base;

// 例如, python 中的浮点数的实现是用 double 来实现的
// Include/cpython/floatobject.h
typedef struct {
    PyObject_HEAD
    double ob_fval;
} PyFloatObject;
```

然后再看 `PyTypeObject`:

```C
// Include/pytypedefs.h
typedef struct _typeobject PyTypeObject;

// Include/cpython/object.h
struct _typeobject {
    PyObject_VAR_HEAD
    const char *tp_name;
    // ...
}
```

宏

```C
// _pyObject_EXTRA_INIT 宏可忽略
#define PyObject_HEAD_INIT(type)  \
{                                 \
    _pyObject_EXTRA_INIT          \
    { _Py_IMMORTAL_REFCNT },      \
    (type)                        \
},

#define PyVarObject_HEAD_INIT(type, size) \
{                                         \
    PyObject_HEAD_INIT(type)              \
    (size)                                \
},

// python 中的 int 类型对象
PyTypeObject PyLong_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0),
    "int",
    // ...
};
```

将宏展开后, PyTypeObject 实例化的第一个参数是 PyVarObject 类型的, 而 PyVarObject 内部包含 `PyObject ob_base;` 和 `Py_ssize_t ob_size;` 两个属性, 在 int 类型中, 后者是 0, 前者应该包含一个 union 类型的引用计数是 `_Py_IMMORTAL_REFCNT`, 以及一个 `*PyTypeObject`, 这里是 `&PyType_Type`. 

```C
PyTypeObject PyLong_Type = {
    {
        {
            { _Py_IMMORTAL_REFCNT },
            (&PyType_Type)
        },
        (0)
    },
    "int",
    // ...
};
```

那么 `PyType_Type` 是什么呢? 以下定义似乎在定义 PyType_Type 时使用了 PyType_Type, 这是什么原因呢

```C
// Objects/typeobject.c
// PyType_Type 是一个全局变量(静态变量)
PyTypeObject PyType_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0),
    "type",
    // ...
}
```

解释: 这里用的是静态全局变量初始化的特性: 先分配了 `MyType_Type` 这个实例的地址, 所以在实例化时能使用其自身的地址进行实例化

```C
#include <stdio.h>

// 定义一个结构体，包含一个指向自身类型的指针
typedef struct MyType {
    const char *name;
    struct MyType *type_ptr;
} MyType;

// 初始化一个全局变量，指向自身
MyType MyType_Type = {
    "MyType",        // name 字段
    &MyType_Type     // type_ptr 字段指向自己
};

int main() {
    // 打印 name 和 type_ptr 指向的结构体的 name
    printf("Type name: %s\n", MyType_Type.name);
    printf("Type name: %p\n", MyType_Type.type_ptr);
    printf("Type pointer name: %s\n", MyType_Type.type_ptr->name);
    printf("Type pointer name: %s\n", MyType_Type.type_ptr->type_ptr->name);
    printf("Type pointer name: %s\n", MyType_Type.type_ptr->type_ptr->type_ptr->name);

    return 0;
}

// 输出:
// Type name: MyType
// Type name: 0x55ac9fd46010
// Type pointer name: MyType
// Type pointer name: MyType
// Type pointer name: MyType
```
