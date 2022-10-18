# Python 高阶与最佳实践

## 1. 装饰器

内置装饰器

```python
class A:
    b = 0
    def __init__(self):
        self.a = 1
    @classmethod
    def foo(cls, a):
        print(a)
    @classmethod
    def bar(cls, a):
        cls.b += a
        print(cls.b)
A.bar(3)
A.bar(2)
```

自定义装饰器

例子1: 

```python
def my_decorator(func):
    def wrapper_function(*args, **kwargs):
        print("*"*10)
        res = func(*args,  **kwargs)
        print("*"*10)
        return res
    return wrapper_function
@my_decorator
def foo(a):
    return a
# 相当于foo=my_decorator(foo)
x = foo(1)
```


例子2:
```python
from functools import wraps
def node_func(name):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if name == "A":  # in self.nodes_df.columns:
                return 1  # dict(self.nodes_df[name])
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorate
# 等价于：foo1 = node_func("A")(foo1)
@node_func("A")
def foo1(a):
    return "a"

@node_func("B")
def bar1(a):
    return "a"
```

**`property` 装饰器**

例子来源于 [Python 官方文档](https://docs.python.org/3/library/functions.html#property)。

```python
class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x
```

根据前面所述，装饰器只是一个语法糖。property 函数的特征标（signature）如下：

```
property(fget=None, fset=None, fdel=None, doc=None) -> object
```

前一段代码等价于这种直接使用 `property` 函数的做法：

```python
class C:
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x

    def setx(self, value):
        self._x = value

    def delx(self):
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")
```

备注：property 本质上是一个 Descriptor，参见后面。

## 2. 魔术方法与内置函数

### 2.0 Python 官方文档

- 官方文档主目录：https://docs.python.org/3/
- 对 Python 语言的一般性描述：https://docs.python.org/3/reference/index.html
  - 数据模型：https://docs.python.org/3/reference/datamodel.html
- Python 标准库：https://docs.python.org/3/library/index.html
  - build-in functions（官方建议优先阅读此章节）：https://docs.python.org/3/library/functions.html
  - build-in types：https://docs.python.org/3/library/stdtypes.html
- Python HOWTOs（深入介绍一些主题，可以认为是官方博客）：https://docs.python.org/3/howto/index.html
  - Descriptor HowTo Guide：https://docs.python.org/3/howto/descriptor.html

### 2.1 object 类

```python
>>> dir(object())
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
```

```python
class Basic:
    pass
basic = Basic()
set(dir(basic)) - set(dir(object))
# {'__dict__', '__module__', '__weakref__'}
```

**`__module__`**

**`__weakref__`**

### 2.2 `__str__`、`__repr__` 特殊方法，str、repr 内置函数

**从设计理念上说：两者都是将对象输出，一般而言，`__str__` 遵循可读性原则，`__repr__` 遵循准确性原则。**

分别对应于内置方法 `str` 与 `repr`，二者在默认情况（不重写方法的情况下）下都会输出类似于 `<Classname object at 0x000001EA748D6DC8>` 的信息.

```python
>>> class Test:
...     def __init__(self):
...         self.a = 1
...     def __repr__(self): # 一般遵循准确性, 例如出现类似<class xxx>
...         return "__repr__"
...     def __str__(self): # 一般遵循可读性
...         return "__str__"
...
>>> test = Test()
>>> test
__repr__
>>> print(test) # print使用__str__
__str__
```

```python
>>> class Test1:
...     def __str__(self):
...         return "__str__"
...
>>> test1 = Test1()
>>> print(test1)  # print使用__str__
__str__
>>> test1
<__main__.Test1 object at 0x000001EA748D6DC8>
```

备注: 在 jupyter notebook 中, 对 `pandas` 的 `DataFrame` 使用 `print` 方法，打印出的结果不美观，但不用 `print` 却很美观，原因未知。

### 2.3 内置函数 vars 与 `__dict__` 属性

**从设计理念上说，`vars` 函数的作用是返回对象的属性名（不会包含方法及特殊属性）。`__dict__` 属性里保存着对象的属性名（不会包含方法以及特殊属性）。这里的特殊属性指的是 `__xxx__`。**

一般情况下，Python 中的对象都有默认的 `__dict__` 属性。而 `vars(obj)` 的作用就是获取对象 `obj` 的 `__dict__` 属性。关于 `vars` 函数的解释可以参考[官方文档](https://docs.python.org/3/library/functions.html#vars)，如下：

> Return the `__dict__` attribute for a **module, class, instance, or any other object with a `__dict__` attribute**.
>
> Objects such as modules and instances have an updateable `__dict__` attribute; however, other objects may have write restrictions on their `__dict__` attributes (for example, classes use a [`types.MappingProxyType`](https://docs.python.org/3/library/types.html#types.MappingProxyType) to prevent direct dictionary updates).
>
> Without an argument, `vars()` acts like [`locals()`](https://docs.python.org/3/library/functions.html#locals). Note, the locals dictionary is only useful for reads since updates to the locals dictionary are ignored.
>
> A `TypeError` exception is raised if an object is specified but it doesn’t have a `__dict__` attribute (for example, if its class defines the [`__slots__`](https://docs.python.org/3/reference/datamodel.html#object.__slots__) attribute).

```python
# vars(x)
x.__dict__  # 必须定义为一个字典
```

备注：object 类没有 `__dict__` 属性，但继承自 object 子类的对象会有一个默认的 `__dict__` 属性（有一个例外是当该类定义了类属性 `__slots__` 时，该类的对象就不会有 `__dict__` 属性）。

**`__dict__` 属性与 Python 的查找顺序（lookup chain）息息相关，详情见 Descriptor**。

### 2.4 `__slots__`属性

**从设计理念上说，`__slots__` 属性的作用是规定一个类只能有那些属性，防止类的实例随意地动态添加属性。**

可以定义类属性 `__slots__`（一个属性名列表），确保该类的实例不会添加 `__slots__` 以外的属性。一个副作用是定义了 `__slots__` 属性的类，其实例将不会拥有 `__dict__` 属性。具体用法如下：

```python
class A:
    __slots__ = ["a", "b"]
a = A()
a.a = 2
a.c = 3  # 报错
```

注意：假设类 `B` 继承自定义了 `__slots__` 的类 `A`，那么子类 `B` 的实例不会受到父类 `__slots__` 的限制。

### 2.5 内置函数 dir 与 `__dir__` 方法

**从设计理念上说：不同于 vars 与 `__dict__`，dir 方法倾向于给出全部信息：包括特殊方法名**

`dir` 函数返回的是一个标识符名列表，逻辑是：首先寻找 `__dir__` 函数的定义（object 类中有着默认的实现），若存在 `__dir__` 函数，则返回 `list(x.__dir__())`。备注：`__dir__` 函数必须定义为一个可迭代对象。

若该类没有自定义 `__dir__` 函数，则使用 object 类的实现逻辑，大略如下：

> If the object does not provide [`__dir__()`](https://docs.python.org/3/reference/datamodel.html#object.__dir__), the function tries its best to gather information from the object’s [`__dict__`](https://docs.python.org/3/library/stdtypes.html#object.__dict__) attribute, if defined, and from its type object. The resulting list is not necessarily complete, and may be inaccurate when the object has a custom [`__getattr__()`](https://docs.python.org/3/reference/datamodel.html#object.__getattr__).
>
> The default [`dir()`](https://docs.python.org/3/library/functions.html?highlight=dir#dir) mechanism behaves differently with different types of objects, as it attempts to produce the most relevant, rather than complete, information:
>
> - If the object is a module object, the list contains the names of the module’s attributes.
> - If the object is a type or class object, the list contains the names of its attributes, and recursively of the attributes of its bases.
> - Otherwise, the list contains the object’s attributes’ names, the names of its class’s attributes, and recursively of the attributes of its class’s base classes.
>
> ——https://docs.python.org/3/library/functions.html

备注：官方文档对默认的 `dir` 函数的实现逻辑有些含糊不清，只能简单理解为默认实现会去寻找 `__dict__` 属性，故暂不予以深究。这里留一个测试例子待后续研究：

例子

```python
class Test:
    __slots__ = ["a", "b", "c"]
    def __init__(self):
        self.a = 3
        self.b = 1
        # self._c = 2
        # self.__d = 3
        # self.__dict__ = {"a": 1}

    def __dir__(self):
        # return "abc"
        # return {"a": "dir_a"}
        print("Test: __dir__")
        return super().__dir__()
    
    def __getattribute__(self, name: str):
        print(f"Test: __getattribute__, args: {name}")
        return super().__getattribute__(name)
    
    def __getattr__(self, name):
        print(f"Test: __gatattr__, args: {name}")
        return "default"
        # return super().__getattr__(name) # object没有__getattr__方法
test = Test()
print(dir(test))
```

输出结果为：（`__getattribute__` 与 `__getattr__` 见下一部分，大体上是寻找了 `__dict__` 属性与 `__class__` 属性）

```
Test: __dir__
Test: __getattribute__, args: __dict__
Test: __gatattr__, args: __dict__
Test: __getattribute__, args: __class__
['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattr__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__slots__', '__str__', '__subclasshook__', 'a', 'b', 'c']
```

### 2.6 `__getattr__`、`__getattribute__` 特殊方法，`getattr` 内置函数

**从设计理念上说，这三者的作用是使用属性名获取属性值，也适用于方法**

**作用：`__getattribute__` 会拦截所有对属性的获取。**

首先内置函数 `getattr(object, name[, default])` 的功能等同于 `object.name`，例如：`getattr(a, "name")` 等价于 `a.name`。实现细节上，内置函数 `getattr` 会首先调用 `__getattribute__`，如果找不到该属性，则去调用 `__getattr__` 函数。

备注：object 类只有 `__getattribute__` 的定义，而没有 `__getattr__`。

备注：对于以双下划线开头的变量，编译时会对其名称进行修改：

```python
class A:
	class A:
    def __init__(self):
        self.__a = 1
a = A()
dir(a)  # 会显示 "_A__a"
vars(a)  # 会显示 "_A__a"
a._A__a  # ok
getattr(a, "_A__a")  # ok
```

备注：如果要自定义 `__getattribute__` 函数，最好在其内部调用 `object.__getattribute__(self, name)`。

### 2.7 `delattr` 内置方法、`__delattr__` 特殊方法、del 语句

**作用：`__delattr__` 会拦截所有对属性的删除。**

### 2.8 `setattr` 内置方法、`__setattr__` 特殊方法

**作用：`__setattr__` 会拦截所有对属性的赋值。**

### 2.9 Descriptor、`__get__`、`__set__`、`__del__`

参考： 

- [RealPython (Python-descriptors)](https://realpython.com/python-descriptors/)
- [Python 官方文档 (Howto-descriptor)](https://docs.python.org/3/howto/descriptor.html)
- [Python 官方文档 (library-build-in-functions)](https://docs.python.org/3/library/functions.html)
- [Python 官方文档 (reference-data-model)](https://docs.python.org/3/reference/datamodel.html)

**注：大多数情况下，无须使用 Descriptor**

#### 概念

按照如下要求实现了 `__get__`、`__set__`、`__delete__` 其中之一的类即满足 Descriptor 协议，称这样的类为 Descriptor（描述符） 。若没有实现 `__set__` 及 `__delete__` 方法，称为 **data descriptor**，否则称为 **non-data descriptor**。

```python
__get__(self, obj, type=None) -> object
__set__(self, obj, value) -> None
__delete__(self, obj) -> None
__set_name__(self, owner, name)
```

#### Descriptor 的作用

在 Python 的底层，`staticmethod()`、`property()`、`classmethod()`、`__slots__` 都是借助 Descriptor 实现的。



`def foo(self, *args)` 可以使用 `obj.foo(*args)` 进行调用也是使用 Descriptor 实现的。

> The starting point for descriptor invocation is a binding, `a.x`. How the arguments are assembled depends on `a`:
>
> - Direct Call
>
>   The simplest and least common call is when user code directly invokes a descriptor method: `x.__get__(a)`.
>
> - Instance Binding
>
>   If binding to an object instance, `a.x` is transformed into the call: `type(a).__dict__['x'].__get__(a, type(a))`.
>
> - Class Binding
>
>   If binding to a class, `A.x` is transformed into the call: `A.__dict__['x'].__get__(None, A)`.
>
> - Super Binding
>
>   If `a` is an instance of [`super`](https://docs.python.org/3/library/functions.html#super), then the binding `super(B, obj).m()` searches `obj.__class__.__mro__` for the base class `A` immediately preceding `B` and then invokes the descriptor with the call: `A.__dict__['m'].__get__(obj, obj.__class__)`.
>
> —— https://docs.python.org/3/reference/datamodel.html#invoking-descriptors

#### 查找顺序

完整的顺序如下，对于 `obj.x`，获得其值的查找顺序为：

- 首先寻找命名为 `x` 的 **data descriptor**。即如果在 `obj` 的类 `Obj` 定义里有如下形式：

  ```
  class Obj:
  	x = DescriptorTemplate()
  ```

  其中 `DescriptorTemplate` 中定义了 `__set__` 或 `__del__` 方法。

- 若上一条失败，在对象 `obj` 的 `__dict__` 属性中查找 `"x"`。

- 若上一条失败，寻找命名为 `x` 的 **non-data descriptor**。即如果在 `obj` 的类 `Obj` 定义里有如下形式：

  ```
  class Obj:
  	x = DescriptorTemplate()
  ```

  其中 `DescriptorTemplate` 中定义了 `__get__` 但没有定义 `__set__` 及 `__del__` 方法。

- 若上一条失败，则在 `obj` 类型的 `__dict__` 属性中查找，即 `type(obj).__dict__`。

- 若上一条失败，则在其父类中查找，即 `type(obj).__base__.__dict__`。

- 若上一条失败，则按照父类搜索顺序 `type(obj).__mro__`，对类祖先的 `__dict__` 属性依次查找。

- 若上一条失败，则得到 `AttributeError` 异常。

例子：

如果类没有定义 `__slot__` 属性及 `__getattr__` 方法，且 `__getattribute__`、`__delattr__`、`__setattr__` 这些方法都直接继承自 object 类，那么 `__dict__` 的构建将会是如下默认的方式：

```python
class Vehicle():
    can_fly = False
    number_of_weels = 0

class Car(Vehicle):
    number_of_weels = 4

    def __init__(self, color):
        self.color = color

def foo(self):
    print("foo")

my_car = Car("red")
print(my_car.__dict__)
print(type(my_car).__dict__)
my_car.bar = foo  # 注意这种情况下my_car.bar是一个unbound fuction, 关于这一点参见Descriptor
print(my_car.__dict__)
print(type(my_car).__dict__)
my_car.bar(my_car)
```

```python
{'color': 'red'}
{'__module__': '__main__', 'number_of_weels': 4, '__init__': <function Car.__init__ at 0x000001A3C7857040>, '__doc__': None}
{'color': 'red', 'bar': <function foo at 0x000001A3C76ED160>}
{'__module__': '__main__', 'number_of_weels': 4, '__init__': <function Car.__init__ at 0x000001A3C7857040>, '__doc__': None}
foo
```

查找顺序

```python
my_car = Car("red")
print(my_car.__dict__['color'])  # 等价于 mycar.color
print(type(my_car).__dict__['number_of_weels'])  # 等价于 mycar.number_of_wheels
print(type(my_car).__base__.__dict__['can_fly'])  # 等价于 mycar.can_fly
```

#### 使用 Descriptor

需实现下列函数，实现 `__get__`、`__set__`、`__delete__` 其中之一即可，`__set_name__` 为 Python 3.6 引入的新特性，可选。参照例子解释：

```python
__get__(self, obj, type=None) -> object
# self指的是Descriptor对象实例number, obj是self所依附的对象my_foo_object, type是Foo
__set__(self, obj, value) -> None
# self指的是Descriptor对象实例number, obj是self所依附的对象my_foo_object, value是3
__delete__(self, obj) -> None
# self指的是Descriptor对象实例number, obj是self所依附的对象my_foo_object
__set_name__(self, owner, name)
# self指的是Descriptor对象实例number, owner是Foo, name是"number"
```

例子

```python
class OneDigitNumericValue():
    def __set_name__(self, owner, name):
        # owner is Foo, name is number
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name) or 0

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = value

class Foo():
    number = OneDigitNumericValue()

my_foo_object = Foo()
my_second_foo_object = Foo()

my_foo_object.number = 3
print(my_foo_object.number)
print(my_second_foo_object.number)

my_third_foo_object = Foo()
print(my_third_foo_object.number)
```

#### 实用例子

**避免重复使用 `property`**

```python
class Values:
    def __init__(self):
        self._value1 = 0
        self._value2 = 0
        self._value3 = 0

    @property
    def value1(self):
        return self._value1

    @value1.setter
    def value1(self, value):
        self._value1 = value if value % 2 == 0 else 0

    @property
    def value2(self):
        return self._value2

    @value2.setter
    def value2(self, value):
        self._value2 = value if value % 2 == 0 else 0

    @property
    def value3(self):
        return self._value3

    @value3.setter
    def value3(self, value):
        self._value3 = value if value % 2 == 0 else 0

my_values = Values()
my_values.value1 = 1
my_values.value2 = 4
print(my_values.value1)
print(my_values.value2)
```

可以使用如下方法实现

```python
class EvenNumber:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None) -> object:
        return obj.__dict__.get(self.name) or 0

    def __set__(self, obj, value) -> None:
        obj.__dict__[self.name] = (value if value % 2 == 0 else 0)

class Values:
    value1 = EvenNumber()
    value2 = EvenNumber()
    value3 = EvenNumber()
    
my_values = Values()
my_values.value1 = 1
my_values.value2 = 4
print(my_values.value1)
print(my_values.value2)
```

### 2.10 pickle 与 `__setstate__`、`__getstate__` 方法

某些时候，一个对象无法进行序列化，则可以自定义 `__getstate__`，在进行序列化时，只序列化 `__setstate__` 的返回值。另外，可自定义 `__setstate__` 方法，在反序列化时，利用 `__getstate__` 的返回值将对象恢复。具体可参考[官方文档](https://docs.python.org/3/library/pickle.html)。

一个说明功能的例子：

```python
import pickle
class A:
    def __init__(self, a):
        self.a = a
    def __getstate__(self):
        return (self.a, self.a+1)
    def __setstate__(self, state):
        a, b = state
        print(a, b)
        self.a = "recover"
a = A(2)
with open("test.pkl", "wb") as fw:
    pickle.dump(a, fw)
with open("test.pkl", "rb") as fr:
    a = pickle.load(fr)
print(a.a)  # "recover"
```

更有意义的例子待补充

## 3. 继承

### MRO (Method Resolution Order) 与 C3 算法

Python 在产生多继承关系时，由于子类可能有多个或多层父类，因此方法的搜索顺序（MRO, Method Resolution Order）很重要，同时，搜索顺序也涉及到类的属性。对于属性或者变量的访问，按照 MRO 的顺序依次搜索，直到找到匹配的属性或变量为止。对于每个类，可以使用如下代码来获取 MRO ：

```python
C.mro()  # C 是一个类
# 或者：
C.__mro__
```

本部分参考 C3 算法[官方文档](https://www.python.org/download/releases/2.3/mro/)：

> unless you make strong use of multiple inheritance and you have non-trivial hierarchies, you don't need to understand the C3 algorithm, and you can easily skip this paper.

**一点历史与 MRO 应满足的性质**

在 Python 的历史上，曾出现了若干种 MRO 算法，自 Python 2.3 以后，使用 C3 算法，它满足两个性质（之前的算法违背了这两个性质，所以可能会引发隐蔽的 BUG）

- local precedence ordering：MRO 的结果里应该保证父类列表的相对顺序不变。例如：

  ```python
  class A(B, C, D): pass
  ```

  MRO(A) 序列必须为 `[A, ..., B, ..., C, ..., D, ...]` 这种形式。

- monotonicity（单调性）：如果 C 的 MRO 序列中 A 排在 B 的前面，那么对于任意继承自 C 的类 D，D 的 MRO 序列中 A 也排在 B 的前面

**C3 算法**

引入记号：

- 用 $$B_1B_2...B_n$$ 代表 $$[B_1,B_2,...,B_n]$$。用 $$C+B_1...B_n$$ 代表 $$CB_1,...B_n$$。即类 $$C$$ 的 MRO 序列为 $$L(C)$$
- 对于序列 $$B_1...B_n$$，$$B_1$$ 称为头，$$B_2...B_n$$ 称为尾

C3 算法描述为：

```
L[C(B1,...,Bn)] = C + merge(L[B1],...,L[Bn], B1B2...Bn)
```

其中 merge 的规则为：

递归调用 merge 操作：

记第一个序列中的头为 $$H$$，若 $H$ 不在其余任意序列的尾中，则将 $$H$$ 添加到 MRO 序列中，并对 merge 中的所有序列中删除 $$H$$，之后对剩余序列继续 merge 操作；否则对第二个序列的头进行上述操作，直至最后一个序列。若直到最后一个序列都无法进行删除操作，那么判定为继承关系不合法。

例子：

```python
O=object
class F(O): pass
class E(O): pass
class D(O): pass
class C(D, F): pass
class B(E, D): pass
class A(B, C): pass
```

```
L[O] = O
L[F(O)] = F + merge(L[O], O) = F + merge(O, O) = FO
L[E(O)] = EO
L[D(O)] = DO
L[C(D, F)] = C + merge(L(D), L(F), DF) = C + merge(DO, FO, DF)
           = CD + merge(O, FO, F)  # D 只在所有序列的头部出现
           = CDF + merge(O, O) # O 在第二个序列的尾部出现，因此接下来对 F 进行判断
           = CDFO
L[B(E, D)] = B + merge(EO, DO, ED) = BEDO
L[A(B, C)] = A + merge(BEDO, CDFO, BC)
           = AB + merge(EDO, CDFO, C)
           = ABE + merge(DO, CDFO, C)
           = ABEC + merge(DO, DFO)
           = ABECDFO
```

### `super` 函数

参考资料：[RealPython](https://realpython.com/python-super/)、《Python Cookbook (3ed)》chapter 8.7。



由于方法覆盖的特性，以方法为例，如果类的 MRO 顺序中有同名方法，那么处于 MRO 靠后类的同名方法将会被隐藏。因此如果需要调用父类被隐藏的方法，需要对 MRO 顺序进行调整。这就是 `super` 方法的作用。



`super` 函数有两种调用形式

- 两个参数的形式：super(cls, obj)。其中第一个参数为子类，obj 为子类对象（也可以是子类的子类对象，但基本不可能会这样去用）。

- 无参数形式：super()。推荐使用

```python
class A:
    def afoo(self):
        print("A::afoo")
class B(A):
    def afoo(self):x
        super().afoo()  # 等价于 super(B, self).afoo()
        print("B::afoo")
class C(B):
    def afoo(self):
        super(B, self).afoo()
        print("C::afoo")
C().afoo()  # 依次调用 A.afoo, C.afoo
B().afoo()  # 依次调用 A.afoo, B.afoo
```

super 实际上是一个类，但注意 `super()` 返回的不是父类对象，而是一个代理对象。

```python
class Base: def __init__(self): print("Base"); super().__init__()
class A(Base): def __init__(self): print("A"); super().__init__()
class B(Base): def __init__(self): print("B"); super().__init__()
class C(A, B): def __init__(self): print("C"); super().__init__()
C()
# 输出：
# C
# A
# B
# Base
```

上例为典型的菱形继承方式，使用 `super` 可以按照 MRO 顺序依次调用 `__init__` 函数一次。

备注：`super` 函数还有单参数的调用形式，参见 [stckoverflow](https://stackoverflow.com/questions/30190185/how-to-use-super-with-one-argument)（理解需要有许多前置知识）。

## 4. 元类

参考资料：[RealPython](https://realpython.com/python-metaclasses/)，[Python 官方文档](https://docs.python.org/3/reference/datamodel.html#metaclasses)，

类是用来构造实例的，因此类也可以被叫做实例工厂；同样地，也有构造类的东西，被称为**元类**。实际上每个类都需要用元类来构造，默认的元类为 `type`。

```python
class A: pass
# 等同于
class A(object, metaclass=type): pass
```

### `type` 函数

Python 中, type 函数是一个特殊的函数，调用形式有两种：

- `type(obj)`：返回 obj 的类型
- `type()`



### `__new__` 函数与 `__init__` 函数（待补充）

### `abc` 模块

`abc` 模块最常见是搭配使用 `ABCMeta` 与 `abstractmethod`。其作用是让子类必须重写父类用 `abstractmethod` 装饰的方法，否则在创建子类对象时就会报错。[参考](https://riptutorial.com/python/example/23083/why-how-to-use-abcmeta-and--abstractmethod)

用法如下：

```python
from abc import ABCMeta, abstractmethod
class Base(metaclass=ABCMeta):
    @abstractmethod
    def foo(self):
        print("foo")
    @abstractmethod
    def bar(self):
        pass
class A(Base):
    def foo(self):
        print("A foo")
    def bar(self):
        print("A bar")
a = A()
super(A, a).foo()
a.foo()
a.bar()
```

注意：不设定 `metaclass=ABCMeta` 时，`abstractmethod` 不起作用，即不会强制子类继承。

使用 `ABCMeta` 与 `abstractmethod` 优于这种写法：

```python
class Base(metaclass=ABCMeta):
    def foo(self):
        print("a foo")
    def bar(self):
        raise NotImplementedError()
class A(Base):
    def foo(self):
        print("A foo")
a = A()
super(A, a).foo()
a.foo()
a.bar()  # 此时才会抛出异常
```

## 5. with语法\(含少量contextlib包的笔记\)

主要是为了理解pytorch以及tensorflow中各种with语句

主要[参考链接](https://www.geeksforgeeks.org/with-statement-in-python/)

### 5.1 读写文件的例子

首先厘清读写文件的一些细节

```python
# test01.py
file = open("record.txt", "w+")
file.write("Hello")  # 由于file没有调用close方法, 所以"Hello"未被写入
file = open("record.txt", "w+")
file.write("World")
file.close()  # 这一行是否有都是一样的, 大概是解释器自动调用了close
# 这个脚本最终只会写入"World"
```

以下三段代码中

* 代码1如果在write时报错, 那么文件无法被close, 有可能引发BUG
* 代码2保证文件会被close, 另外可以通过增加except语句, 使得可以处理各类异常
* 代码3则相对优雅, 并且与代码2功能一致, 即使write出错, close依旧会被调用

```python
# 1) without using with statement
file = open('file_path', 'w')
file.write('hello world !')
file.close()

# 2) without using with statement
file = open('file_path', 'w')
try:
    file.write('hello world')
finally:
    file.close()

# 3) using with statement
with open('file_path', 'w') as file:
    file.write('hello world !')
```

代码3是怎么做到的呢? 其实际上基本等效于

```python
foo = open("file_path", "w")
file = foo.__enter__()
try:
    file.write("hello world !")
finally:
    # 注意: 此处需要传递3个参数, 但一般不会是None
    foo.__exit__(None, None, None)
```

注意到一般情况下, 此处的foo与file是不一样的对象, 参见下节中关于`__enter__`方法的返回值. 但在文件读写的情形下, foo与file是相同的对象. 另外, `__exit__`函数有三个参数, 在自定义这个函数时也应该遵循三个参数的设计\(具体可以参考[这个问答](https://www.reddit.com/r/learnprogramming/comments/duvc2r/problem_with_classes_and_with_statement_in_python/)\).

### 5.2 with语法与怎么让自定义类支持with语法

> This interface of \_\_enter\_\_\(\) and \_\_exit\_\_\(\) methods which provides the support of with statement in user defined objects is called `Context Manager`.

总的来说, 需要让类支持with语法, 只需要定义魔术方法`__enter__`与`__exit__`即可, 一个完整的例子如下

```python
class A():
    def __init__(self):
        print("create A")
    def do_before_enter(self):
        print("do before exit")
        self.a = 1
    def __enter__(self):
        self.do_before_enter()
        print("__enter__")
        return self.a  # 如果使用with A() as x形式, 此处的返回值由x接收
    def __exit__(self, exc_type, exc_value, traceback):
        self.do_before_exit()
        print("__exit__")
    def do_before_exit(self):
        print("do before exit")
        del self.a

x = A()
print(hasattr(x, "a"))  # False
with x as a:
    print(hasattr(x, "a"))  # True
    print(x is a)  # False
    print(f"run with block, a: {a}")
    # 取消下一行的注释, __exit__方法依然会被调用
    # xxx(f"run with block, a: {a}")
print(hasattr(x, "a"))  # False

# 忽略异常处理, 基本等同于如下代码段
# x = A()
# a = x.__enter__()
# print(f"run with block, a: {a}")
# x.__exit__(None, None, None)
```

### \*5.3 使用contextlib包中的函数来使得类支持with语法

按照上一节的做法, 可以使用如下写法让`MassageWriter`支持with语法

```python
class MessageWriter(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __enter__(self):
        self.file = open(self.file_name, 'w')
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

with MessageWriter('my_file.txt') as xfile:
    xfile.write('hello world')
```

也可以使用`contextlib`中的一些方法不进行显式定义`__enter__`与`__exit__`使得自定义类能支持with语法, 例子如下

```python
from contextlib import contextmanager

class MessageWriter(object):
    def __init__(self, filename):
        self.file_name = filename

    # 此处需要定义为生成器而不能是函数，并且该迭代器必须只能有一个
    @contextmanager
    def open_file(self):
        try:
            file = open(self.file_name, 'w')
            yield file
        finally:
            file.close()

message_writer = MessageWriter('record.txt')
with message_writer.open_file() as my_file:
    my_file.write('Hello world')
```

执行顺序为：首先 `open_file` 函数被调用，并且将返回值 `file` 传递给 `my_file`，之后执行 with 语句内部的`write` 方法, 之后再回到 `open_file` 方法的 `yeild file` 后继续执行。可以简单理解为：

* open_file函数从第一个语句直到第一个yield语句为`__enter__`
* open_file函数从第一个yield语句到最后为`__exit__`

### 5.4 "复合"with语句

```python
with open(in_path) as fr, open(out_path, "w") as fw:
    pass
```

```python
from contextlib import ExitStack
import csv
def rel2logic(in_path, logic_dir):
    """将关系表转为逻辑客关系表形式
    Example:
        >>> rel2logic("./python_logical/tests/all_relations.tsv", "./python_logical/tests/gen")
    """
    with ExitStack() as stack:
        fr = csv.DictReader(stack.enter_context(open(in_path, encoding="utf-8")), delimiter="\t")
        fws = {}
        for row in fr:
            start_type, end_type = row["start_type"], row["end_type"]
            start_id, end_id, relation = row["start_id"], row["end_id"], row["relation"]
            key = start_type + "-" + end_type + ".tsv"
            if key not in fws:
                out_path = os.path.join(logic_dir, key)
                fw = stack.enter_context(open(out_path, "w", encoding="utf-8"))
                fws[key] = csv.writer(fw, delimiter="\t", lineterminator="\n")
                fws[key].writerow([start_type, end_type, "relation"])
            fws[key].writerow([start_id, end_id, relation])
```

## 6. for else语法

```python
# 获取[1, n]中的所有素数
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print( n, 'equals', x, '*', n/x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')
# 来源于Cython文档里的例子
```

## 7. python基本数据类型

int: 无限精度整数

float: 通常利用`C`里的`double`来实现

## 8. 函数的参数

参考[知乎](https://www.zhihu.com/question/57726430/answer/818740295)

**函数调用**

```
funcname(【位置实参】,【关键字实参】)
```

使用了 `a=x` 这种方式传参的即为关键字实参。

两个具有一般形式的例子

```python
# 1, 2 为位置实参，
foo(1, 2, a=3, b=4)  # 一般调用形式
foo(1, *[0], 2, *[3, 4], a=1, **{"c": 1}, **{"d": 1})  # 特殊调用形式
```

**函数定义**

```
def funcname(【限定位置形参】,【普通形参】,【特殊形参args】,【限定关键字形参】,【特殊形参kwargs】): pass
```

<font color=red>备注：限定位置形参在 Python 3.8 才被正式引入，即 `/` 这种写法。在此之前仅有后面的四种形参</font>

一个具有一般形式的例子：

```python
def foo(a, b, /, c, d=3, *args, e=5, f, **kwargs): pass
def foo(a, b=1, /, c=2, d=3, *, e=5, f, **kwargs): pass
```

- `a` 与 `b` 为限定位置形参
- `c` 与 `d` 为普通形参
- `e` 与 `f` 为限定关键字形参

**形实结合的具体过程**

首先用位置实参依次匹配限定位置形参和普通形参，其中位置实参的个数必须大于等于限定位置形参的个数，剩余的位置实参依顺序匹配普通形参。

- 若位置实参匹配完全部限定位置形参和普通形参后还有剩余，则将剩余参数放入 `args` 中
- 若位置实参匹配不能匹配完全部普通形参，则未匹配上的普通形参留待后续处理

接下来用关键字实参匹配普通形参和限定关键字形参，匹配方式按参数名匹配即可。

**设定默认值的规则**

为形参设定默认值的规则与前面的规则是独立的。

- 限定关键字形参，带默认值与不带默认值的形参顺序随意
- 限定位置形参和普通形参，带默认值的形参必须位于不带默认值的形参之后

## 9. 导包规则

参考：

- [RealPython: python-import](https://realpython.com/python-import/)
- [RealPython: modules-packages-introduction](https://realpython.com/python-modules-packages)
- [Real Python: Namespaces and Scope in Python](https://realpython.com/python-namespaces-scope/)

首先，需要厘清几个概念：

- namespace

  ```
  import a
  print(a.xxx)
  ```

  这里的 `a` 是一个 `namespace`

- module

  单个 `.py` 文件是一个 `module`

- package

  目录，且目录下有 `__init__.py` 文件

- namespace package

  目录，且目录下没有 `__init__.py` 文件

### 9.1 namespace

- built-in namespace (运行脚本里的变量)
- global namespace
- enclosing namespace (带有内层函数的函数)
- local namespace (函数最里面的一层)

```python
>>> globals()  # 返回global namespace
'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <class '_frozen_importlib.BuiltinImporter'>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 
'builtins' (built-in)>
>>> locals()  # 返回local/enclosing namespace, 当位于运行脚本时, 与globals结果一样
```

```python
global x, y  # 声明当前作用域下引用的是全局变量x, y
nonlocal x, y  # 声明当前作用域下引用的是上一层作用域的x, y
```

```python
from xx import yy
# 将yy引入当前作用域, sys.modules中会有xx模块

import xx.yy  # xx必须是一个包, yy可以是一个包或模块
# sys.modules会显示xx是一个namespace, xx.yy是一个模块
# globals() 只包含xx, 不包含yy及xx.yy

import .xx  # 不允许
```

global namespace 需要额外进行说明，与 import 相关。

### 9.2 import 语法详解

**绝对导入与相对导入**

```python
# 绝对导入
from aa import bb
from aa.bb import C
import aa.bb  # aa.bb 必须为一个module/namespace package/package

# 相对导入：必须以点开头，且只有from ... import ...这一种写法
from . import aa
from ..aa.bb import cc
# import .aa  # 无此语法
```

**`from ... import ...` 语法详解**

下面分别对上述导入语句作解析：

```
from aa import bb
```

导入成功只能为三种情况

- `aa` 是一个不带 `__init__.py` 的文件夹（namespace package）。

  - `bb` 是一个 `bb.py` 文件。则可以直接使用 `bb`，但不能使用 `aa` 以及 `aa.bb`。注意，此时

  ```python
  sys.modules["aa"]  # 显示为namespace
  sys.modules["aa.bb"]  # 显示为module
  sys.modules["bb"]  # 报错
  ```

  - `bb` 是一个带或者不带 `__init__.py` 的文件夹，情况类似，唯一的区别是此时 `bb` 会显示为一个 module 或者是 namespace。

- `aa` 是一个带有 `__init__.py` 的文件夹（package），则上述导入成功的条件为 `bb` 在 `aa/__init__.py` 中是一个标识符，或者 `bb` 是 `aa` 的子目录，或者 `bb.py` 在文件夹 `aa` 下。无论是哪种情况，`aa/__init__.py` 均会被执行，且 `aa` 与 `aa.bb` 不可直接使用。下面是一个例子：

  目录结构为

  ```
  aa
    - __init__.py
    - bb.py
  ```

  文件内容为

  ```python
  # aa/__init__.py
  c = 1
  print(c)
  # bb.py
  # 无内容
  ```

  使用

  ```python
  >>> from aa import bb  # 注意此时已经将c打印了
  1
  >>> bb
  <module 'aa.bb' from 'aa/bb.py'>
  >>> # aa.cc, aa, aa.bb # 三者均不可使用
  >>> import aa  # 注意aa/__init__.py不会再次被执行
  >>> aa.bb
  <module 'aa.bb' from 'aa/bb.py'>
  >>> aa.c
  1
  >>> aa
  <module 'aa.bb' from 'aa/__init__.py'>
  ```

- `aa` 是一个 `aa.py` 文件，则上述导入成功的条件为 `aa.py` 中可以使用 `bb` 这一标识符。

```
from aa.bb import C
```

结论：对于这种形式的导入

```
from xx import yy
from xx.yy import zz
```

`xx.py` 或 `xx/__init__.py` 只要有就会被执行。并且 `xx` 与 `yy` 是 namespace package 还是 package 不影响导入，最终只有 import 后面的东西可以直接使用。

**`import ...` 语法详解**

```python
import aa.bb.cc
```

导入成功只能为一种情况 `aa/bb/cc.py` 或着 `aa/bb/cc` 存在，作用是依次执行 `aa/__init__.py`，`aa/bb/__init__.py`，`aa/bb/cc.__init__.py` （若它们都是package）。无论 `aa` 与 `bb` 是 package/namespace package，以下标识符均可以直接使用：

```
aa
aa.bb
aa.bb.cc
aa.foo  # foo 在 aa/__init__.py 中
aa.bb.bar  # bar 在 bb/__init__.py 中
```

以下不可使用

```
aa.zz  # aa/zz.py文件, 且aa/__init__.py中没有from . import zz
```

备注：无论是 `from ... import ...` 还是 `import ...`，相关包的 `__init__.py` 及 `xx.py` 模块均会被执行一次。后续若再次 import，无论文件是否发生变动，均不会再次运行 `__init__.py` 或 `xx.py` 文件。只是标识符是否可用发生变化。

**彻底理解import**

**step1：官方文档搜索记录**

平时惯用的 import 语法是 `importlib.__import__` 函数的语法糖：

> The [`__import__()`](https://docs.python.org/3/library/importlib.html?highlight=import#importlib.__import__) function
>
> ​		The [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement is syntactic sugar for this function
> ——https://docs.python.org/3/library/importlib.html

其函数定义为（[链接](https://docs.python.org/3/library/importlib.html?highlight=import#importlib.__import__)）：

```python
importlib.__import__(name, globals=None, locals=None, fromlist=(), level=0)
```

官方对此函数的解释为：

> An implementation of the built-in [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) function.
>
> Note: Programmatic importing of modules should use [`import_module()`](https://docs.python.org/3/library/importlib.html?highlight=import#importlib.import_module) instead of this function.

即：`importlib.__import__` 是内置函数的一种实现。备注：此处官方的超链接疑似有误，似乎应该是：**平时惯用的 import 语法是内置函数 `__import__` 函数的语法糖**

而内置函数 `__import__` 的定义为（[链接](https://docs.python.org/3/library/functions.html#__import__)）：

```python
__import__(name, globals=None, locals=None, fromlist=(), level=0)
```

官方对此函数有如下注解：

> This function is invoked by the [`import`](https://docs.python.org/3/reference/simple_stmts.html#import) statement.  It can be replaced (by importing the [`builtins`](https://docs.python.org/3/library/builtins.html#module-builtins) module and assigning to `builtins.__import__`) in order to change semantics of the `import` statement, but doing so is **strongly** discouraged as it is usually simpler to use import hooks (see [**PEP 302**](https://www.python.org/dev/peps/pep-0302)) to attain the same goals and does not cause issues with code which assumes the default import implementation is in use.  Direct use of [`__import__()`](https://docs.python.org/3/library/functions.html#__import__) is also discouraged in favor of [`importlib.import_module()`](https://docs.python.org/3/library/importlib.html#importlib.import_module).

可以看到，`importlib.__import__` 与内置函数 `__import__` 的定义完全相同。

总结：平时所用的 import 语句仅仅是 `importlib.__import__` 函数（也许是内置函数 `__import__`）的语法糖。而 `importlib.__import__` 是内置函数 `__import__` 的一种实现，建议不要直接使用 `importlib.__import__` 与内置的 `__import__` 函数。

整理一下官方说明链接：

- `import` 语法：[链接1](https://docs.python.org/3/reference/simple_stmts.html#import)
- `importlib.__import__` 函数：[链接2](https://docs.python.org/3/library/importlib.html#importlib.__import__)
- `__importlib__` 内置函数：[链接3](https://docs.python.org/3/library/functions.html#__import__)

由于 `importlib.__import__` 函数几乎没有任何说明，因此主要看链接 1 与 3。

**step 2：官方文档理解**

首先，回顾内置函数 `__import__` 的定义：

```
__import__(name, globals=None, locals=None, fromlist=(), level=0)
```

在标准实现中，locals 参数被忽略。import 语法糖与 `__import__` 内置函数的对应关系为：

官方文档的三个例子

```python
import spam
spam = __import__('spam', globals(), locals(), [], 0)
```

```python
import spam.ham
spam = __import__('spam.ham', globals(), locals(), [], 0)
```

```python
from spam.ham import eggs, sausage as saus
_temp = __import__('spam.ham', globals(), locals(), ['eggs', 'sausage'], 0)
eggs = _temp.eggs
saus = _temp.sausage
```

晦涩难懂，之后再补充。



Python 导包的常用方法有：import 语句、`__import__` 内置函数、`importlib` 模块。本质上讲，第一种方法实际上会调用第二种方法，而第三种方法会绕过第二种方法，一般而言不推荐直接使用第二种方法。

import 语句与 `__import__` 内置函数的对应关系可以参见[官方文档](https://docs.python.org/zh-cn/3/library/functions.html#__import__)。

怎样完全删除一个已经被导入的包，似乎做不到，参考[链接](https://izziswift.com/unload-a-module-in-python/)

怎样实现自动检测包被修改过或未被导入过，自动进行 reload 操作：待研究

一些疑难杂症：

**实例1：**

```
pkg1
- inference.py  # Detect
pkg2
- inference.py  # Alignment
```

想获得两个包中的模型实例，将两个模型串联进行推断

```python
# 第三个参数是为了防止模型用torch.save(model)的方式保存, 需要额外引入一些包
def get_model_instance(extern_paths, module_cls_pair, extern_import_modules=None, *args, **kwargs):
    sys.path = extern_paths + sys.path
    extern_import_modules = extern_import_modules if extern_import_modules else []
    extern_list = [importlib.import_module(extern_name) for extern_name in extern_import_modules]
    modname, clsname = module_cls_pair
    mod = importlib.import_module(modname)
    instance = getattr(mod, clsname)(*args, **kwargs)
    # 对sys.modules操作可能不够, 未必能删干净
    for extern in extern_import_modules:
        sys.modules.pop(extern)
    sys.modules.pop(modname)
    sys.path = sys.path[len(extern_paths):]
    return instance
```

```
detector = get_model_instance(["pkg1"], ("inference", "Detect"), [])
detector = get_model_instance(["pkg2"], ("inference", "Alignment"), [])
```

```python
detector = get_model_instance(["./detect/facexzoo"], ("inference", "Detect"), ["models"])
```

用于替代

```python
sys.path = ["./detect/facexzoo"] + sys.path
from inference import Detect
import models
sys.path = sys.path[1:]
detector = Detect()
sys.modules.pop("models")
sys.modules.pop("Detect")
```

**实例2：**

假定目录结构为：

```
ROOT
  - models.py
  - load_detr.py
```

文件内容如下：

```python
# models.py
a = 1

# load_detr.py
import torch
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False)
from models import a
print(a)
```

运行：

```bash
python load_detr.py
```

报错：

```
ImportError: cannot import name 'a' from 'models'
```

原因在于 `torch.hub.load` 的内部逻辑为：

- 按照 `facebookresearch/detr:main` 去 GitHub 下载原始仓库（https://github.com/facebookresearch/detr）的代码至 `~/.cache/torch/hub` 下。

  备注：此处的 `main` 代表 `main` 分支，代码下载解压完毕后，`~/.cache/torch/hub` 目录下会生成子目录 `facebookresearch_detr_main` 存放当前分支下的代码

  备注：如果原始 GitHub 仓库进行了更新，而本地之前已经下载了之前版本的仓库，可以使用如下方法重新下载

  ```python
  model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False, force_reload=True)
  ```

- 接下来使用动态 import 的方式，增加了 `~/.cache/torch/hub/facebookresearch_detr_main` 到 sys.path 并使用 importlib 中的相关函数导入代码仓库顶级目录中的 `hubconf.py` 文件里的 `detr_resnet50` 函数，构建模型并下载权重。随后在 sys.path 中移除了 `~/.cache/torch/hub/facebookresearch_detr_main` 路径。 

问题出现在上述仓库的 `hubconf.py` 文件里有这种 import 语句：

```python
from models.backbone import Backbone, Joiner
from models.detr import DETR, PostProcess
def detr_resnet50(...)
```

导致当前目录下的 models 无法被重新导入

修改策略（未必万无一失）：

```python
import torch
model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False)
import sys
sys.modules.pop("models")
from models import a
```

## 10. Python buid-in fuction and operation

参考资料：[Python 标准库官方文档](https://docs.python.org/3/library/functions.html)

**Truth Value Testing**

任何对象都可以进行 Truth Value Testing（真值测试），即用于 `bool(x)` 或 `if` 或 `while` 语句，具体测试流程为，首先查找该对象是否有 `__bool__` 方法，若存在，则返回 `bool(x)` 的结果。然后再查找是否有 `__len__` 方法，若存在，则返回 `len(x)!=0` 的结果。若上述两个方法都不存在，则返回 `True`。

备注：`__bool__` 方法应返回 `True` 或者 `False`，`__len__` 方法应返回大于等于 0 的整数。若不遵循这些约定，那么在使用 `bool(x)` 与 `len(x)` 时会报错。相当于：

```python
def len(x):
	length = x.__len__()
	check_non_negative_int(length)  # 非负整数检验
	return length
def bool(x):
    if check_bool_exist(x):  # 检查__bool__是否存在
        temp = x.__bool__()
        check_valid_bool(temp)  # bool值检验
        return temp
    if check_len_exist(x):  # 检查__len__是否存在
        return len(x) != 0
    return True
```

备注：`__len__` 只有被定义了之后，`len` 方法才可以使用，否则会报错

**boolean operation: or, and, not**

运算优先级：`非bool运算 > not > and > or`，所以 `not a == b ` 等价于 `not (a == b)`

注意这三个运算符的准确含义如下：

```python
not bool(a)  # not a
a and b  # a if bool(a)==False else b
a or b  # a if bool(a)==True else b
```

```python
12 and 13  # 13
23 or False  # 23
```

**delattr function and del operation**

```python
delattr(x, "foo")  # 等价于 del x.foo
```

## 11. Python 内存管理与垃圾回收（待补充）

## 12. 怎么运行 Python 脚本

主要参考（翻译）自：[RealPython](https://realpython.com/run-python-scripts/)

主要有：

- python xx/yy.py
- python -m xx.yy
- import
- runpy
- importlib
- exec

## 13. 迭代器与生成器

```python
class A:
	def __iter__(self):
        for i in range(10):
            yield i
a = A()  # a是一个可迭代对象(Iterable)
iter(a)  # 返回的是一个生成器(特殊的迭代器)
```

## 14. 写一个 Python 包

### github

对照几份源码进行学习

- pip: 
- numpy: https://github.com/numpy/numpy
- pytorch: https://github.com/pytorch/pytorch

可以使用 `git clone <url>.git` 的方式克隆源代码，这里的 url 的形式为 `https://github.com/<username or groupname>/<projectname>`。而以 numpy 为例，其简化版目录结构如下：

```
ROOT/
    numpy/
    doc/
    setup.py
    README.md
```

大体上讲，平时使用 `pip install numpy` 实际发生的事情是，将此处的目录 `numpy` 放到 `site-packages` 目录下，而其余的 `doc` 目录的内容将不会被安装。



### 一个最简的例子

目录结构如下
```
ROOT/  # 譬如说对应于https://github.com/<username or groupname>/<projectname>

```

### 项目组织形式

参考 [stackoverflow](https://stackoverflow.com/questions/193161/what-is-the-best-project-structure-for-a-python-application)，推荐以类似这种形式组织，注意这些 `__init__.py` 文件是必须的，以确保

```
Project/
|-- bin/
|   |-- project
|
|-- project/
|   |-- test/
|   |   |-- __init__.py
|   |   |-- test_main.py
|   |   
|   |-- __init__.py
|   |-- main.py
|
|-- setup.py
|-- README
```

安装方式为：

```bash
python setup.py install  # 安装在site-packages目录下
pip install /path/to/Project  # 安装在site-packages目录下
pip install -e /path/to/Project  # 安装在当前目录, 适用于开发阶段, 对项目的修改会直接生效, 做修改后无需重新安装包
```

<font color=red>特别说明</font>：关于测试数据与测试代码文件：以下为个人理解，不一定为最佳实践，测试代码中读取数据时应该要获取完整的路径，可以考虑使用 `__file__` 结合相对路径以获取绝对路径。关于这一点，有如下的一个源码分析案例：

源码分析：参考 [scikit-image](https://github.com/scikit-image/scikit-image) 的源代码

```python
from skimage import data
camera = data.camera()
```

其中，`data.camera` 函数的定义位于 `skimage/data/__init__.py`，它进一步调用了同文件下的 `_load("data/camera.png")`，而 `_load` 函数又调用了同文件下的 `_fetch("data/camera.png")`，而 `_fetch` 函数的关键代码如下：

```python
def _fetch(data_filename):
    resolved_path = osp.join(data_dir, '..', data_filename)  # data_dir为该文件的全局变量, 使用了类似os.path.abspath, __file__ 的方式得到
    return resolved_path
```

例子：

项目

```
Foo/
  foo/
  	main.py
  	__init__.py
  	data/
  	  data.txt
  setup.py
```

`foo/main.py`

```python
import os
cur_dir = os.path.dirname(__file__)
with open(os.path.join(cur_dir, "./data/data.txt")) as f:
  print(f.readlines())
```

`foo/__init__.py` 内容为空

`data/data.txt`

```
hello
```

`setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="Foo",
    version="1.0",
    author="yourname",
    packages=find_packages(),
    install_requires=[],
    include_package_data=True,
    package_data={"foo": ["data/*"]}
)
```

安装与使用

```python
# python setup.py install path/to/Foo
from foo import main
```

备注：安装在 `site-packages` 目录下

```
Foo-1.0.dist-info  # Foo与setup.py中的name相对应
foo  # python源代码
```

### 安装依赖包

**第一步：获取requirements.txt**

**方法一: 只获取必要的包（推荐使用）**

```text
pip install pipreqs
cd project_path
pipreqs ./ --encoding=utf8
```

**方法二: 获取当前环境下的所有包**

此方案尽量避免使用, 或者在一个干净的虚拟环境下使用

```text
pip freeze > requirements.txt
```

**第二步：利用requirements.txt安装依赖包**

```text
pip install -r requirements.txt
```

### 项目打包详解

问题引出:

* 想开发一个python包上传到PyPI
* 在一个项目中想使用另一个项目的功能: [stackoverflow的一个问题](https://stackoverflow.com/questions/14509192/how-to-import-functions-from-other-projects-in-python)

一些历史, 关于`distutils`, `distutils2`, `setuptools`等, [参考链接](https://zhuanlan.zhihu.com/p/276461821). 大体来说, `distutils`是最原始的打包工具, 是Python标准库的一部分. 而`setuptools`是一个第三方库, 在`setuptools`的变迁过程中, 曾出现过一个分支`distribute`, 现在已经合并回`setuptools`, 而`distutils2`希望充分利用前述三者:`distutils`, `setuptools`, `distribute`的优点成为标准库的一部分, 但没有成功, 并且已经不再维护了. 总之, `distutils`是标准库, `setuptools`是开发者常用的第三方库, 安装好后还额外带着一个叫`easy_install`的第三方管理工具, 而`easy_install`目前用的比较少, `pip`是其改进版. 顺带提一句: python源码安装一般是下载一个压缩包\(先解压, 再编译, 再安装\), 二进制安装一般是下载一个`.egg`或者`.whl`的二进制文件进行安装, 后者已经取代前者成为现今的通用标准. 下面仅介绍基于`setuptools`的使用, 其关键在于编写`setup.py`. 上传到PyPI的方法参考[python官方文档.](https://packaging.python.org/tutorials/packaging-projects/)


#### `pip install` vs `python setup.py install`

一般来说，参考[stackoverflow](https://stackoverflow.com/questions/15724093/difference-between-python-setup-py-install-and-pip-install)，推荐使用 `pip install`.

- pip 会自动安装依赖包，使用 setup.py 通常需要手动安装。（此条存疑）解释：使用 pip 安装时一般只需要 `pip install <PACKAGE_NAME>` 即可，而 setup.py 通常需要 `pip install -r requirements.txt & python setup.py install`
- pip 会自动追踪包的 metadata, 所以在卸载包时可以使用 `pip uninstall <PACKAGE_NAME>`，但是使用 setup.py 需要手动卸载再升级
- pip 可以不需要手动下载：`pip install xx`（PyPi），`pip install git+https://github.com/xxx/xxx.git`（github/gitlab/...），或者对压缩包或whl文件安装：`pip install xx.tar.gz`，`pip install xx.whl`。而 `setup.py` 只能下载并解压后才能安装


备注

对同一个包的安装混用 pip 与 setup 有时会出现一些难以解决的 bug。

|pip|setup||
|---|---|---|
|pip install .|python setup.py install||
|pip install -e .|python setup.py develop||


#### setup.py 的编写与使用简介

首先尝鲜，在介绍各个参数的用法（完整列表参见[官方文档](https://setuptools.readthedocs.io/en/latest/references/keywords.html)）

```text
funniest/
    funniest/
        __init__.py
        text.py
    setup.py
```

```python
from setuptools import setup

setup(name='funniest',  # 包的名称, 决定了用pip install xxx
      version='0.1.1',  # 版本号
      description='The funniest joke in the world',  # 项目描述
      url='http://github.com/storborg/funniest',  # 项目链接(不重要)
      author='Flying Circus',  # 作者名(不重要)
      author_email='flyingcircus@example.com',  # 作者邮箱(不重要)
      license='MIT',
      packages=['funniest'], # 实际上是内层的funniest, 决定了import xxx
      install_requires=[
          'markdown',
      ])  # 依赖项, 优于手动安装requires.txt里的包的方法
```

```bash
# 源码安装只需一行
python setup.py install

# 上传到PyPI也只需一行(实际上有三步: 注册包名, 打包, 上传)
python setup.py register sdist upload
# 上传后就可以直接安装了
pip install funniest

# 打包为whl格式(以后补充)
```

#### setup.py 的 setup 函数的各个参数详解

**xx_requires**
```python
setup(
    install_requires=['numpy'],  # 若当前环境没有,会从pypi下载并安装
    setup_requires=['pdr'],  # setup.py本身依赖的包,通常是给setuptools的插件准备的配置,若缺少,不会自动安装, 而是会在执行pip install或python setup.py install时直接报错
    tests_require=['pytest>=3.3.1', 'pytest-cov>=2.5.1'],  # 执行python setup.py test时h会自动安装的库
    extras_require={
        "PDF": ["pdfplumber"],
        "Excel": ["pandas==1.0.0"]
    },  # 不会自动安装, 在深度使用时, 需要手动安装
    python_requires='>=3.7, <=3.10'
)
```

**entry_points 参数**

```python
entry_points={
        "console_scripts": [
            "labelme=labelme.__main__:main",
            "labelme_draw_json=labelme.cli.draw_json:main"
        ],
    },
```

指定这组参数后，例如：`"labelme=labelme.__main__:main"` 这一行表示执行完安装命令后，与可执行文件 `python` 同级的目录下会出现可执行文件 `labelme`，如果执行该文件，则等同于执行 `labelme.__main__.py` 文件内的 `main` 函数。

**scripts 参数**

似乎不推荐使用

**其他参数**
|参数 |含义|
|:-----------|:-----------------------|
|zip_safe|设置为`False`表示以文件夹的形式安装(方便调试), 设置为`True`表示安装形式为一个`.egg`压缩包|

**已经弃用的参数**

| 已弃用的参数 | 替代品             | 含义                     |
| :----------- | :----------------- | :----------------------- |
| `requires`   | `install_requires` | 指定依赖包               |
| `data_files` | `package_data`     | 指定哪些数据需要一并安装 |

将非代码文件加入到安装包中，注意：这些非代码文件需要放在某个包（即`packages` 列表）下，使用以下两种方式之一即可

* 使用`MANIFEST.in`文件\(放在与`setup.py`同级目录下\), 并且设置`include_package_data=True`, 可以将非代码文件一起安装
* `package_data`参数的形式的例子为：`{"package_name":["*.txt", "*.png"]}`

**例子 1**

[labelme-4.5.12](https://github.com/wkentaro/labelme) 的源码目录如下：

```
ROOT
  - .github/      # 无__init__.py文件
  - docker/       # 无__init__.py文件
  - docs/         # 无__init__.py文件
  - examples/     # 无__init__.py文件
  - github2pypi/  # 有__init__.py文件
  	- __init__.py
  	- replay_url.py
  	- ...
  - labelme/
    - __init__.py
    - ...
  - tests/        # 无__init__.py文件
  	- labelme_tests/
  	  - __init__.py
  	  - test_app.py
  	  - ...
  	- doc_tests/
  - setup.py
  - README.md
  - .gitignore
  - MANIFEST.in
  - LICENSE
  - ...
```

其中 labelme 文件夹内部的文件目录为：

```
- __init__.py
- cli/
  - __init__.py
  - draw_json.py
  - draw_label_png.py
  - json_to_dataset.py
  - on_docker.py
- config  # 存放着非py文件，安装后有此目录及文件
  - __init__.py
  - default_config.yaml
- icons  # 存放着非py文件，安装后有此目录及文件
  - *.png
- translate/  # 存放着非py文件，安装后无此目录
- utils/
- widgets/
- ...
```

setup 函数如下

```python
setup(
    name="labelme",
    version=version,
    packages=find_packages(exclude=["github2pypi"]),
    description="Image Polygonal Annotation with Python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Kentaro Wada",
    author_email="www.kentaro.wada@gmail.com",
    url="https://github.com/wkentaro/labelme",
    install_requires=get_install_requires(),
    license="GPLv3",
    keywords="Image Annotation, Machine Learning",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    package_data={"labelme": ["icons/*", "config/*.yaml"]},
    entry_points={
        "console_scripts": [
            "labelme=labelme.__main__:main",
            "labelme_draw_json=labelme.cli.draw_json:main",
            "labelme_draw_label_png=labelme.cli.draw_label_png:main",
            "labelme_json_to_dataset=labelme.cli.json_to_dataset:main",
            "labelme_on_docker=labelme.cli.on_docker:main",
        ],
    },
    data_files=[("share/man/man1", ["docs/man/labelme.1"])],
)
```

使用 `python setup.py install` 后，安装相关的存储路径（conda）例如

```
anaconda3/envs/env_name/Scripts
anaconda3/envs/env_name/Lib/site-packages/labelme
anaconda3/envs/env_name/Lib/site-packages/labelme-4.5.12.dist-info
```

`Scripts` 目录下多出了

```
labelme.exe
labelme_draw_json.exe
labelme_draw_label_png.exe
labelme_json_to_dataset.exe
labelme_on_docker.exe
```

**打包方式最佳实践**

目前主流的打包格式为 `whl` 格式（取代 `egg` 格式），发布到 PyPi 的包一般使用下面的命令进行安装

```shell
pip install <packagename>
```

实际过程为按照包名 `<packagename>` 在互联网上搜索相应的 .whl 文件，然后进行安装。因此对于源码安装的最佳实践也沿用上述过程，详述如下：

`setup.py` 文件的 `setup` 函数的参数 `packages` 列表长度最好刚好为 1，此时 `setup.py` 文件的 `setup` 函数的参数 `name` 应与 `packages` 的唯一元素相同，且命名全部用小写与下划线，且尽量不要出现下划线。使用下面两条命令安装

```
python setup.py bdist_wheel  # 打包为一个.whl文件，位于当前文件夹的dist目录下
pip install dist/xxx-1.7.4-py3-none-any.whl
```

在 site-packages 目录下会出现类似于如下两个目录

```
xxx-1.7.4.dist-info
xxx
```

备注：whl 格式实际上是 zip 格式，因此可以进行解压缩查看内容

### 发布到 PyPi

参考资料

- [参考realpython](https://realpython.com/pypi-publish-python-package/#different-ways-of-calling-a-package)

## 附录 1

### 骚操作

来源：torch/cuda/amp/grad_scaler.py，`GradScaler:scale` 函数

```
type((1, 2))([1, 2, 3])
```



### 不能实例化的类

```python
from typing import List
List[int]()  # 注意报错信息
```

### python dict与OrderedDict

关于python自带的字典数据结构, 实现上大致为\([参考stackoverflow回答](https://stackoverflow.com/questions/327311/how-are-pythons-built-in-dictionaries-implemented)\):

* 哈希表\(开放定址法: 每个位置只存一个元素, 若产生碰撞, 则试探下一个位置是否可以放下\)
* python 3.6以后自带的字典也是有序的了\([dict vs OrderedDict](https://realpython.com/python-ordereddict/)\)

说明: 这里的顺序是按照key被插入的顺序决定的, 举例

### 深复制/浅复制/引用赋值

引用赋值: 两者完全一样, 相当于是别名: `x=[1, 2, 3], y=x` 浅赋值: 第一层为复制, 内部为引用: `list.copy(), y=x[:]` 深复制: 全部复制, `import copy; x=[1, 2]; copy.deepcopy(x)`

[Python 直接赋值、浅拷贝和深度拷贝解析 \| 菜鸟教程 \(runoob.com\)](https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html)

### Immutable与Hashable的区别

immutable是指创建后不能修改的对象, hashable是指定义了`__hash__`函数的对象, 默认情况下, 用户自定义的数据类型是hashable的. 所有的immutable对象都是hashable的, 但反过来不一定.

另外还有特殊方法`__eq__`与`__cmp__`也与这个话题相关