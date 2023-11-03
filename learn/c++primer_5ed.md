# C++ Primer (5ed)

基于C++11标准

备注: c++11标准虽然不是最新的, 但是后续的重要基础, c++14对c++11做了少数修补, c++17与c++20不清楚改动多大. 由于此书交叉引用过多, 所以计划先初步做笔记, 多看几遍后再补充完整。目录完全照搬原书，但所记内容不一定。

# 第 1 章

`main`函数的返回值必须为`int`

# 第 2 章: 变量与基本数据类型 (Variables and Basic Types)

## 2.1 原生内置类型 (Primitive Build-in Types)

`bool/char/wchar_t/char16_t/char32_t/short/int/long/long long/float/double/long double/void`

* 数值类型分为两类, 一类是整数类型 (包括 bool 与 char 在内 ), 另一类是浮点型.
* 注意C++标准没有规定 `char` 类型是指 `unsigned char` 还是 `signed char`
* 一般情况下, 整数类型默认为`signed` (`char`是个例外).
* C++标准规定有符号整数正负区间应大致平分, 但并未规定其存储形式. 以 `signed char` 为例, C++标准规定其至少要包含 `[-127, 127]` 这些整数, 但一般而言, 大多数机器都使用补码形式存储整数, 所以一般取值范围一般为 `[-128, 127]`

关于类型使用的一些建议(略)

**类型转换**

```text
unsigned int a = -1  // 直接以unsigned int的方式解读补码形式的-1即可(2^32-1)
```

待确认: 整数混合计算时, 按如下方式进行转换 (以最大容量为准), 例子: **32 位无符号整数认为具有 32 位, 32 位有符号整数认为具有 31 位**, 操作数先统一转为较大者再计算

```text
unsigned int a;
int b;
long long c;
a + b;  // a + (unsigned int)(b)
a + c;  // (long long)(a) + c
```

(Literals) 字面量:

整数/浮点数/字符/字符串/布尔/指针(void). 注意`"abc"`的类型是`const char[]`

## 2.2 变量 (Variables)

**变量初始化写法**

以下写法均可

```text
int i = 0;//用0.0初始化OK,但会产生截断
int i = {0};//不能用0.0初始化
int i{0};//不能用0.0初始化
int i(0);//用0.0初始化OK,但会产生截断
```

**声明与定义**

```text
extern int i;//变量声明
extern int i = 0;//变量定义, extern不起作用
void foo(int i);//函数声明
```

**命名规则与命名规范**

> C++ Primer 5ed 2.2.3 Identifies
>
> The standard also reserves a set of names for use in the standard library. The identifiers we define in our own programs may not contain two consecutive underscores, nor can an identifier begin with an underscore followed immediately by an uppercase letter. In addition, identifiers defined outside a function may not begin with an underscore.

C++ Primer中提及到表示符中不要出现连续的两个下划线, 或者以下划线加大写字母开头, 函数外定义的变量不要以下化线开头, 但实际测试时并不会有报错信息

```text
#include<iostream>
//以下均为不符合规范的命名方式, 但不会报错
int _a = 1;
int main() {int __as = 1, as__ = 1, _B = 2; return 0;}
```

备注: C++中可以使用`and`, `or`等

## 2.3 复合类型 (Compound Types)

复合类型: 引用与指针

> a declaration is a base type followed by a list of declarators

例如: `int i = 0; &a = i; *p = &i`, `base type`为`int`, 后面的都叫`declarators`, 引用不是一个变量, 但指针是一个变量. 指针的值可以是: 一个对象的地址; 一个对象的下一个地址; 空指针字面量\(nullptr\); 无效地址

## 2.4 常量限定符 (const Qualifier)

用 `const` 修饰的变量不能被重新赋值，声明时必须初始化。

```c++
const int a = get_size();  // initialized at run time
const int b = 1;  // initialized at compile time
const int c;  // error!
```



```text
const int a = 0, *p = &a;//可以将const int合在一起视为base type
```

**const 与 extern**

在默认情况下，const 限定的标识符只在本文件中有效，即假设 `a.cpp` 中与 `b.cpp` 中同时用 `const int i = 1;` 进行了定义，`gcc a.cpp b.cpp -o out` 命令编译时不会报错。但假如没有用 `const` 限定，则会出现重定义的错误。

备注：`extern` 关键字的作用是**声明**标识符有全局作用域

为了使得一个常量被多个文件使用，建议按如下方式:

```text
extern const int a = foo(); // test.cpp
int b = 1;  // test.cpp
extern const int a; // test.h
extern int b;  // test.h
```

**top-level const 与 low-level const**

top-level const指变量本身不可变, low-level const指与之相关的变量不可变, top-level const变量可以用非const变量赋值, 但low-level const属性必须一致

**constexpr关键字**

`const expression`\(常量表达式\)在编译器被计算出来, 字面量是一个常量表达式, 一个用常量表达式初始化的`const`对象是一个常量表达式. `constexpr`关键词被用来提示编译器检查后面的内容必须是常量表达式, 确保其在编译过程中可以计算并计算出来

`decltype`关键字

```text
decltype(f()) sum = x;//利用f函数的返回类型得到类型
```

## 2.5 类型处理 (Dealing with Types)

## 2.6 自定义数据结构 (Defining Our Own Data Structures)


# 第 6 章: 函数 (Functions)

## 6.7 函数指针 (Pointer to Functions)

函数指针准确理解应该是: 指向函数类型的指针

涉及到函数指针, 类型会变得很复杂且费解, 下面的示例里主要注意这几点:

- function type 与 pointer to function type 有时候会发生自动转换, 但它们是不同的东西 (C++ Primer Plus 一书中指出了这个自动转换为什么合法的原因只是两种学说有矛盾, 因此 C++ 规定自动转换是合法的)
- `using`, `decltype` 在这里的使用, 通常能简化很多复杂的符号

```c++
bool lengthCompare(const string &, const string &);  // 函数的前向声明
bool (*pf)(const string &, const string &);  // 定义一个函数指针
// 函数名会自动转为指针, 两种写法都一样 (猜测的解释, 函数名本身在编译时会编译为一个地址)
pf = lengthCompare; // pf now points to the function named lengthCompare
pf = &lengthCompare; // equivalent assignment: address-of operator is optional

// 同样的道理使用函数指针调用函数时解引用符号 * 也是可以省略的
bool b1 = pf("hello", "goodbye"); // calls lengthCompare
bool b2 = (*pf)("hello", "goodbye"); // equivalent call
bool b3 = lengthCompare("hello", "goodbye"); // equivalent call

pf = nullptr;  // OK
// 函数指针在赋值时必须函数签名完全一致, 因此函数重载时总是会匹配到唯一的一个

// 将函数指针作为参数时, 下面两种写法也是没有区别的
// third parameter is a function type and is automatically treated as a pointer to function
void useBigger(const string &s1, const string &s2, bool pf(const string &, const string &));
// equivalent declaration: explicitly define the parameter as a pointer to function
void useBigger(const string &s1, const string &s2, bool (*pf)(const string &, const string &));

// 调用以函数指针作为参数的函数
// automatically converts the function lengthCompare to a pointer to function
useBigger(s1, s2, lengthCompare);

// 函数指针类型比较难写, 所以通常会用 typedef 来简化
// 注意 Func 与 Func2 是等价的类型, 而 FuncP 与 FuncP2 是等价的类型
// Func and Func2 have function type
typedef bool Func(const string&, const string&);
typedef decltype(lengthCompare) Func2; // equivalent type
// FuncP and FuncP2 have pointer to function type
typedef bool(*FuncP)(const string&, const string&);
typedef decltype(lengthCompare) *FuncP2; // equivalent type
// 备注: decltype 的返回的是 function type

// 然而在这里的函数时, Func 与 FuncP2 又都是等价的 (即上面 4 者都等价)
// equivalent declarations of useBigger using type aliases
void useBigger(const string&, const string&, Func);  // 自动认为Func是函数指针类型
void useBigger(const string&, const string&, FuncP2);

// 使用 using 语法起到 typedef 的作用, 以下展示返回类型是函数指针的函数(注意一个函数的返回类型不能是函数类型, 只能是函数指针)
using F = int(int*, int); // F is a function type, not a pointer
using PF = int(*)(int*, int); // PF is a pointer type
// 这里又很诡异, 返回类型必须是函数指针, 不能是函数类型
PF f1(int); // ok: PF is a pointer to function; f1 returns a pointer to function
F f1(int); // error: F is a function type; f1 can’t return a function
F *f1(int); // ok: explicitly specify that the return type is a pointer to function

// 表示一个返回类型是一个函数指针的函数, 这个返回的函数指针的类型是 int(int), 而这个函数的入参类型是(int*, int)
int (*f1(int))(int*, int);  // 很难读懂
auto f1(int) -> int (*)(int*, int);  // 与上面等价, 但很难读懂

string::size_type sumLength(const string&, const string&); // 函数声明
string::size_type largerLength(const string&, const string&);
// depending on the value of its string parameter,
// getFcn returns a pointer to sumLength or to largerLength
decltype(sumLength) *getFcn(const string &);
// genFcn 是一个函数指针, 函数的返回值是一个 sumLength 函数指针, 函数的入参是(const string&)
```

疑问: 一个具体的返回参数是函数的例子