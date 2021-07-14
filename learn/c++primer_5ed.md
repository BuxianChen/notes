# C++primer\(5ed\)

基于C++11标准

备注: c++11标准虽然不是最新的, 但是后续的重要基础, c++14对c++11做了少数修补, c++17与c++20不清楚改动多大. 由于此书交叉引用过多, 所以计划先初步做笔记, 多看几遍后再补充完整

#### chapter 1

`main`函数的返回值必须为`int`

#### chapter 2

2.1 Primitive Build-in Types\(原生内置类型\)

bool/char/wchar\_t/char16\_t/char32\_t/short/int/long/long long/float/double/long double/void

* 数值类型分为两类, 一类是整数类型\(包括bool与char在内\), 另一类是浮点型.
* 注意C++标准没有规定`char`类型是指`unsigned char`还是`signed char`
* 一般情况下, 整数类型默认为`signed`\(char是个例外\).
* C++标准规定有符号整数正负区间应大致平分, 但并未规定其存储形式. 以`signed char`为例, C++标准规定其至少要包含\[-127, 127\]这些整数, 但一般而言, 大多数机器都使用补码形式存储整数, 所以一般取值范围一般为\[-128, 127\]

关于类型使用的一些建议\(略\)

**类型转换**

```text
unsigned int a = -1  // 直接以unsigned int的方式解读补码形式的-1即可(2^32-1)
```

待确认: 整数混合计算时, 按如下方式进行转换\(以最大容量为准\), 例子: 32位无符号整数认为具有32位, 32位有符号整数认为具有31位, 操作数先统一转为较大者再计算

```text
unsigned int a;
int b;
long long c;
a + b;  // a + (unsigned int)(b)
a + c;  // (long long)(a) + c
```

\(Literals\)字面量:

整数/浮点数/字符/字符串/布尔/指针\(void\). 注意`"abc"`的类型是`const char[]`

2.2 变量

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

2.3 Compound Types\(复合类型\)

复合类型: 引用与指针

> a declaration is a base type followed by a list of declarators

例如: `int i = 0; &a = i; *p = &i`, `base type`为`int`, 后面的都叫`declarators`, 引用不是一个变量, 但指针是一个变量. 指针的值可以是: 一个对象的地址; 一个对象的下一个地址; 空指针字面量\(nullptr\); 无效地址

**2.4 const Qualifier\(常量限定符\)**

```text
const int a = 0, *p = &a;//可以将const int合在一起视为base type
```

在默认情况下, const限定的标识符只在本文件中有效, 即假设`a.cpp`中与`b.cpp`中同时用`const int i = 1;`进行了定义, `gcc a.cpp b.cpp -o out`命令编译时不会报错. 但假如没有用`const`限定, 则会出现重定义的错误. 为了使得一个常量被多个文件使用, 建议按如下方式:

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

