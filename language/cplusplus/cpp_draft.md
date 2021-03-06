# cpp\_draft



## 0. 编译

### 安装gcc\(多版本共存\)

参考博客: [Linux下编译安装GCC 4.9.4 - Caosiyang's Blog](https://caosiyang.github.io/posts/2016/05/04/installing-gcc/)

```text
# 下载gcc源码解压后的目录假定为gcc-4.9.4
cd gcc-4.9.4
sh ./contrib/download_prerequisites
cd ..
mkdir build-gcc-4.9.4
cd build-gcc-4.9.4
# gcc-4.9.4将会安装至/usr/local/gcc-4.9.4/目录下
../gcc-4.9.4/configure --prefix=/usr/local/gcc-4.9.4/ --enable-checking=release --enable-languages=c,c++ --disable-multilib
make -j4
make install
```

### 安装cmake

[https://cmake.org/install/](https://cmake.org/install/)

### 关于cmake, make, gcc/cl

cmake与make均有windows版本

三者关系参考: [GCC/Make/CMake 之 GCC - 知乎 \(zhihu.com\)](https://zhuanlan.zhihu.com/p/342151242)

cmake用于生成makefile

`CMakeLists.txt` -- `cmake` -&gt; `Makefile` -- `make/nmake` -&gt; `gcc/cl` -&gt; `.o/.exe/.dll/.so/.lib/.a`

cmake基于`CMakeLists.txt`文件，写法参考[GitHub - ttroy50/cmake-examples: Useful CMake Examples](https://github.com/ttroy50/cmake-examples)上的例子及官方文档

```text
mkdir build
cd build
cmake .. -G "MinGW Makefiles"  # 用gcc进行编译, 在当前目录生成一个Makefile
make
./xx.exe
```

### C/C++程序的编译\(待补充与修改\)

[参考博客](https://www.cnblogs.com/alexdeblog/p/3240343.html)

[GCC编译器30分钟入门教程 \(biancheng.net\)](http://c.biancheng.net/gcc/)

C或C++文件编译的过程为:

* 编译预处理\(注意只编译`.c`或`.cpp`文件, `.h`文件只用于预处理时的复制粘贴\), 得到object文件\(linux中是`.o`文件\), 这个过程可以有没有定义的部分\(但要有声明\).
* 链接\(生成可执行文件或库文件, 前者需要`main`函数, 后者不需要\), 分为动态链接与静态链接, windows中的动态链接库文件为`.dll`, linux中为`.so`，它们可以被其他程序在运行时被加载进来.

下面以`gcc`来介绍

```cpp
#include<iostream>
int Double(int x){
    return x * 2;
}

int main(){
    std::cout << "succeed" << std::endl;
    std::cout << Double(1) << std::endl;
    return 0;
}
```

默认情况下生成`a.out`文件\(可执行文件\), 可使用`-o`选项指定生成的文件名

```text
g++ add.cpp
./a.out
g++ add.app -o add.out
./add.out
```

使用`-c`选项对编译过程进行控制

```c
// foo.h
#ifndef __FOO_H__
#define __FOO_H__
void print_msg();
#endif
// foo.c
#include <stdio.h>
#include "foo.h"
void print_msg(){printf("Hello\n");}
// test.c
#include "foo.h"
int main(){
    print_msg();
    return 0;
}
```

记住`-E`选项表示进行到编译预处理为止\(输出为文本文件, linux上以`.i`作为后缀\), `-S`选项表示进行到汇编代码为止\(输出为文本文件, linux上以`.s`作为后缀名\), `-c`选项表示到进行到目标机器码为止

```bash
gcc -c foo.c -o foo.o  # 注意不需要加.h文件, 会被自动复制粘贴, 生成文件名结尾为.o, 不是可执行文件
gcc -c test.c -o test.o
# 可能不是静态链接(待确认): gcc test.o foo.o -o test.out  # 静态链接生成可执行文件
# ar rcs libmystatic.a foo.o  # linux生成静态链接接库
# gcc -static test.o libmystatic.a  #使用静态链接的方式生成可执行文件
./test.out  # 运行代码
# 以上也可直接写为:
# gcc -c foo.c test.c -o test.out
# 或者:
# gcc -c foo.c -o foo.o
# gcc test.c foo.o -o test.out

rm *.o *.out
```

**动态链接**

备注: `shared Object`是linux上的叫法, `dynamic linked library`是windows上的叫法, 两者在概念上说是一致的.

shared表示指定编译器生成shared library, -fPIC选项中PIC是Position-Independent Code的缩写, 意思是生成的机器代码不依赖于某个绝对内存地址. 因为shared library可能被多个程序使用, 若是用绝对地址的话不同的程序使用过程中很可能会产生冲突, 所以使用-fPIC选项, 使得生成的代码适合作为shared library使用. gcc/g++有不少选项是-f开头的, 这些选项常常是一些小功能的开关

```bash
gcc -shared -fPIC foo.c -o libfoo.so  # 生成shared object
gcc test.c libfoo.so -o test.out  # 动态链接生成可执行文件
# 此时如果将libfoo.so移动位置, test.out将不能被运行
./test.out
```

更通用的情况

test.c与foo.c和foo.h分别放在两个不同的路径中，设test.c的路径为/u/dev/test/test.c，foo.c\(h\)的路径为/u/dev/foo/foo.c\(h\)。通常库和调用库的程序是不同时写的，放在不同的地方也是最通常的情况。首先到/u/dev/foo/里面去编译libfoo.so，所用的命令同上面的一样。然后回到/u/dev/test/，来编译test。首先第一个问题就是到哪里去找foo.h？这需要用到`-I`选项，它指定了include文件的路径。于是我们可以用这个命令

```bash
# cd /u/dev/foo
# gcc -shared -fPIC foo.c -o libfoo.so
cd /u/dev/test
gcc -I/u/dev/foo/ test.c /u/dev/foo/libfoo.so -o test
```

往往库文件会集中放在一个路径中，要指定所有库文件的完整路径非常费事，gcc提供了一个`-L`选项，用来指定库文件的路径，结合`-l`（小写L）来使用，指定用哪一些库文件。用在这个例子上，完整的命令就变为

```bash
# -l选项后跟的是库文件的名字，注意gcc自动把文件名的lib前缀去掉了，所以-lfoo实际上用的是libfoo.so，如果文件名是foo.so则反而找不到。若/u/dev/foo中有多个库文件需要用，那么使用-lfoo1 -lfoo2就比用完整路径简单多了。据说在某些新版本的gcc中，-l选项要放到-o选项的后面，按照依赖关系来排，不过我还暂时没有遇到这个问题
gcc -I/u/dev/foo -L/u/dev/foo -lfoo test.c -o test.out
```

接下来运行test文件时仍然不能正常运行，这是因为找不到foo库文件的缘故。一个（可能不太好但可以用的）解决的办法是使用LD\_LIBRARY\_PATH环境变量，这个环境变量指定了shared library的路径。在运行test之前，更新这个环境变量

```bash
export LD_LIBRARY_PATH=/u/dev/foo:$LD_LIBRARY_PATH
```

然后就可以正常执行

```text
./test.out
```

在通常的开发过程中，编译好的.so文件一般放在名为lib的文件夹中，.h文件一般放在include文件夹中，linux系统本身也如此。在使用这些库的时候，编译时用-I, -L和-l选项即可。将上述环境变量的更新放在.bashrc中，便可以省去每次运行前都指定库文件目录的麻烦了



## 1. C++基本使用

### 1.1 命名规范与项目结构

| 类别 | 命名方式 |
| :--- | :--- |
| 目录与文件名 | 全小写加下划线 |
| namespace | 全小写加下划线 |
| 类/结构体/typedef/模板参数/枚举类型名 | 大写字母开头的驼峰式命名，例如：MyString |
| 函数名 | 小写字母开头的驼峰式命名，例如：printList |
| 变量名 | 小写字母开头的驼峰式命名，例如：myString |
| 枚举类型内部元素/宏定义 | 全大写加下划线 |

### 1.2 基础语法及原理查漏补缺

#### lambda

```cpp
#include<iostream>

int main() {
    int id = 23;
    auto func = [id]() mutable {
        std::cout << "inside lambda, id = " << id << std::endl;
        ++id;
    };
    
	id = 0;
    func(); // 23
    std::cout << "outside lambda, id = " << id << std::endl;  // 0
    func();  // 24
    std::cout << "outside lambda, id = " << id << std::endl;  // 0
    return 0;
}
```

上述 `lambda` 函数**基本**等价于

```cpp
// 在其他位置定义
class Func{
private:
	int id;  // mutable表示给了修改权限
public:
    Func(int id) {this->id = id;}  // [id]为[=id]的简写
    // 如果改为[&id], 则相当于将此处改为
    // Func(int &id) {this->id = id;}
    // 那么外部的id也将被修改
	void operator() (){
        std::cout << "inside lambda, id = " << id << std::endl;
        ++id;
    }
}

// 这一行相当于那行lambda的定义
func = Func(id);  // id的值为23, 由于传递方式为传值, 所以后续的修改不影响外部的id
```

备注：

- 如下两种写法均会报错

  ```cpp
  int id = 23;
  auto func = [id]() {...};
  // error: increment of read-only variable ‘id’
  
  const int 23;
  auto func = [&id]() mutable {...};
  // error: increment of read-only reference ‘id’
  ```

  

#### 通过指针修改某块内存的值

下面的程序段演示了数组的诸多问题，这些都没有很好的解决方案，用C++有时就是这么麻烦，可能还需要查看以下别的包例如opencv，gmp加深体会

```cpp
//test.cpp
#include<iostream>
using namespace std;
//深复制操作
//此函数有诸多可以讨论之处，其一：数组b必须写在参数列表中，这导致main函数中必须事先给b分配内存；其二：必须指定数组长度，这也是无法避免的。
//假设修改为链表形式，也必须将整型指针b在main函数中事先定义，b也必须写在参数列表中，但n可以去掉。
//但是链表在某些情况下不如数组方便使用，例如排序。
//vector类是一个比较好的替代方式，之后再研究。
void f(int *a, int *b, int n)
{
    for(int i = 0; i < n; i++)
        b[i] = a[i];
}
int main()
{
    const int n = 4;
    int a[n] = {1,2,5,7};
    int b[n] = {0};
    f(a,b,n);
    for(int i = 0; i < n; i++)
        cout << b[i] << '\t';
    cout << endl;
    return 0;
}
```

#### 关于重载

```cpp
#include <iostream>
using namespace std;
long foo(long a) { cout << "long" << endl; return a; }
int foo(int a) { cout << "int" << endl; return a; }  // 此方法被调用
int main(){
    short y = 1;
    foo(y);
    system("pause");
    return 0;
}
```

注意: 似乎自动转为选择容量小的重载形式进行执行, 与定义或声明的顺序无关.

#### 关于枚举类型与宏定义

```cpp
enum AlternateUrlTableErrors  // 大写开头驼峰式
{
    OK = 0,
    OUT_OF_MEMORY = 1,  // 全大写加下划线
    MALFORMED_INPUT = 2
};
//宏命名
#define ADD_ONE(x) ((x) + 1)
#define PI_ROUNDED 3.0
```

#### 指针与数组

两个主要的区别：

* 数组名可以近似看作是常量指针
* sizeof的运算结果不同

```cpp
// 指针不是地址，而是类型为地址的变量
// 数组名存放在符号表里
int a[2] = { 1, 2 };
int *p = &a[0];
sizeof(a);  // 2*sizeof(int)/*=8*/
sizeof(p);  // sizeof(int*)/*=4*/
// &a 在C语言里是未定义的行为，许多编译器对此返回的是一个地址（而非指针）
// &1 是非法的
```

数组名只能作为右值而不能作为左值。

```text
char *p = "asd"; // 报错
char a[] = "asd"; // 正常
//typeid("asd").name()的输出结果为char const [4]
```

#### 函数声明时展示抛出的异常类型

```
// 返回类型 函数名(形参列表) throw() {函数体}
void check(void) throw (std::out_of_range);
```



#### 运算符优先级

[C++运算优先级\(cpp referrence\)](https://zh.cppreference.com/w/cpp/language/operator_precedence%20)

简单记忆\(按优先级从高到低, 大多数结合性都是从左到右\)

其他&gt;算术&gt;移位&gt;比较&gt;按位逻辑&gt;逻辑&gt;三目, 赋值&gt;逗号

其他

* ::
* a++  a--  type\(\)  a\(\)  a\[\]  .  -&gt;
* \(从右到左\)++a  --a  +a  -a  !  ~  \(type\)  \*  &  sizeof  new  delete
* .  _-&gt;_
* \*  /  %
* +  -
* &gt;&gt;  &lt;&lt;
* &lt;=  &lt;  &gt;=  &gt;
* ==  !=
* &
* ^
* \|
* &&
* \|\|
* \(从右到左\)  a?b:c  {运算符}=
* ,

#### range based for

**例子1：基础用法**

```cpp
// C++ 11 特性
#include <iostream>
#include <vector>

int main()
{
    // 创建含有整数的 vector
    std::vector<int> v = { 7, 5, 16, 8 };

    // 添加二个整数到 vector
    v.push_back(25);
    v.push_back(13);

    // 迭代并打印 vector 的值
    // 如果不加&, 那么它是只读的, 不会对原数组修改
    for (int &n : v) {
        n = n + 1;
        std::cout << n << '\n';
    }
    for (int n : v) {
        std::cout << n << '\n';
    }
    system("pause");
    return 0;
}
```

备注: 到目前为止, `for (auto i: collection)`的写法不支持从容器的第2个元素开始遍历的写法

**最佳实践**:

* 在可能的情况下, 使用`auto`+`range based for loop`, 省事并且显得专业, 但前提是知道它实际上是什么

  ```cpp
  for (auto i: vec) {...}
  for (auto const &i: vec) {...}
  
  for (auto &[x, y]: map1) {...}
  ```

* 如果不能用上述方式, 优先使用迭代器

  ```cpp
  for (auto item=map1.begin()+1; item!=map1.end(); item++) {
  // for (map<int, int>::iterator item=map1.begin(); item!=map1.end();item++) {
      item -> first; // 不能修改
      item -> second;  // 可以修改
      ...
  }
  ```

* 最次, 对于可以使用下标进行索引的容器, 可以考虑下标遍历

  ```cpp
  for (int i = 0; i < vec.size(); i++) {...}
  ```

详细介绍:

```cpp
for (auto &[x, y]: map1) {...}
// 等价于
for (map<int, int>::iterator item=map1.begin(); item!=map1.end();item++) {
    const int &x = item -> first;
    int &y = item -> second;
    ...
}
// 等价于
for (pair<const int, int> p: map1) {
    int const &x = p.first;
    int &y = p.second;
}
```

**例子2：统计字符串中各个字符出现的次数, 重复两次**

```cpp
string compressString(string S) {
    unordered_map<char, int> count;
    // 注意: 这种range based for loop语法迭代的每个元素不是迭代器, 而是类型本身
    for (char c: S) {
        count[c]++;
    }
    string result = "";
    // 注意此处的写法有两种:
    // (1)这种写法返回的p是一个引用, 但是不能修改p.first
    // pair<const char, int> &p: count
    // (2)这种写法返回的不是引用, 修改p不会影响容器中的数据, 应避免使用
    // pair<char, int> p: count
    for (pair<const char, int> &p:count) {
        result.push_back(p.first);
        result.append(to_string(p.second+1));
    }
    // 这种写法返回的是常引用, 最好使用这种写法替代写法(2)
    for (auto const &[x, y]:count) {
        result.push_back(x);
        result.append(to_string(y));
    }
    cout << result << endl;
    return result;
}
```



#### 多级指针

```c
//允许多级指针, 但一般最多只使用到二级指针
#include <stdio.h>
int main(){
    int a =100;
    int *p1 = &a;
    int **p2 = &p1;
    int ***p3 = &p2;
    printf("%d, %d, %d, %d\n", a, *p1, **p2, ***p3);
    printf("&p2 = %#X, p3 = %#X\n", &p2, p3);
    printf("&p1 = %#X, p2 = %#X, *p3 = %#X\n", &p1, p2, *p3);
    printf(" &a = %#X, p1 = %#X, *p2 = %#X, **p3 = %#X\n", &a, p1, *p2, **p3);
    return 0;
}
/*运行结果:
100, 100, 100, 100
&p2 = 0X28FF3C, p3 = 0X28FF3C
&p1 = 0X28FF40, p2 = 0X28FF40, *p3 = 0X28FF40
 &a = 0X28FF44, p1 = 0X28FF44, *p2 = 0X28FF44, **p3 = 0X28FF44
*/
```

#### 类\(模板\)成员函数带有其他模板参数\(member template\)

```cpp
namespace test01{
#include<iostream>
    template<typename T>
    class Foo{
    public:
        T data;
        template<typename T1>
        void print(T1 a);
    };

    template<typename T>  //类模板参数必须在前
    template<typename T1>
    void Foo<T>::print(T1 a){std::cout << "print " << typeid(data).name() <<
        " " << typeid(a).name() << std::endl;}

    void test_main(){Foo<float> foo;foo.print(2L);}
}

int main(){test01::test_main();return 0;} //输出: print f l
```

#### 模板特化

函数模板不能特化?[https://www.fluentcpp.com/2017/08/15/function-templates-partial-specialization-cpp/](https://www.fluentcpp.com/2017/08/15/function-templates-partial-specialization-cpp/)

#### 运算类型隐式转换

问题来源于在leetcode上刷题时, 如下一行代码报错:

```cpp
long long w = (1 << 32) - 1;
// runtime error: shift exponent 32 is too large for 32-bit type 'int'
```

原因在于右侧表达式参与运算的数都是int类型, 所以在计算`1 << 32`时会报错, 改正方法是

```cpp
long long w = (1LL << 32) - 1;
```

#### 静态成员变量

```c++
class A{
public:
    static int a; //注意：此处只是声明，并未分配空间
};
int A::a; // 此处为静态成员变量的定义，即分配空间
// 也可以使用 int A::a=1; 表示定义并同时初始化。
```

#### typedef 与 using

两者都用于为类取别名，C++ 11 推荐使用 using

```c++
typedef unsigned char PixelType;
using PixelType = unsigned char;
```

using 作用于命名空间，相当于对命名空间里的所有标识符取别名（待确认）

```c++
using namespace std;
// 既可以用std::cout, 也可以直接使用endl。
std::cout << "abc" << endl;
```

#### uniform initialization（待补充）

这是 C++ 11 新特性，之前的初始化的写法有：

```c++
int i = 1;
A a = {1, 2}; // 结构体
vector<int> a({1, 2, 3});
B b(1, 2, 3); // 类的构造函数
```

C++ 11 引入了一种一致性的初始化写法：

```
vector<int> arr({1, 2, 3});
```

其背后原理如下，编译器在编译阶段，将 `{1, 2, 3}` 自动转换为：

```c++
std::array<int> a(3) = {1, 2, 3};
initializer_list<int> init_list=(a.begin(), 3)
// 因此上述初始化最终转化为: vector<int> arr(init_list)
// 即调用了如下构造函数
template<int>
vector<int> (initializer_list<int>);
```

#### explicit

C++ 11 之前，只适用于单参数构造函数（或者是只有一个参数没有默认值的构造函数），用于防止编译器做隐式的类型转换：

```c++
class Complex {
public:
    explicit Complex(int i, int j=0):i(i), j(j){}
    Complex operator+(const Complex& other);
private:
    int i, j;
};
Complex a(1, 2);
Complex b = a + 1;  // 报错, explicit阻止了编译器进行隐式转换
```

C++ 11 后，此关键字也可使用于多个参数的构造函数。

#### const 与 extern

在多文件链接时，变量具有 external linkage，函数具有 external linkage，而常量具有 internal linkage 的特性。external linkage（外部链接）指的是在链接时，同一个标识符在多个文件中只能被定义一次，而 internal linkage （内部链接）指的是在链接时，该标识符在只存在于该编译单元中。

对于变量而言，声明/定义有如下形式

```c++
extern int x = 1;  // (1)
int x = 1;  // (2)
int x;  // (3)
extern int x;  // (4)
```

其中 (1) 与 (2) 是完全等价的，表示定义变量并赋初值。(3) 表示只定义变量。(4) 表示变量声明。

对于常量而言，声明/定义有如下形式

```c++
extern const int x = 1;  // (1)
const int x = 1;  // (2)
extern const int x;  // (3)
```

其中 (1) 表示定义常量，并且该常量的作用范围为全局。(2) 表示定义常量，该常量只在本编译单元中有效，多文件编译时其他编译单元也可以定义该标识符。(3) 表示声明该常量已在别处定义。

因此，为了使得编译通过，对于变量而言，只能有一个文件进行变量定义，其他文件不能重复使用这个变量名。即

```c++
// file1.cpp
// 以下三者均为定义
extern int x = 1;
// 或者以下两者均可
int x = 1;
int x;

// file2.cpp
extern int x;  // 声明

// file3.cpp
extern int x;  // 声明
```

而对于常量而言

- 定义多个“局部”常量

  ```c++
  // file1.cpp
  const int x = 1;  // 定义, 仅在file1.cpp中有效
  
  // file2.cpp
  const int x = 2;  // 定义, 仅在file2.cpp中有效
  
  // file3.cpp
  const int x = 3;
  ```

- 使用“共同”的常量

  ```c++
  // file1.cpp
  extern const int x = 1;  // 定义, 三个文件中的x为同一个
  
  // file2.cpp
  extern const int x;  // 声明
  
  // file3.cpp
  extern const int x;  // 声明
  ```

- 特殊情况：覆盖

  ```c++
  // file1.cpp
  extern const int x = 1;  // 定义
  
  // file2.cpp
  const int x = 2;  // 定义, 本文件中
  ```

对于函数而言，定义与声明的方式分别为

```c++
int foo(int a, int b){return a+b;} // 定义, 具有外部链接性
int foo(int, int);  // 声明
int foo(int a, int b); // 声明
```

因此，函数定义只能在多个文件出现一次，函数声明可以出现若干次。

例子：

```c++
// file1.cpp
extern const int a = 1;

int foo1() {
    return a;
}

// file2.cpp
#include<iostream>
const int a = 2;

int foo1();
void foo2(){
    std::cout << "file2:" << a << std::endl;
    std::cout << "file1::a + file2::a = " << (foo1() + a) << std::endl;
}

int main(){
    std::cout << "file1:" << foo1() << std::endl;
    foo2();
    return 0;
}

// g++ file1.cpp file2.cpp -o main && ./main
/* 输出结果为
file1:1
file2:2
file1::a + file2::a = 3
*/
```

#### 同时定义多个“相似”类型变量

结论: 在同一条语句里, 非const与const不能混用, 唯一的例外是常量指针

**最佳实践**:

* 避免const变量与非const用一条语句定义
* 避免用一条语句中前面的变量来计算/初始化后面的变量

关于使用一条语句定义变量的问题, 目前为止, 这种语法是可以的\(不确定用a和b对p与r初始化是否是C++标准\)

```cpp
// 常量指针(本身不可变)是可以的
int a = 1, b = 100, *p = &a, * const r = &b;
```

这种定义方式后三个不行

```cpp
// 指向常量的指针, 常量引用, 常量数据都会报错
int a = 1, const b = 100, const *p = &a, const &c=a;
```

#### 整数与字符串的转换

C语言中提供函数`atoi`与`itoa`分别进行字符串到整数以及整数到字符串的转换, C++中则使用`std::stoi`与`std::to_string`, 这两个函数在`string`头文件中定义.

#### 内建函数

`__builtin_popcount(n)`: GCC内建函数, 统计`n`的二进制表示中有多少个1

#### C/C++整数存储/自动转换/溢出

**计算机存储\(整数\)**

一个数在机器中的表示形式称为"机器数", 其代表的值称为"真值". 所谓表示, 实际上就是制定一套真值与机器数的对应规则, 同时希望真值的运算跟机器数的某种计算方法对应上.

为方便讨论, 假设一个无符号int型的字节数为1\(注意: 采用补码形式的时候, 此时它能表示的数据范围为: \[-128, 127\)共256个数字\).首先讨论带符号数的存储:

| 十进制 | 原码表示          | 反码表示          | 补码表示             |
| :----- | :---------------- | :---------------- | :------------------- |
| 10     | 00001010          | 00001010          | 00001010             |
| -10    | 10001010          | 11110101          | 11110110             |
|        | 00000000 10000000 | 00000000 11111111 | 00000000 1\|00000000 |

原码表示易于理解: 即最高位用于存储符号\(0表示整数, 1表示\), 其余位置为二进制表示. 注意0有两种表示.

反码表示: 正数与原码相同, 负数在原码的基础上, 保留最高位不变, 其余位按位取反.

补码表示\(计算机的实际存储方式\): 整数与原码相同, 负数在补码的基础上加1.

为何要采取补码表示? 首先要从计算机的运算说起, 任何运算都必然是取模运算\(例如上述的两个整型数相加实际上是模256=2^8进行的\). 结果我们发现补码表示的10与-10相加刚好是256\(也就是0\). 所以20-10=00010100+11110110=1\|00001010. \(可以进行数学上的证明, 此处不提, 但注意如果是123+23会超过127得到的会是一个负数, 原因是我们实际上不能表示146这个数\).

接下来, 无符号型整数实际上就是没有符号位的整数, 于是表示范围变成了\[0, 255\].

注: 实际上, 可以将负数赋值给无符号整型, 实际上机器数是不变的. 即unsigned a = -10, 即a = 11110110, 但输出a会变成一个正数. 另外注意输出函数实际上是将机器数翻译为10进制数, 其翻译准则由函数所定义.

特别说明: \|,&,^,&lt;&lt;,&gt;&gt;都是直接对补码所有位\(包括最高位\)进行的. 特别说明右移的时候, 左边补的数字由原来数字的最高位决定.

**隐式转换**

```cpp
int temp1 = 1;
unsigned int temp2 = 1;
std::cout << typeid(temp1 + temp2).name() << std::endl;// unsigned int
```

```cpp
char temp3 = 129  // signed
std::cout << typeid(temp3 + temp3).name() << std::endl;  // int
int temp5 = -1;
unsigned int temp6 = temp5;
unsigned long long temp7 = temp5;
//4294967295      18446744073709551615
std::cout << temp6 << "\t" << temp7 << std::endl;
```

**C语言溢出判断**

乘法溢出最简单的办法是

```cpp
long long a = 0;
int b = 1000000, c = 1000000;
a = long long(b) * long long(c);
if (a != int(a))
    printf("overflow\n");
```

C语言有符号整数溢出判断

引用自[stackoverflow](https://stackoverflow.com/questions/3944505/detecting-signed-overflow-in-c-c)

> according to the C standard, signed integer overflow is _undefined behavior._ So you can't cause undefined behavior, and then try to detect the overflow after the fact.

C语言无符号整数溢出判断

一些疑惑：

```cpp
// 下面的{1, 2, 3}到底是什么?涉及到explicit关键词的问题
int a[3] = {1, 2, 3};
vector<int> a({1, 2, 3});

// 注: 函数对象指的是重载了()的类的实例, 即 Fun fun(...); fun();
priority_queue<T, vector<T>, cmp> Q; //这里的cmp一定要是函数对象, 但原本的默认值less与operator <的关系是怎么回事

priority_queue<T> Q; //这种写法需要重载<运算符, 但似乎要严格地写为如下形式
// bool operator < (const T &other) const;
```

#### 自加运算符：++（待补充）

`i++`称为后置自加, `++i`称为前置加加.

* 前置加加效率更高
* 后置加加优先级\(由左到右\)&gt;前置加加优先级\(由右到左\)=解引用优先级\(由右到左\)

**重载++**

当操作数 `i` 为重载时

在运算符重载时, 以STL中`list<T>::iterator`为例:

```cpp
//GNU C++ 2.9.1

//节点定义
template<class T>
struct __list_node{
    typedef void* void_pointer;
    void_pointer prev;
    void_pointer next;
    T data;
};

//链表
template<class T, class Alloc=alloc>
class list{
protected:
    typedef __list_node<T> list_node;
public:
    typedef list_node* link_type;
    typedef __list_iterator<T, T&, T*> iterator;
protected:
    link_type node; //即:__list_node<T>* node, 虚拟节点, 指向最后一个元素的下一个元素
//...
}

//list::iterator定义
template<class T, class Ref, class Ptr>
struct __list_iterator{
    typedef __list_iterator<T, Ref, Ptr> self;
    typedef bidirectional_iterator_tag iterator_category;  //(1)
    typedef T value_type;    //(2)
    typedef Ptr pointer;     //(3), 一般为T*
    typedef Ref reference;   //(4), 一般为T&
    typedef __list_node<T>* link_type;
    typedef ptrdiff_t difference_type;  //(5)

    link_type node;

    reference operator*() const {return (*node).data;}
    pointer operator->() const {return &(operator*());}
    self& operator++() {node=(link_type)((*node).next); return *this;} //前置++重载
    self operator++(int) {self tmp=*this; ++*this; return tmp;} //后置++重载
    //...
};
```

#### 运算符重载（待补充）

**特殊情况：`->`的重载**

[https://blog.csdn.net/friendbkf/article/details/45949661](https://blog.csdn.net/friendbkf/article/details/45949661)

**特殊情况：`*`的重载**

## 2. C++标准库

[https://blog.csdn.net/lyh03601/column/info/geek-stl](https://blog.csdn.net/lyh03601/column/info/geek-stl)

vector,string,algorithm,iterator

迭代器是一种检查容器内元素并遍历元素的数据类型。C++更趋向于使用迭代器而不是下标操作，因为标准库为每一种标准容器（如vector）定义了一种迭代器类型，而只用少数容器（如vector）支持下标操作访问容器元素。（链接：[C++迭代器](https://www.cnblogs.com/maluning/p/8570717.html)）

示例代码

```cpp
#include<iostream>
#include<string>
#include<vector>
using namespace std;
int main()
{
    //动态创建一维int型vector
    //实例输入：
    //4
    //5 2 1 9
    vector<int> a;
    int num;
    cin >> num;
    for(int i = 0; i < num; i++)
    {
        int temp;
        cin >> temp;
        a.push_back(temp);//末尾追加
    }

    cout << a.size() << endl;//元素个数
    //迭代器
    //迭代器操作类似指针，注意a.begin()为第一个元素的位置，a.end()为最后一个元素位置加1。
    vector<int>::iterator it = a.begin();
    for (; it != a.end(); it++) 
    {
        cout << *it << ' ';
    }
    cout << endl;
    system("pause");
    return 0;
}
```

以下是对链接中的部分注解：

**size\_t类型**

`size_t`类型：建议下标类型为size\_t。即：

```cpp
#include<iostream>
using namespace std;
int main()
{
    int a[4] = {1,3,1,7};
    //int i = 2;
    size_t i = 2;
    cout << a[i] << endl;
    cout << sizeof(i)
    return 0;
}
```

那么实际上，`size_t`类型占多少内存呢？似乎与编译器生成的程序的位数有关，32或者64位，分别对应4字节与8字节。更多可参考：[https://blog.csdn.net/Richard\_\_Ting/article/details/79433814](https://blog.csdn.net/Richard__Ting/article/details/79433814)

```cpp
//注意sizeof()可以查看变量类型或变量所占的内存
//注意sizeof()的返回类型实际上正是size_t
//建议这样使用size_t：用在表示字节大小或数组索引的地方
#include<iostream>
using namespace std;
int main() {
    int arr[3] = { 1, 5, 2 };
    int *p = arr;
    size_t i = 1;
    cout << arr[i] << endl;
    cout << "sizeof(int)=" << sizeof(int) << "\t"
        << "sizeof(size_t)=" << sizeof(size_t) << "\t"
        << "sizeof(i)=" << sizeof(i) << "\t"
        << "sizeof(arr)=" << sizeof(arr) << "\t"
        << "sizeof(p)=" << sizeof(p) << endl;
    system("pause");
    return 0;
}
/*输出结果：调试器选择x86（VS2019）
5
sizeof(int)=4   sizeof(size_t)=4        sizeof(i)=4     sizeof(arr)=12  sizeof(p)=4
输出结果：调试器选择x64（VS2019）
5
sizeof(int)=4   sizeof(size_t)=8        sizeof(i)=8     sizeof(arr)=12  sizeof(p)=8
*/
```

```cpp
//字节对齐，机理待补充
/*顺带汇总下x86（VS2019）下的一些数据类型所占内存（不知道是否与调试器，机器位数有关）
int,unsigned int,long,float        4字节
double,long long                8字节
*/
#include<iostream>
using namespace std;
int main() {
    bool flag = true;
    Test t(flag, 1, 1.3);
    cout << "sizeof(flag)=" << sizeof(flag) << "\t"
        << "sizeof(int)=" << sizeof(t.x) << "\t"
        << "sizeof(float)=" << sizeof(t.y) << "\t"
        << "sizeof(Test)=" << sizeof(Test) << "\t"
        << "sizeof(t)=" << sizeof(t) << endl;
    system("pause");
    return 0;
}
/*输出结果：调试器选择x86（VS2019）（以后的默认选项）
sizeof(flag)=1  sizeof(int)=4   sizeof(float)=4 sizeof(Test)=12 sizeof(t)=12
输出结果：调试器选择x64（VS2019）
sizeof(flag)=1  sizeof(int)=4   sizeof(float)=4 sizeof(Test)=12 sizeof(t)=12
*/
```

### 2.1 容器与迭代器

#### vector，stack，queue

#### string

[关于string使用方法的野博客](https://www.cnblogs.com/xFreedom/archive/2011/05/16/2048037.html)

_**提示：string，cstring，string.h**_

```cpp
# include<string.h>  // c语言里的头文件, 不可以定义string s, 可以使用strcpy等函数
# include<cstring>  // c++语言里的头文件, 不可以定义string s, 可以使用strcpy等函数
# include<string>  // c++中STL库里的头文件, 可以定义string s, 可以使用strcpy等函数
```

_可以简单认为cstring是C++里对string.h的一层包装_

```cpp
namespace std
{
    #include<string.h>
    ...
}
```

`std::string`的常见操作

```cpp
// 初始化
string s1 = "aaa";

// 添加字符
s1.push_back('a');
```

#### 哈希表，集合，字典

### 2.2 算法、仿函数、lambda

实现 `numpy` 的 `argsort` 函数

```cpp
//argsort.cpp
#include<vector>
#include<random>
#include<iostream>
#include<algorithm>

template<typename T>
class Compare {
private:
    std::vector<T> arr;
public:
    Compare(const std::vector<T> &arr) {this->arr = arr;}
    bool operator()(int v1, int v2) { return arr[v1] < arr[v2]; }
};

template<typename T> std::vector<int> argsort(const std::vector<T>& array) {
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    std::vector<int> numpy_index(array_len, 0);
    for (int i = 0; i < array_len; i++) {array_index[i] = i;}
    auto func = [&array](int pos1, int pos2) {return (array[pos1] < array[pos2]);};
    // std::sort(array_index.begin(), array_index.end(), Compare<(array)); // 可以用仿函数实现
    std::sort(array_index.begin(), array_index.end(), func);  // 用c++11新特性lambda实现
    for (int i = 0; i < array_len; i++) {numpy_index[array_index[i]] = i;}
    //return array_index; // array_index[0] = 6 表示最小的数为array[6]
    return numpy_index;  // numpy_index[0] = 1 表示array[0]为第二小的数
}

int main() {
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 20);
    std::vector<int> vec_data;
    for (int i = 0; i < 10; ++i)
        vec_data.push_back(distribution(generator));
    for (int item : vec_data)
        std::cout << item << "\t";
    std::cout << std::endl;
    std::vector<int> index = argsort(vec_data);
    for (int item: index)
        std::cout << item << "\t";
    std::cout << std::endl;
    return 0;
}
```



