# draft

## 0. 杂录

#### 查漏补缺

**1. 同时定义多个变量**

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

**2. 整数与字符串的转换**

C语言中提供函数`atoi`与`itoa`分别进行字符串到整数以及整数到字符串的转换, C++中则使用`std::stoi`与`std::to_string`, 这两个函数在`string`头文件中定义.

**3. auto关键字与容器遍历**

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

例子: 统计字符串中各个字符出现的次数, 重复两次

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

**4. 内建函数**

`__builtin_popcount(n)`: GCC内建函数, 统计`n`的二进制表示中有多少个1

#### 安装gcc\(多版本共存\)

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

#### 安装cmake

[https://cmake.org/install/](https://cmake.org/install/)

#### 关于cmake, make, gcc/cl

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

#### 无题

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

#### C/C++关于整数的一些问题

**计算机存储\(整数\)**

一个数在机器中的表示形式称为"机器数", 其代表的值称为"真值". 所谓表示, 实际上就是制定一套真值与机器数的对应规则, 同时希望真值的运算跟机器数的某种计算方法对应上.

为方便讨论, 假设一个无符号int型的字节数为1\(注意: 采用补码形式的时候, 此时它能表示的数据范围为: \[-128, 127\)共256个数字\).首先讨论带符号数的存储:

| 十进制 | 原码表示 | 反码表示 | 补码表示 |
| :--- | :--- | :--- | :--- |
| 10 | 00001010 | 00001010 | 00001010 |
| -10 | 10001010 | 11110101 | 11110110 |
|  | 00000000 10000000 | 00000000 11111111 | 00000000 1\|00000000 |

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

#### ++

`i++`称为后置自加, `++i`称为前置加加.

* 前置加加效率更高
* 后置加加优先级\(由左到右\)&gt;前置加加优先级\(由右到左\)=解引用优先级\(由右到左\)

**1. 重载++**

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

运算符重载

**`->`的重载**

[https://blog.csdn.net/friendbkf/article/details/45949661](https://blog.csdn.net/friendbkf/article/details/45949661)

**`*`的重载**

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

#### 使用冒号遍历vector

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

```cpp
// C++ unordered_map(hash_map没有列入C++标准)
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

2.面向对象编程

可见性，继承，多态

3.其他

命名空间与作用域

复杂变量类型

## 3. CSAPP

### chapter 2

#### 整型数值表示

将负数$-x$表示为二进制的方式为: 首先用二进制表示出$+x$, 按位取反后加1即可.

反过来, 依据二进制计算其表示的数字的方法为: 首先依据最高位判断符号, 若最高位为$0$, 则直接利用二进制转十进制确定; 若最高位为$1$, 则确定符号为负, 其绝对值由减一后按位取反, 再用二进制转十进制确定.

**练习**

假定`INT`类型为1字节有符号整数, 则数值表示范围为$\[-2^7,2^7-1\]=\[-128,127\]$.

+10 = 00001010 =&gt; 11110101 =&gt; -10 = 11110110 =&gt; 256 + \(-10\) = 246 = 11110110

11110110 =&gt; - ~\(11110101\)=&gt;- 00001010 =&gt; -10

#### 浮点型数值表示

### chapter 3

```c
//mstore.c
long mult2(long, long);
void multstore(long x, long y, long *dest) {
long t = mult2(x, y);
*dest = t;
}
```

```text
$ gcc -Og -S mstore.c
$ gcc -c -Og -o mstore.o mstore.c
```

显示.o文件里的部分机器码:

```text
$ gdb mstore.o
(gdb) x/14xb multstore # 显示机器码0x0 <multstore>:    0x53    0x48    0x89    0xd3    0xe8    0x00    0x00    0x00
0x8 <multstore+8>:    0x00    0x48    0x89    0x03    0x5b    0xc3
```

显示.s文件里的内容:

```text
//mstore.s
    .file    "mstore.c"
    .text
    .globl    multstore
    .type    multstore, @function
multstore:
.LFB0:
    .cfi_startproc
    pushq    %rbx
    .cfi_def_cfa_offset 16
    .cfi_offset 3, -16
    movq    %rdx, %rbx
    call    mult2
    movq    %rax, (%rbx)
    popq    %rbx
    .cfi_def_cfa_offset 8
    ret
    .cfi_endproc
.LFE0:
    .size    multstore, .-multstore
    .ident    "GCC: (GNU) 5.2.0"
    .section    .note.GNU-stack,"",@progbits
```

反汇编mstore.o:

```bash
$ objdump -d mstore.o

mstore.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <multstore>:
   0:    53                       push   %rbx
   1:    48 89 d3                 mov    %rdx,%rbx
   4:    e8 00 00 00 00           callq  9 <multstore+0x9>
   9:    48 89 03                 mov    %rax,(%rbx)
   c:    5b                       pop    %rbx
   d:    c3                       retq
```

## 4. C++Primer\(5ed-en\)阅读笔记--基于c++11标准

备注: c++11标准虽然不是最新的, 但是后续的重要基础, c++14对c++11做了少数修补, c++17与c++20不清楚改动多大. 由于此书交叉引用过多, 所以计划先初步做笔记, 多看几遍后再补充完整

### chapter 1

`main`函数的返回值必须为`int`

### chapter 2

2.1 Primitive Build-in Types\(原生内置类型\)

bool/char/wchar\_t/char16\_t/char32\_t/short/int/long/long long/float/double/long double/void

* 数值类型分为两类, 一类是整数类型\(包括bool与char在内\), 另一类是浮点型. 
* 注意C++标准没有规定`char`类型是指`unsigned char`还是`signed char`
* 一般情况下, 整数类型默认为`signed`\(char是个例外\).
* C++标准规定有符号整数正负区间应大致平分, 但并未规定其存储形式. 以`signed char`为例, C++标准规定其至少要包含\[-127, 127\]这些整数, 但一般而言, 大多数机器都使用补码形式存储整数, 所以一般取值范围一般为\[-128, 127\]

关于类型使用的一些建议\(略\)

#### 类型转换

```cpp
unsigned int a = -1  // 直接以unsigned int的方式解读补码形式的-1即可(2^32-1)
```

待确认: 整数混合计算时, 按如下方式进行转换\(以最大容量为准\), 例子: 32位无符号整数认为具有32位, 32位有符号整数认为具有31位, 操作数先统一转为较大者再计算

```cpp
unsigned int a;
int b;
long long c;
a + b;  // a + (unsigned int)(b)
a + c;  // (long long)(a) + c
```

\(Literals\)字面量:

整数/浮点数/字符/字符串/布尔/指针\(void\). 注意`"abc"`的类型是`const char[]`

2.2 变量

#### 变量初始化写法

以下写法均可

```cpp
int i = 0;//用0.0初始化OK,但会产生截断
int i = {0};//不能用0.0初始化
int i{0};//不能用0.0初始化
int i(0);//用0.0初始化OK,但会产生截断
```

#### 声明与定义

```cpp
extern int i;//变量声明
extern int i = 0;//变量定义, extern不起作用
void foo(int i);//函数声明
```

#### 命名规则与命名规范

> C++ Primer 5ed 2.2.3 Identifies
>
> The standard also reserves a set of names for use in the standard library. The identifiers we define in our own programs may not contain two consecutive underscores, nor can an identifier begin with an underscore followed immediately by an uppercase letter. In addition, identifiers defined outside a function may not begin with an underscore.

C++ Primer中提及到表示符中不要出现连续的两个下划线, 或者以下划线加大写字母开头, 函数外定义的变量不要以下化线开头, 但实际测试时并不会有报错信息

```cpp
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

#### 2.4 const Qualifier\(常量限定符\)

```cpp
const int a = 0, *p = &a;//可以将const int合在一起视为base type
```

在默认情况下, const限定的标识符只在本文件中有效, 即假设`a.cpp`中与`b.cpp`中同时用`const int i = 1;`进行了定义, `gcc a.cpp b.cpp -o out`命令编译时不会报错. 但假如没有用`const`限定, 则会出现重定义的错误. 为了使得一个常量被多个文件使用, 建议按如下方式:

```cpp
extern const int a = foo(); // test.cpp
int b = 1;  // test.cpp
extern const int a; // test.h
extern int b;  // test.h
```

#### top-level const 与 low-level const

top-level const指变量本身不可变, low-level const指与之相关的变量不可变, top-level const变量可以用非const变量赋值, 但low-level const属性必须一致

#### `constexpr`关键字

`const expression`\(常量表达式\)在编译器被计算出来, 字面量是一个常量表达式, 一个用常量表达式初始化的`const`对象是一个常量表达式. `constexpr`关键词被用来提示编译器检查后面的内容必须是常量表达式, 确保其在编译过程中可以计算并计算出来

`decltype`关键字

```cpp
decltype(f()) sum = x;//利用f函数的返回类型得到类型
```

## 5. C++ Primer Plus \(6ed\)笔记

```cpp
int divide(int dividend, int divisor) {
    if (dividend == INT_MIN && divisor == -1) {
        return INT_MAX;
    }
    long dvd = labs(dividend), dvs = labs(divisor), ans = 0;
    int sign = dividend > 0 ^ divisor > 0 ? -1 : 1;
    while (dvd >= dvs) {
        long temp = dvs, m = 1;
        while (temp << 1 <= dvd) {
            temp <<= 1;
            m <<= 1;
        }
        dvd -= temp;
        ans += m;
    }
    return sign * ans;
}


int divide(int A, int B) {
    if (A == INT_MIN && B == -1) return INT_MAX;
    int a = abs(A), b = abs(B), res = 0, x = 0;
    while (a - b >= 0) {
        for (x = 0; a - (b << x << 1) >= 0; x++);
        res += 1 << x;
        a -= b << x;
    }
    return (A > 0) == (B > 0) ? res : -res;
}
```

