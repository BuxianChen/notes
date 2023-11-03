# C++ Primer Plus (6ed)

# 第 9 章: 内存模型与命名空间 (Memory Models and Namespaces)

## 9.1 分块编译 (Separate Compilation)

C++ 代码文件的推荐组织形式

- 头文件（.h）：声明
- 源文件（.cpp）：定义
- 源文件（.cpp）：调用代码

其中头文件里应该存放：

- 函数原型
- 使用 `#define` 或 `const` 定义的常量
- 结构体声明、类声明
- 模板
- 内联函数

# 第 18 章: C++ 11 新特性 (C++ 11 New Features)

本章笔记将对原书进行扩充，主要体现在：

- 补充参考资料：侯捷老师的相关网课等；
- 逐步增加书中没有覆盖到的新特性，并逐步增加 C++14/17/20 的新特性；

## New DataTypes

C++ 11新增加了 `long long`，`unsigned long long` 类型，确保其至少支持 64 位，`char16_t` 与 `char32_t`。

## Uniform Initialization

使用大括号的形式（被称为 *list-initialization*）进行初始化，并且等号可以省略。这一初始化方式对内置类型与用户自定义类型均适用，例如以下初始化方式都是合法的：

```c++
int x = {5};
double y {2.75};
short quar[5] {4, 5, 3, 76, 1};
int *ar = new int [4] {2, 4, 6, 7};  // C++11
class Stump {
private:
    int roots;
    double weight;
public:
    Stump(int r, double w) : roots(r), weight(w) {}
};
Stump s1(3, 15.6);  // old style
Stump s2{5, 43.4};  // C++11
Stump s3 = {4, 32.1};  // C++11
```



