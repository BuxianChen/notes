
# gcc/g++ 编译

**例子1**

目录结构及文件内容

```
main.cc
cheader/
  - MathFunctions.cc
  - MathFunctions.h

// main.cc文件
#include "MathFunctions.h"
```

编译方法

**单条指令编译**
```bash
gcc main.cc cheader/MathFunctions.cc -I cheader/ -o Demo
```
- `-I` 选项用于增加include目录
- `-o` 选项用于指定编译输出的文件位置

**先编译链接库，再利用链接库编译应用**
```bash
# 编译动态链接库: 两种都可以
gcc -shared -o cheader/libMathFunctions.so -fPIC cheader/MathFunctions.cc
# cd cheader && gcc -shared -o libMathFunctions.so -fPIC MathFunctions.cc && cd ..

# 使用动态链接库
# 方式一：此处main.cc与cheader/libMathFunctions.so顺序不能乱
gcc -I cheader -o Demo main.cc cheader/libMathFunctions.so
# 方式二：(待修改)可以找到链接库但找不到符号?
gcc -I cheader -o Demo -Lcheader -l MathFunctions main.cc
```
- `-L`用于增加静态/动态链接库的搜索路径，`-l`用于指定静态/动态链接库的具体名称(注意不含lib前缀及文件扩展名)。[stackoverflow问答的解释](https://stackoverflow.com/questions/71544910/usr-bin-ld-cannot-find-lname-of-the-library-while-compiling-with-gcc)

- 可以使用 gcc 的组件 nm 命令查看 .so 文件中的符号。[stackoverflow](https://stackoverflow.com/questions/43256459/g-undefined-reference-although-symbol-is-present-in-so-file)
  ```bash
  nm --demangle --defined-only --extern-only cheader/libMathFunctions.so
  # 或
  nm -D cheader/libMathFunctions.so
  ```


# Make

# CMake
资源：
- 官方Tutorial：[链接](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- cmake-demo：[Gitbook](https://www.hahack.com/codes/cmake/)，[Github](https://github.com/wzpan/cmake-demo)
- cmake-examples：[Github](https://github.com/ttroy50/cmake-examples)