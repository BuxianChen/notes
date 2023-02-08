
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

## 编译步骤、动态链接库与静态链接库

编译步骤拆解为：
- 编译预处理（pre-processing）：将 `#include` 处理好，宏展开等。使用 `-E` 指定，生成文件后缀名习惯用 `.i`。
- 编译（compiling）：转换为汇编代码。使用 `-S` 指定，生成文件后缀名习惯用 `.s`。
- 汇编（assembling）：将汇编代码转换为目标文件。使用 `-c` 指定，生成文件后缀名习惯用 `.o`。
- 链接（linking）：将目标文件进行链接，最终生成可执行文件
  - 可以把多个目标文件打包为一个作为函数库，这一过程可以借助 `ar` 命令来完成。函数库分为动态链接库与静态链接库

参考资料：

- 不确定好坏的资料：https://tldp.org/HOWTO/Program-Library-HOWTO/index.html


# Make

# CMake
资源：
- 官方Tutorial：[链接](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- cmake-demo：[Gitbook](https://www.hahack.com/codes/cmake/)，[Github](https://github.com/wzpan/cmake-demo)
- cmake-examples：[Github](https://github.com/ttroy50/cmake-examples)