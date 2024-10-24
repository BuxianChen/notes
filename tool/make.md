
# 编译

## 安装 gcc (多版本共存)

- 阿里云镜像：http://mirrors.aliyun.com/gnu/gcc/

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

## 关于头文件与库的搜索路径

```
C_INCLUDE_PATH  # C 头文件库搜索路径, 备注: 系统本身的不在这个变量里
CPLUS_INCLUDE_PATH  # C++ 头文件库搜索路径, 备注: 系统本身的不在这个变量里
LD_LIBRARY_PATH  # 动态链接库搜索路径, 备注: 系统本身的不在这个变量里
LIBRARY_PATH  # # 静态链接库搜索路径, 备注: 系统本身的不在这个变量里
```

为什么通常会需要配置 `LD_LIBRARY_PATH` 而不需要配置 `C_INCLUDE_PATH` 和 `CPLUS_INCLUDE_PATH`, 例如:

- 安装 CUDA
- 安装 openCV
- 安装 tensorRT

## 关于 `ldconfig`

注：以下均为简单理解，并非准确理解

在 Linux 下，默认动态链接库的搜索路径保存在 `/etc/ld.so.conf` 中，默认动态链接库的访问使用缓存机制，存放在 `/etc/ld.so.cache` （二进制格式），如果在默认路径下添加了动态链接库，则需要使用 `ldconfig` 更新默认路径里的动态链接库至 `/etc/ld.so.cache`。具体来说，可能会遇到需要使用 `ldconfig` 命令的场景例如安装 `mysql` 时，默认会将 mysql 的库文件安装到 `/usr/local/mysql/lib`，这个目录一般是在默认的动态链接库搜索路径下，因此由于缓存机制的存在，如果在不执行 `ldconfig` 时，在需要使用 mysql 相关的动态链接库时，会报找不到库的错误。

`ldconfig -v` 用于查看已经缓存的动态链接库。

另外，ldconfig 是系统层面的一些机制，在用户层面，也可以配置 `LD_LIBRARY_PATH` 来添加库目录


## 示例(杂录)

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

```bash
# 下面的命令用于观察具体的执行命令
# 然而使用 CMake 生成 Makefile, 接下来使用 make -n 命令以获取具体步骤, 往往不会深入到 gcc 命令
make --dry-run  # make -n
make VERBOSE=1
make --dry-run VERBOSE=1
```

# CMake

资源：

- 官方Tutorial：[链接](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- cmake-demo：[Gitbook](https://www.hahack.com/codes/cmake/)，[Github](https://github.com/wzpan/cmake-demo)
- cmake-examples：[Github](https://github.com/ttroy50/cmake-examples)

## 项目结构与 cmake 基本命令

CMake 用于生成 Makefile (如果使用 make 的话), 典型的编译步骤如下

```bash
# 步骤一: 配置(Configure)
# 生成 Makefile 文件 (准确地说会根据平台不同有所变化)
mkdir build && cd build && cmake ..
# 或者是
cmake -B build -S .. && cd build
# 加上下面的参数, 会额外生成 compile_commands.json, 可以观察到相应的编译命令
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# 步骤二: 构建(Build)
# (1) 可以直接用 make 命令
make
# 只展示编译步骤不执行, 但得不到想要的 gcc 编译命令
make --dry-run
# 或者简写为
make -n

# (2) 或者用 cmake 命令
cmake --build .
# 只展示编译步骤不执行, 但得不到想要的 gcc 编译命令
cmake --build . -- --dry-run

# 步骤三: 安装(Install), 可选, cmake --build 命令在 cmake>=3.15 才能使用, 之前的版本只能用 make --install
# 如果在配置环节不指定 -DCMAKE_INSTALL_PREFIX=/path/to/you/wish, 默认会安装到 /usr/local/bin, /usr/local/include, /usr/local/lib 这些路径
cmake --install .

# 指定安装目录既可以在 Configure 阶段指定 -DCMAKE_INSTALL_PREFIX=/path/to/you/wish
# 也可以在 Install 阶段
cmake --install . --prefix "/path/to/you/wish"

# (TODO: 不确定含义) 如果使用 multi-configuration tools, 可以使用这个
cmake --install . --config Release

# 也可以用下面的方式代替 cmake --install . 但 cmake --install . 更具兼容性和移植性
# cmake --build . --target install
```

一种典型的项目组织如下

```
root/
  - src/      # 源文件
  - include/  # 头文件
  - build/    # out-of-tree build
  - CMakeLists.txt
```

备注: 有些项目源文件和头文件不会分离, 而是放在同一级目录下, 也是常见的


## CMakeLists.txt 语法

### 重要语法索引 (cheetsheet)

初学先概览, 回头再复习. TODO: 此部分内容来自 GPT, 回头再做确认(似乎不太靠谱, 很多出现在 CMake 官方 tutorial 中的命令这里没有, 许多这里出现的, CMake 官方 tutorial 中没有).

- 项目信息
  - `cmake_minimum_required(VERSION X.Y)`: 指定 CMake 的最低版本要求。
  - `project(<project_name>)`: 定义项目的名称，可以可选地指定语言（C、C++等）。
- 源文件与目标
  - `add_executable(<target> <source1> <source2> ...)`: 创建一个可执行目标。
  - `add_library(<target> <source1> <source2> ...)`: 创建一个库目标（静态库或共享库）。
  - `aux_source_directory(<directory> <variable>)`: 查找指定目录中的所有源文件，并将它们的名称存入变量。
- 依赖关系
  - `target_link_libraries(<target> <library1> <library2> ...)`: 指定目标所依赖的库。
  - `add_subdirectory(<directory>)`: 添加子目录，通常用于包含其他 CMakeLists.txt 文件。
- 变量和选项
  - `set(<variable> <value>)`: 定义或设置变量。
  - `option(<option_name> <description> <default>)`: 定义一个布尔选项，用户可以选择该选项。
  - `if(<condition>) ... endif()`: 条件语句，用于控制构建时的行为。
  - `foreach(<var> <items>) ... endforeach()`: 迭代语句，用于遍历列表中的每一项。
- 包含和查找模块
  - `find_package(<package_name> [REQUIRED])`: 查找并加载指定的包。
  - `include_directories(<directory>)`: 添加包含目录。
  - `link_directories(<directory>)`: 添加链接目录。
- 生成规则和安装
  - `install(TARGETS <target> DESTINATION <directory>)`: 指定安装目标及其安装位置。
  - `file(<command> <args>)`: 处理文件的多种操作，例如复制、删除、生成等。
- 配置和构建选项
  - `set(CMAKE_BUILD_TYPE <type>)`: 设置构建类型，例如 Debug 或 Release。
  - `set(CMAKE_CXX_STANDARD <version>)`: 设置 C++ 标准。
- 脚本与宏
  - `macro(<name> <args>) ... endmacro()`: 定义一个宏，可以多次调用。
  - `function(<name> <args>) ... endfunction()`: 定义一个函数，具有局部作用域。
- 自定义命令和目标
  - `add_custom_command(...)`: 定义自定义构建命令。
  - `add_custom_target(...)`: 定义自定义构建目标。
- 其他
  - `message(<message>)`: 打印消息到标准输出。
  - `include(<module>)`: 包含其他 CMake 文件。


`add_library`, `add_executable` 分别表示创建一个库/可执行文件以及其所需要的源文件, 它们之后可能会跟上 `target_link_libraries`, `target_include_directories`, `target_compile_definitions`, 分别表示为了生成这个库/可执行文件需要的其他库名,头文件目录,编译时宏定义.

这类 `target_xxx` 的命令还有:

- `target_compile_features`: 用于指定 C++ 标准, 例如: `target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)`
- `target_compile_options`: 直接指定编译选项, 但需要手动解决跨平台问题. 例如: `target_compile_options(tutorial_compiler_flags INTERFACE -Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused)`. 所谓跨平台, 指的是例如指定 C++ 标准, 使用 `gcc` 编译器, 写法是 `gcc -std=c++11 ...`, 而 `MSVC` 编译器的写法是 `cl /std:c++11`. 因此一般来说, 使用 `target_compile_options` 时可能会见到下面的写法:

```cmake
# 下面的生成器表达式语法需要 >=3.15
cmake_minimum_required(VERSION 3.15)

# 这个是一个“虚拟”的库,用于添加编译选项
add_library(tutorial_compiler_flags INTERFACE)
target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_11)

# COMPILE_LANG_AND_ID 是 cmake 一个内置的生成器表达式, 生成器表达式的语法是 $<xxx:yyy>, 而 COMPILE_LANG_AND_ID 的语法是
# $<COMPILE_LANG_AND_ID:lang, compiler_id>
# 其中 lang 代表编程语言, compiler_id 是编译器名(可以有多个,用逗号隔开), 如果检查到编译器, 这一项的值为"1",否则为"0"

set(gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc_cxx "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

# "$<1:...>" 的值为 "...", "$<0:...>" 的值为空字符串 ""
target_compile_options(tutorial_compiler_flags INTERFACE
  "$<${gcc_like_cxx}:-Wall;-Wextra;-Wshadow;-Wformat=2;-Wunused>"
  "$<${msvc_cxx}:-W3>"
)
```

特殊变量: `PROJECT_SOURCE_DIR`, `PROJECT_BINARY_DIR`, `CMAKE_CURRENT_SOURCE_DIR`

### 官方 Tutorial 笔记

链接 (本文写作时的版本为`cmake==3.31.0-rc2`): [https://cmake.org/cmake/help/latest/guide/tutorial](https://cmake.org/cmake/help/latest/guide/tutorial)

Step1: A Basic Starting Point

主要介绍了单个源文件和头文件的情形, 但内容其实比较多

(1) 首先介绍了 `project`, `cmake_minimum_required`, `add_executable` 命令, 以及使用 cmake 命令行工具进行配置和构建

(2) 然后使用 `set` 命令配置 cmake 内置的变量 `CMAKE_CXX_STANDARD` 和 `CMAKE_CXX_STANDARD_REQUIRED`, 从而实现对 C/C++标准 的管控

(3) 最后补充介绍了 `project` 中可以定义项目的版本号, 在这之后的 `CMakeLists.txt` 内容里就可以使用 `<PROJECT-NAME>_VERSION_MAJOR` 和 `<PROJECT-NAME>_VERSION_MINOR` 内置变量, 为了将这两个变量变得可以在源码中使用(源码中用宏来获取这两个变量), 可以采用如下方案: 配置 `config.h.in`, 在这个模板文件中使用 `@<PROJECT-NAME>_VERSION_MAJOR@` 的语法用于宏值的定义, 并且在 `CMakeLists.txt` 中使用 `configure_file` 命令使得 cmake 在构建项目时生成真正的 `config.h` 文件, 最后使用 `target_include_directories` 让可执行文件的编译过程正确地 include 出于 `PROJECT_BINARY_DIR` 的 `config.h` 文件.

Step2: Adding a Library

主要介绍了多个源文件的情形, 且库代码位于单独的目录内, 且包含有自己的 `CMakeLists.txt` 文件.

(1) 内层 `CMakeLists.txt` 引入了 `add_library` 命令, 而外层的 `CMakeLists.txt` 为了使得最终的 main 程序能链接库, 需要使用 `add_subdirectory` 用于执行得到库, 使用 `target_link_libraries` 来让 main 链接到库, 使用 `target_include_directories` 使得 main 在编译时能识别到库的头文件

(2) 演示了如何利用 `option` 来为 cmake 在配置阶段增加 `-D<OPTION_NAME>=<value>` 来控制编译行为. 在官方的例子中, 场景是这样的, 我们的库是一个计算 `sqrt` 的函数, 而 main 程序希望实现如下功能: 如果希望使用我们自己实现的数学库, 那么就编译出该数学库, 并且 main 程序链接并使用其实现; 如果希望使用官方的数学库 `cmath`, 那么就不编译我们自己的数学库. 具体实现方式如下: TODO

Step3: Adding Usage Requirements for a Library

主要是优化 Step1 和 Step2 中的一些写法: 在 Step1 中, C/C++标准 的管控是使用 `set` 命令全局管控 `CMAKE_CXX_STANDARD` 和 `CMAKE_CXX_STANDARD_REQUIRED` 来实现的, 这样就不能实现不同模块采用不同的 C/C++标准; 在 Step2 中, 外层的 main 不仅需要链接库, 还需要通过 `target_include_directories` 添加库文件目录.

Step4: Adding Generator Expressions

本部分感觉不是核心用法, Generator Expressions 是 `CMakeLists.txt` 的脚本特性

Step5: Installing and Testing

引入 cmake 项目安装, 以及单元测试, TODO: 此处代码移除

```bash
# 首先 config
mkdir build && cd build && cmake ..
# 然后可以 install/test/pack
ctest
cpack
```

Step6: Adding Support for a Testing Dashboard

本部分感觉不是核心用法, 单元测试可以更花哨, 可以有个仪表盘来展示

Step7: Adding System Introspection

本部分与 Step6 都使用了 `include` 语法, 来引入 `/path/to/cmake/Modules` 中的其他“脚本”文件

Step8: Adding a Custom Command and Generated File

本部分实现了一个特殊功能: 首先编写一段 C 代码, 其执行后会生成一个头文件, 整个项目的构建过程是先得到这个头文件 (通过自定义命令来实现: `add_custom_command`), 然后其他的库/可执行文件可能需要使用这个头文件

Step9: Packaging an Installer

本部分介绍了项目打包, 具体介绍了使用 cpack 的打包方式, 打包为 `.tar.gz` 文件(二进制安装), 操作比较简单, `CMakeLists.txt` 加几行配置即可. 其他的项目打包和发布方式待探索

Step10: Selecting Static or Shared Libraries

前序步骤所有的库都是使用的静态库(当然也可以在 `add_library` 是设定为动态库, 但前面没有明确提到), 本部分介绍怎么“优雅”地配置以生成动态链接库

Step11: Adding Export Configuration

TODO: 似乎是让本项目在其他项目中能使用, TODO

Step12: Packaging Debug and Release

TODO


### 单文件

**引入**: `cmake_minimum_required`, `project`, `add_executable`, `aux_source_directory`

最简单的情况:

- 一个源文件和头文件, 用于编译得到一个可执行文件(引入 `cmake_minimum_required`, `project`, `add_executable`)
- 多个源文件和头文件在同一目录, 最终得到一个可执行文件(引入 `aux_source_directory`)


### 库与链接

**引入**: `add_library`, `target_link_libraries`, `target_include_directories`, `add_subdirectory`

```cmake
# cmake_minimum_required
# project (Demo)
add_subdirectory(math)

# math/CMakeLists.txt 包含如下内容
# aux_source_directory(. DIR_LIB_SRCS)
# add_library (MathFunctions ${DIR_LIB_SRCS})

add_executable(Demo main.cc)
target_link_libraries(Demo MathFunctions)
# 类似地, 也有一个 target_include_directories 来指定 include 目录
# target_include_directories(Demo PUBLIC "${PROJECT_BINARY_DIR}")
```

注意: `add_executable` 用于指定可执行文件依赖的源文件, 而 `target_link_libraries` 用于指定可执行文件依赖的库文件. 并且 `add_executable` 必须在 `target_link_libraries` **之前**. 虽然按照标准的 gcc 编译流程, 编译步骤确实是:

(1) 先编译并打包得到库文件 `libMathFunctions.a`
(2) 使用源文件 `main.c` 编译得到 `main.o`
(3) 使用 `main.o` 和 `libMathFunctions.a` 链接得到 `Demo`

但 CMakeLists.txt 的语法里不关心 `main.o` 这种中间产物, 它只关系为了生成 `Demo`, 它需要哪些源文件 (对应于 `add_executable`), 哪些库文件 (对应于 `target_link_libraries`)

`add_subdirectory` 命令表示“执行”子目录的 `CMakeLists.txt`: 父目录的 `CMakeLists.txt` 在 `add_subdirectory` 之前通过 `set` 定义的变量, 在子目录中的 `CMakeLists.txt` 是可见的, 并且在子目录 `CMakeLists.txt` 中定义的变量, 在父目录 `add_subdirectory` 之后, 对父目录的 `CMakeLists.txt` 也是可见的

### configure_file

`configure_file`: 根据模板文件生成宏定义头文件, 模板文件名通常是 `config.h.in`, 而宏定义头文件名一般是 `config.h`, 在使用 cmake 生成 Makefile 时, 会自动得到这个 `config.h` (与编译选项无关). 在 C/C++ 源代码中可以去通过 `#include "config.h"` 从而确定 `config.h` 中的宏定义

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.10)
# 自动生成如下变量: Demo_VERSION_MAJOR 是 1, Demo_VERSION_MINOR 是 0
project(Demo VERSION 1.0)

set(USE_MYMATH ON)

configure_file (config.h.in config.h)
# 也可以更明确地这么写, PROJECT_SOURCE_DIR 和 PROJECT_BINARY_DIR 是预定义变量
# configure_file ("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/config.h")

# 添加可执行文件的源文件
add_executable(Demo demo.cxx)
# 明确添加 include 目录
target_include_directories(Demo PUBLIC "${PROJECT_BINARY_DIR}")
```

`demo.cxx`

```c++
#include <iostream>
#include "config.h"

int main(int argc, char* argv[])
{
#ifdef USE_MYMATH
    std::cout << "USE_MYMATH defined" << std::endl;
#else
    std::cout << "USE_MYMATH undefined" << std::endl;
#endif
    std::cout << " Version " << VERSION_MAJOR << "." << VERSION_MINOR << std::endl;
    return 0;
}
```

`config.h.in` 的写法如下:

```
// 这是注释:
// 如果在 CMakeLists.txt 的逻辑里, USE_MYMATH 变量的值为 ON, 那么生成的 config.h 里将包含 #define USE_MYMATH
// 如果在 CMakeLists.txt 的逻辑里, USE_MYMATH 变量的值为 OFF, 那么生成的 config.h 里将包含 /* #undef USE_MYMATH */
#cmakedefine USE_MYMATH
// 生成的 config.h 中包含宏和值: VERSION_MAJOR:1, VERSION_MINOR:0
#define VERSION_MAJOR @Demo_VERSION_MAJOR@
#define VERSION_MINOR @Demo_VERSION_MINOR@
```


### 自定义编译选项

TODO: 搞完整

- `option` + `add_definitions`
- `option` + `config.h.in` + `include "config.h"`

**引入**: `option`, `add_definition`, `if`

前面已经看到过 cmake 像这样 `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 的选项, 事实上, cmake 可以自定义编译选项. 涉及到的相关 cmake 语法有

- `option`: 增加一个 cmake 的编译选项, 也就是增加一个这种用法 `cmake -D<选项名>=<值>`
- `add_definitions`: 将编译选项直接传递给编译器, 也就是利用 gcc 的 `-D<宏名>=<值>` 参数, 例如: `gcc -DDEBUG main.c -o main`


```bash
cmake -DUSE_MYMATH=ON ..
```

为此需要在 CMakeLists.txt 中包含如下内容

```cmake
# USE_MYMATH 的默认设置为 ON, 也就是定义 USE_MYMATH 这个宏
option (USE_MYMATH "Use provided math implementation" ON)
add_definitions(-DUSE_MYMATH)

# 根据宏来确定编译方式
if (USE_MYMATH)
  include_directories ("${PROJECT_SOURCE_DIR}/math")
  add_subdirectory (math)
  set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
endif (USE_MYMATH)
```

C 代码中可以这样使用

```c
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MYMATH
  #include "MathFunctions.h"
#else
  #include <math.h>
#endif

int main(int argc, char *argv[])
{
    if (argc < 3){printf("Usage: %s base exponent \n", argv[0]); return 1;}
    double base = atof(argv[1]);
    int exponent = atoi(argv[2]);
#ifdef USE_MYMATH
    printf("Now we use our own Math library. \n");
    double result = power(base, exponent);
#else
    printf("Now we use the standard library. \n");
    double result = pow(base, exponent);
#endif
    printf("%g ^ %d is %g\n", base, exponent, result);
    return 0;
}
```


### 安装与测试

CMake 官方 Tutorial-5

```bash
mkdir Step5_build
mkdir Step5_install
cd Step5_install
# 设置为 ON 或者 OFF 会决定 libSqrtLibrary.a 是否被安装
cmake -DUSE_MYMATH=OFF ../Step5
cmake --build .
cmake --install . --prefix=/path/to/Step5_install

# libtutorial_compiler_flags.a 不被安装是因为它被这样声明为了 INTERFACE
# add_library(tutorial_compiler_flags INTERFACE)
```

---

TODO: 基本已确认

关于 cmake 命令与 CMakeLists.txt 内的相对路径问题: CMakeLists.txt 的写法只需关注它本身与源码的相对路径, 而不必关心 cmake 命令行的参数. 这一点和 python 有区别 (python 代码里的 import 需要与 python 作为启动脚本时的当前目录需要相配)

```
CMakeLists.txt
main.cc
MathFunctions.cc
MathFunctions.h
```

`CMakeLists.txt` 的文件内容如下:

```cmake
cmake_minimum_required (VERSION 2.8)
project (Demo2)

# 查找目录下的所有源文件, 并将名称保存到 DIR_SRCS 变量, 在这个例子里是 main.cc MathFunctions.cc
aux_source_directory(. DIR_SRCS)

# 指定生成目标, 表示生成的可执行文件名为 Demo
# 在这个例子里, 等价于 add_executable(Demo main.cc MathFuctions.cc)
add_executable(Demo ${DIR_SRCS})
```

使用 `cmake .` 和 `mkdir build && cd build && cmake ..` 或者 `cmake -B build -S .` 均可以成功, 后两者的 Makefile 在 `build/Makefile`

`CMAKE_CURRENT_SOURCE_DIR` 总是指向 `CMakeLists.txt` 所在目录, 而 `CMAKE_CURRENT_BINARY_DIR` 总是指向生产的文件目录
---