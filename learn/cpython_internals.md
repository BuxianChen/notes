# Cpython Internals 笔记

参考 [https://realpython.com/cpython-internals/resources/](https://realpython.com/cpython-internals/resources/), 此书基于 cpython3.9.0b1

## Compiling Cpython

使用 python 源码编译的步骤如下:

```bash
git clone -b v3.9.0b1 --depth=1 https://github.com/python/cpython.git
sudo apt install build-essential
sudo apt install libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev libffi-dev
cd cpython
# configure 是一个可执行 shell 脚本, 它是由 autoconf 工具(命令)生成的, 其更原始的文件可能是代码库中的 configure.ac 这种文件通过 autoconf 命令得到的, 但是通过其他文件得到 configure 脚本的过程似乎是由 Python 开发者手工执行 autoconf 命令后上传至代码仓库并且被 git 所管理的, 因此 configure 脚本本身是怎么得到的这一细节暂时不太清楚
# ./configure 的执行结果主要是得到 Makefile 文件(以及一些其他文件, 具体有哪些不太确定, 按理是 make 命令执行时所需的配置文件), 注意这个 Makefile 文件并不由 git 所管理
# 目前的一些观察: git 会管理 configure 和 Makefile.pre 文件, 而后者似乎是最终生成的 Makefile 的模板, 猜测在执行 ./configure 脚本时, 应该是使用到了 Makefile.pre 文件的
# 总之: 就目前来说, ./configure 命令只需要执行一次, 后续即使改动了 Cpython 源代码, 需要重新编译时, 也只需要执行 make 命令, 而不需要执行 ./configure 命令. 因此研究和探索 Cpython 的起点可以定在 Makefile 已经得到了.
./configure --with-pydebug

# 使用下面的命令得到的二进制文件 python 只会在当前路径下, 不会被复制进 /bin, /usr/bin, /usr/local/bin 这类路径, 因此不用担心对全局的 python 有任何污染
make -j2 -s

# make install 会将编译好的 python 复制进 /bin, /usr/bin, /usr/local/bin 这类路径, 并且会修改 python 命令的指向, 例如: 首先将编译好的 python 复制进 /usr/bin/python3.9, 然后构建 /usr/bin/python -> /usr/bin/python3.9 这种软链接, 因此不推荐这种做法, 而是推荐用 make altinstall
# make install

# make altinstall 不会修改 python 命令的软链接, 因此是推荐的安装方式
# make altinstall

# 注意: 我们不想对全局的 python 或 python3 或 python3.x 有任何污染, 因此我们既不执行

# 使用如下方式运行编译好的 python
./python
```

使用 `configure` 脚本得到的 Makefile 很冗长, 暂时没法过于细究, 但这确实是一切事情的“主入口”, 因此很有必要知道 `make -j2 -s` 具体执行了什么. 对这里用到的 make 命令做些简单介绍, `-j2` 是打开多线程编译 (大约就是各个 .c 文件编译为 .o 文件是没有先后依赖的, 所以可以通过多线程来加速编译过程), `-s` 的作用只是不打印具体的执行命令, 但出于学习考虑, 知道 make 的具体执行内容是重要的, 而直接读 Makefile, 里面的分支过于繁杂, 这种时候, 可以使用 `make -n` 命令来得知 make 的具体执行步骤, 注意: `make -n` 只打印需要执行的命令, 但它不会真的执行:

```bash
make -n
```

输出内容大体如下:

```bash
# 这种 gcc 由 .c 文件得到 .o 文件的命令有 100 多个
gcc -pthread -c -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall    -std=c99 -Wextra -Wno-unused-result -Wno-unused-parameter -Wno-missing-field-initializers -Werror=implicit-function-declaration -fvisibility=hidden  -I./Include/internal  -I. -I./Include    -DPy_BUILD_CORE -o Programs/python.o ./Programs/python.c
# 此处略去 100 多个 gcc 由 .c 文件得到 .o 文件的编译命令

# 删除原本的静态链接库
rm -f libpython3.9.a
# 使用 ar 命令打包静态链接库: 打包的 .o 文件大概也有上百个, 其实就是前面 gcc 编译出来的 .o 文件
ar rcs libpython3.9.a Modules/getbuildinfo.o Parser/acceler.o Parser/grammar1.o ...

# 这一步应该就是通过链接器得到 python 这个可执行文件, 具体命令看不太懂
gcc -pthread     -Xlinker -export-dynamic -o python Programs/python.o libpython3.9.a -lcrypt -lpthread -ldl  -lutil -lm   -lm 

# 以下内容都有些看不太懂, 感觉主要是一些 python config 的内容, 不太确定
echo "none" > ./pybuilddir.txt
./python -E -S -m sysconfig --generate-posix-vars ;\
if test $? -ne 0 ; then \
	echo "generate-posix-vars failed" ; \
	rm -f ./pybuilddir.txt ; \
	exit 1 ; \
fi
case "`echo X $MAKEFLAGS | sed 's/^X //;s/ -- .*//'`" in \
    *\ -s*|s*) quiet="-q";; \
    *) quiet="";; \
esac; \
echo " CC='gcc -pthread' LDSHARED='gcc -pthread -shared    ' OPT='-DNDEBUG -g -fwrapv -O3 -Wall' \
	_TCLTK_INCLUDES='' _TCLTK_LIBS='' \
	./python -E ./setup.py $quiet build"; \
 CC='gcc -pthread' LDSHARED='gcc -pthread -shared    ' OPT='-DNDEBUG -g -fwrapv -O3 -Wall' \
	_TCLTK_INCLUDES='' _TCLTK_LIBS='' \
	./python -E ./setup.py $quiet build
gcc -pthread     -Xlinker -export-dynamic -o Programs/_testembed Programs/_testembed.o libpython3.9.a -lcrypt -lpthread -ldl  -lutil -lm   -lm 
# Substitution happens here, as the completely-expanded BINDIR
# is not available in configure
sed -e "s,@EXENAME@,/usr/local/bin/python3.9," < ./Misc/python-config.in >python-config.py
# Replace makefile compat. variable references with shell script compat. ones;  -> 
LC_ALL=C sed -e 's,\$(\([A-Za-z0-9_]*\)),\$\{\1\},g' < Misc/python-config.sh >python-config
# On Darwin, always use the python version of the script, the shell
# version doesn't use the compiler customizations that are provided
# in python (_osx_support.py).
if test `uname -s` = Darwin; then \
	cp python-config.py python-config; \
fi
```

总之, make 命令的最终目的主要就是得到 python 可执行文件, 到此为止, 我们需要明确目标: 在执行 `python xx.py` 时到底发生了什么, 这里先简单探索下: 首先, 我们知道 `./Programs/python.c` 文件里应该有作为入口的 `main` 函数, 然后 `xx.py` 作为命令行参数传递给了这个 `main` 函数, 我们简单看一下 `./Programs/python.c` 的文件内容, 果然不出所料, 包含下面的代码

```C
int main(int argc, char **argv)
{
    return Py_BytesMain(argc, argv);
}
```

所以这里便是一切的主入口了, 值得注意的是 Cpython 的实现语言是 C 语言, 而非 C++ 语言.

如果没有任何参考资料, 只能“顺藤摸瓜”学习源代码的话, 下一步便是继续深入 main 函数, 了解它每一步都在做啥. 参考书介绍了各种组件, 因此最后还缺乏一个从 main 入手的总览, TODO: 这个步骤非常重要, 但留待后续研究清楚后再补齐.

## The Python Language and Grammar (TODO)

本章主要是介绍 Python 的语法, 所谓语法就是: 文本文件符合什么书写规范时, 才能被执行. 从“微观层面”说: 譬如说 python 语言中规定的 if 语句得这么写:

```python
if x > 1:
    print("x is great than 1")
else:
    print("x is not great than 1")
```

而不能是 C 语言的写法:

```C
if (x > 1) printf("x is great than 1");
else printf("x is not great than 1");
```

我们可以想象, 应该会有一个文件规定了 python 语言的语法, 并且 python 可执行程序(也就是 python 解释器) 也是根据这个文件来将 `xx.py` 文件转化为具体的执行指令的. 这个文件也就是语法文件.

## Configuration and Input
