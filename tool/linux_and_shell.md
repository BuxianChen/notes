# Linux & Shell

## 第 1 课：shell 命令

注意：本节课讲的都是 bash 命令。

### shell 命令的一般性介绍

#### 机器上有哪些 shell 命令？

当我们在输入命令时，机器会依据 `$PATH` 变量中存储的目录依次查找，直到找到对应的命令，之后执行命令。如果找不到命令，则会输出报错信息。

```bash
$ echo $PATH  # 打印出 $PATH 变量，注意到分隔符为冒号
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
$ which echo  # 使用的命令的具体路径
/bin/echo
$ /bin/echo $PATH  # 直接用命令的完整路径
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

注意：当前目录一般不被包含在搜索路径中，因此需要使用类似于 `./custom_command args` 的方式运行当前路径下的命令，而不能使用 `custom_command args` 这种写法。

#### echo 命令

echo 命令的作用是向输出流输出东西

```bash
$ echo hello  # hello 也可以用双引号或单引号包起来，但结果都是一样的
hello
$ echo -e "h\nllo"  # -e 表示执行转义
h
llo
$ echo "Path is $PATH"  # 使用双引号时 $PATH 会作为变量
Path is /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
$ echo 'Path is $PATH'  # 使用单引号时 $PATH 只是普通的字符串
Path is $PATH
$ echo `ls`  # 执行命令
file_1 file_2 dir_1 dir_2
```

#### 流（输入、输出）

简单理解：文件属于流，标准输入流与标准输出流是特殊的流

**重定向**

```bash
$ echo "abc" > data.txt  # 清空 data.txt 内容，将输出流中的东西写入 data.txt 中
$ echo "df" >> data.txt  # 追加至 data.txt
$ cat < 1.txt  # 1.txt 的文件内容先移到输入流，输入流再作为 cat 命令的输入
```

**管道**

后一条命令所需的输入流为前一条命令执行后的输出流

```bash
$ cat data.txt | grep a
ac
```

注意区分输入、输出、返回（见下一小节）的概念。管道是针对输入与输出的，不是针对返回的。

#### shell 命令的“返回”

> Commands will often return output using `STDOUT`, errors through `STDERR`, and a Return Code to report errors in a more script-friendly manner. The return code or exit status is the way scripts/commands have to communicate how execution went. A value of 0 usually means everything went OK; anything different from 0 means an error occurred.
> [Shell Tools and Scripting · the missing semester of your cs education (mit.edu)](https://missing.csail.mit.edu/2020/shell-tools/)

shell 命令一般将输出（output）写入到标准输出流中，将错误信息（errors）写入到标准错误流中。另外，每条命令执行结束后会返回一个返回状态码（Return Code），返回状态码为 0 表示正常运行，非零表示存在错误。注意：状态码的作用是为了方便脚本与命令之间的通信，而输出与错误信息是为了方便用户。

输出与错误信息可以用于重定向或者管道操作。

状态码的用途举例：

- 在布尔操作（例如：`&&` 和 `||`，注意它们都是短路求值的）被使用到，`false` 命令的返回状态码为 1，`true` 命令的返回状态码为 0。

  ```bash
  false || echo "Oops, fail"
  true || echo "Will not be printed"
  ```

- 使用特殊变量 `$?` 获取上一条命令的返回状态码

  ```
  $ ls f.txt
  ls: cannot access f.txt: No such file or directory
  $ echo $?
  2
  ```

#### shell 命令形式的一般性说明

**命令的形式**

一般而言，命令的形式为 `命令名 参数列表` 形式。某些命令还需要接收流或是文件里的数据，例如：直接执行 `grep pattern` 命令时，会要求继续输入，直至按下 `Ctrl+Z` 快捷键结束，对于这类命令，一般要使用管道，最常见的例子是：

```bash
ls | grep pattern  # 匹配符合 pattern 的文件名
grep pattern data.txt  # data.txt 中符合 pattern 的行
```

**引号**

在 bash 命令中，很多时候引号不是必须的。且单引号与双引号的效果是不一样的

```bash
echo "$PATH"  # $PATH 会被当作变量
echo '$PATH'  # $PATH 会被当作普通的字符串
```

**空格**

空格是很重要的分隔符，因此向命令传递带有空格的参数时，要用单引号或双引号将该参数包裹起来，或者使用 `\ `（反斜线空格）的形式进行转义，例如：

```shell
mkdir "my photo"
cd my\ photo
```

再例如，在 bash 中定义变量时，等号前后不能有空格：

```bash
$ ABCD=1  # 正确
$ ABCD = 1  # 错误，bash 会将 ABCD 看作是一个命令
-bash: ABCD: command not found
$ ABCD=$(ls)  # 使用命令的输出值为变量复制
```

备注：空格需要特别小心，很容易出错，需仔细检查。

**寻求命令的帮助**

寻求命令的帮助可以使用 `命令名 --help`，退出帮助文档的快捷键为 `q`。

#### shell 基础命令

```bash
$ cd ~/data/images  # 路径切换，~ 代表当前用户的家目录，典型地，~ 将被扩展为：/home/username
$ cd animal/dog  # bash 中以及 shell 脚本中，使用 # 来进行单行注释
$ cd -  # 回到上一次切换的目录，即：~/data
$ cd -  # 回到上一次切换的目录，即：~/data/images/animal/dog
$ pwd  # 显示当前的绝对路径
/home/username/data/images/animal/dog
$ ls  # 列出当前目录下的文件或子目录名
image_1.jpg
image_2.jpg
```

### 目录及文件的权限含义

推荐链接：[鸟哥私房菜](http://linux.vbird.org/linux_basic/0210filepermission.php#filepermission)

```bash
$ ls -l
drwxr-xr-x  4 root root 4096 Jun 10 14:08 downloads
-rw-r--r--  1 root root    8 Jun 29 16:29 log.txt
```

第一项的第一个字母代表类型，其中：`d` 代表目录，`-` 代表文件。后面九个字母分为三组，依次代表所有者、用户组、其他人的权限，包括读（`r`）、写（`w`）、可执行（`x`）三种。对于目录来说：

- 读权限代表可以查看目录下的文件列表。即可以对目录使用 `ls` 命令；
- 写权限代表可以：在目录下创建子目录或文件；删除目录下的子目录或文件（无论子目录或文件的权限是什么）；修改子目录或文件的文件名；移动子目录或文件位置。即可以对该目录下的东西使用 `rm`，`mkdir`，`mv` 等命令；
- 可执行权限代表可以进入该目录。即可以 `cd` 进入该目录；

第二项代表连接数

第三项代表文件所属用户

第四项代表文件所属群组

第五项代表文件大小，单位为字节，对于文件夹，其大小**不是**指该文件夹下的所有文件的总大小

第六项为文件最后修改时间

第七项为文件名

### 环境变量

变量可以分为 shell 变量与环境变量。环境变量是从父进程中继承过来的，如果当前进程产生了子进程，则子进程将会继承所有的环境变量。shell 变量只在当前进程中起作用，不会发生继承关系。

列出当前 shell 的环境变量：

```bash
$ export -p
```

添加环境变量：

```bash
$ MY_ENV_VAR=xyz; export MY_ENV_VAR
# 或者
$ export MY_ENV_VAR=xyz
```

#### IFS（并不是环境变量）

IFS （internal field separator，内部字段分隔符）环境变量默认为空格，制表符，换行符。

### shell 命令记录

#### sed（待补充完整）

sed 命令是用来处理文本行的

> `sed` is a [stream editor](http://man7.org/linux/man-pages/man1/sed.1.html) that works on piped input or files of text

以例子说明其用法

文件内容如下：`a.py`

```
def fizz_buzz(limit):
    for i in range(1, limit+1):
        if i % 3 == 0:
            print('fizz', end="")
        if i % 5 == 0:
            print('fizz', end="")
        if i % 3 and i % 5:
            print(i)
def main():
    fizz_buzz(10)
if __name__ == "__main__":
    main()
```

**例子 1：使用正则表达式做文本替换**

此处的 `s/to/do/` 表示将 `to` 替换为 `do`，`s` 表示 substitute（替换）。

```bash
$ echo howto | sed "s/to/do/"
howdo
```

**例子 2：挑选指定行**

此处表示挑选第 4 至第 5 行打印，在默认情况下 sed 命令会将每行都打印出来，使用 `-n` 选项可以抑制这一行为，`4,5p` 中的 `p` 表示 print（打印）。

```bash
$ sed -n "4,5p" a.py
            print('fizz', end="")
        if i % 5 == 0:
```

可以使用 `-e` 选项选择多个行，注意到第 2 行与第 3 行打印了两次

```bash 
$ sed -n -e "1,4p" -e "2,3p" a.py
def fizz_buzz(limit):
    for i in range(1, limit+1):
    for i in range(1, limit+1):
        if i % 3 == 0:
        if i % 3 == 0:
            print('fizz', end="")
```

可以使用 `5~3p` 这种写法表示从第 5 行开始每隔 3 行进行打印

```bash
$ sed -n -e '5~3p' a.py
        if i % 5 == 0:
            print(i)
if __name__ == "__main__":
```

**例子 3：文本替换**

只替换第一个匹配

```bash
$ sed -n 's/=/!/p' a.py
        if i % 3 != 0:
            print('fizz', end!"")
        if i % 5 != 0:
            print('fizz', end!"")
if __name__ != "__main__":
```

替换所有匹配

```bash
$ sed -n 's/=/!/gp' a.py
        if i % 3 !! 0:
            print('fizz', end!"")
        if i % 5 !! 0:
            print('fizz', end!"")
if __name__ !! "__main__":
```

另外也可以进一步将 `gp` 换成 `gip`，其中 `i` 表示忽略大小写（case insensitive），另外 `gip` 三个字母的顺序可以随意调换

取前四行，将连续多个空格替换为一个空格

```bash
$ sed -n '1,4s/  */ /gp' a.py
def fizz_buzz(limit):
 for i in range(1, limit+1):
 if i % 3 == 0:
 print('fizz', end="")
```

多个替换规则，两种写法均可，注意到第 2 行没有任何模式被匹配上，由于 `-n` 参数的原因没有被打印出来。

```bash
$ sed -n -e '1,4s/zz/aa/gp' -e '1,4s/==/!=/gp' a.py
$ sed -n '1,4s/zz/aa/gp;1,4s/==/!=/gp' a.py
def fiaa_buaa(limit):
        if i % 3 != 0:
            print('fiaa', end="")
```

#### dirname

```bash
$ dirname <file|dir> # 返回文件或目录的父目录
```

#### tee（待补充）

```bash
$ tee a.txt b.txt  # 同时向 a.txt 和 b.txt 中写入相同的内容，输入这行命令后，需要继续输入要写入的内容，以 Ctrl+Z 结束。注意 tee 命令写入的东西同时也会打印至屏幕上。
```

一般使用管道的方式进行使用，例如

```bash
$ echo "abc" | tee a.txt b.txt
abc  # 写入文件并同时将写入的信息输出至标准输出流
$ echo "abc" | tee a.txt > b.txt  # 用重定向的方式同时写入两个文件，并且不显示在屏幕上
```

一个常见的错误参考例 7：修改屏幕亮度

#### printf

```bash
$ printf "%-5s %-10s %-4.2f\n" 123 acb 1.23
```

printf 命令仿照 C 语言的 printf 函数，用于格式化输出，格式控制符 `%-5s` 表示左对齐（不加 `-` 则表示右对齐），**最少**使用 5 个字符长度，以字符串的形式输出。格式控制符 `%-4.2f` 表示最少使用 4 个字符长度，小数点后保留 2 位，以浮点数形式输出。

#### !

在 shell 中，! 被称为 *Event Designators*，用于方便地引用历史命令。

- `!20` 表示获取 history 命令中的第 20 条指令；
- `!-2` 表示获取 history 命令中的倒数第 2 条指令；
- `!!` 是 `!-1 `的一个 alia；
- `!echo` 表示最近地一条以 `echo` 开头的指令；
- `!?data` 表示最近的一条包含 `data` 的指令

#### echo、stty

echo 命令可以使用 `-e` 参数使得输出字符串中的转义字符产生效果；另外，echo 命令还可以控制输出的字体颜色，详细情形不赘述。

```bash
$ echo -e "1\t2\t"
1	2	
$ echo -e "\e[1;31mred\e[0m" # 输出的字体颜色为红色
```



```bash
#!/bin/bash
# filename: password.sh
echo -e "Enter password"
stty -echo
read password
stty echo
echo Password read
echo "password is $password"
```

#### alias

别名相当于自定义命令，可以使用 alias 命令实现，也可以定义函数实现。此处仅介绍 alias 命令。

```
$ alias myrm='rm -rf'
$ myrm data/
```

alias 命令产生的别名只在当前 shell 有效

#### du 与 df（待补充）

linux 中，目录本身占用一个 block，其大小为 4K，用于存储一些元数据，例如权限信息、修改时间等。

du 命令意为 disk usage，用于显示目录或文件的大小

```bash
# 最基础的用法: -h意为--human-readable,表示自适应地使用K/M/G来显示文件(夹)大小

# 递归地显示<dir>以及所有子目录的大小(不会显示目录下的单个文件大小)
# 目录大小 = 目录大小(4kB)+当前目录下文件大小总和+子目录大小总和
$ du -h <dir>

# 显示<file>文件大小
$ du -h <file>
```

```bash
$ du -sh <dir>  # 只显示<dir>目录的大小
$ du -h --max-depth=1 <dir>  # 只显示<dir>目录以及一级子目录大小
$ # 注意-s与--max-depth不能混用, --max-depth=0与-s的作用一致
$ du -h --max-depth=1 --exclude=<subdir> <dir>  # 输出时不显示<subdir>的大小
$ du -Sh --max-depth=1 <dir>  # 计算目录大小时不包括子目录大小
```

df 命令意为 disk free

```bash
$ df -h  # 查看挂载信息
```

#### watch

```bash
# 每秒钟执行一次nvidia-smi命令以监控GPU使用情况，按ctrl+c退出监控
$ watch -n 1 nvidia-smi
```

#### sort

```bash
$ docker images | sort -n -k 7
```

sort 命令的作用是以行为单位进行排序。`-n` 选项表示把字符当作数字进行排序，`-k 7` 选项表示选择第 7 列进行排序。可以使用 `-t :` 来指定列的分割符为 `:`，可以使用 `-r` 选项进行降序排列（默认是升序排列）

#### paste

```bash
paste -d= a.txt b.txt > c.txt
```
将 `a.txt` 与 `b.txt` 按行进行拼接，拼接字符为 `=`，即:

```bash
paste -sd+ a.txt | bc
```
`-s`表示把 `a.txt` 的每行进行拼接，拼接字符为 `+`，此条命令用于对一列数字求和


#### history（待补充）

history 命令用于显示历史命令。经过测试发现它的行为与 `~/.bash_history` 文件有关，一个可以解释的逻辑如下，系统保留有一份历史命令的记录文件，进入一个终端后，执行任意一条指令都会在 `.bash_history` 文件末尾追加一条记录（不同终端有着独立的记录，互不影响），退出终端时，

打开两个终端，分别输入（各条命令的实际）：

```bash
$ echo dosome1
$ echo dosome3
```

```bash
$ echo dosome2
$ echo dosome4
```

#### mount/umount

```bash
$ mount -t nfs <device> <dir>  # 将设备/目录device挂载到目录dir(称为挂载点)上, -t参数表示指定档案系统型态, 通常无需指定
$ umount <device/dir>  # 取消挂载, 用挂载点或者设备名均可
```

若 umount 时出现 `device is busy` 的错误，可以参照[链接](https://www.cnblogs.com/xuey/p/7878529.html)进行解决，摘录如下：

```bash
$ # fuser命令使用 apt install psmisc 进行安装
$ fuser -m -v <dir>  # 显示所有占用该目录的进程
$ fuser -k <dir>  # 杀死所有占用该的进程
$ umount <dir>

$ # 以下为手动删除的办法, 但有时不奏效
$ # ps aux | grep <pid>  # 显示该进程的信息
$ # kill -9 <pid>  # 可以直接杀死进程
```

#### ps（待补充）

用于列出所有进程

[参考链接](https://www.jianshu.com/p/943b90150c10)

#### nmap

打印本机打开的所有端口信息

```bash
$ # apt install nmap
$ nmap 127.0.0.1
```

#### Shell-Parameter-Expansion

[man官方文档](https://www.gnu.org/software/bash/manual/bash.html#Shell-Parameter-Expansion)


** `${varname:-"abc"}` **

这种语法表示给变量设定缺省值，若 `varname` 未被定义，则 `varname` 将被赋值为 `abc`，否则该条语句不起作用。例如：

已经被定义的情形

```bash
$ varname="111"
$ echo ${varname}
111
$ echo ${varname:-"abc"}
111
$ echo ${varname}
111
```

没有被定义的情形

```bash
$ echo ${varname}  # 未被定义的变量没有输出

$ echo ${varname:-"abc"}
abc
$ echo ${varname}
abc
```

实例：在 Github Pytorch 项目的 README 文档介绍如何进行源码安装时，需要执行如下命令

```bash
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
```

** `${!arr[@]}`，`${!arr@}`，`${!var}` **

- `${!arr[@]}` 用于返回数组下标，例如：

  ```bash
  arr=(h0 h1)
  for i in ${!arr[@]}; do echo $i; done  # 输出0 1
  arr[10]=h10
  for i in ${!arr[@]}; do echo $i; done  # 输出0 1 10
  ```

- `${!arr@}` 表示输出以"arr"开头的变量名，例如：
  ```bash
  arr1=1
  arr2=2
  arr=3
  for i in ${!arr@}; do echo $i; done  # 输出arr1 arr2 arr
  ```
- `${!var}` 表示取出以var变量的值命名的变量的值（类似于C语言中的指针），例如：
  ```bash
  tmp=/path/to/temp
  path=tmp
  echo ${path}  # 输出/path/to/temp
  ```

** 字符串替换：`` **

参考 [stackoverflow](https://stackoverflow.com/questions/13210880/replace-one-substring-for-another-string-in-shell-script)

- `${parameter/pattern/string}` 表示将 `parameter` 变量中**第一次**出现的 `pattern` 替换为 `string`，例如：
  ```bash
  var="data-clean-100"
  echo ${var/-1/_1} # data-clean_100
  ```
- `${parameter/pattern/string}` 表示全部替换，例如：
  ```bash
  var="data-clean-100"
  echo ${var/-/_} # data_clean_100
  ```

**获取文件扩展名**

```bash
filename="/a/b/c.train.json"
echo ${filename%.*}  # /a/b/c.train  从右到左非贪心匹配".*"并剪切掉
echo ${filename%%.*}  # /a/b/c  从右到左非贪心匹配".*"并剪切掉
echo ${filename#*.}  # train.json  从左到右非贪心匹配并剪切掉
echo ${filename##*.}  # json  从左到右贪心匹配并剪切掉
```

实例

```bash
cat amiBuild-75723-Wed-Jun-22-2022.wget.sh | grep -v '#'|awk '{print $3" "$4}'|while read -r x y ; do echo $x/${y##*/}>>all.txt; done
```

输入：
```
#xxxx
wget    -P amicorpus/ES2002a/audio https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav --no-check-certificate
wget    -P amicorpus/ES2002b/audio https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav --no-check-certificate
```

输出：
```
amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
amicorpus/ES2002b/audio/ES2002b.Mix-Headset.wav
```

#### ln

```bash
$ # ln -s 原始路径 目标路径
$ ln -s /home/to/directory /data  # 得到/data/directory
```

#### file -i

```bash
$ file -i 文件名  # 查看编码格式
```


#### command

command 用于 shell 脚本中，忽略脚本定义的函数，而直接去寻找同名的命令

```shell
function ls() {
  echo "haha"
}
command ls  # 输出: 当前路径下的文件(夹)
# 使用PATH环境搜索ls
command -p ls  # 输出: 当前路径下的文件(夹)
```

另一种情况更加常见：在 `~/.bashrc` 中为 `ls` 定义了 alias

```shell
alias ls='ls --color=auto'  # 自动按文件类型及权限显示目录内容
```

这种情况下

```shell
command -p ls  # 输出不带颜色
command -V ls  # 输出: ls is aliased to `ls --color=auto`
command -v ls  # 输出: alias ls='ls --color=auto'
```

#### getopt

[stackoverflow](https://unix.stackexchange.com/questions/85787/invoking-shell-script-with-option-and-parameters)

### 命令例子

#### 例 1：/dev/null、文件描述符

```bash
ls data.txt 2>/dev/null 1>log.txt
```

主要参考[知乎问答](https://www.zhihu.com/question/53295083/answer/135258024)

此命令的作用是将命令 `ls data.txt` 运行过程中产生的标准错误信息忽略，而将产生的标准输出信息重定向（写入）至 `log.txt` 中。具体解释如下：

**文件描述符**

> 文件描述符是与文件输入、输出关联的整数。它们用来跟踪已打开的文件。最常见的文件描述符是stdin、stdout、和stderr。我们可以将某个文件描述符的内容重定向到另外一个文件描述符中。
> *《linux shell脚本攻略》*

具体来说，常见的文件描述符为 0、1、2 这三个，分别对应 stdin（标准输入）、stdout（标准输出）、stderr（标准错误）。事实上：

- stdin 对应于 `/dev/stdin`
- stdout 对应于 `/dev/stdout`
- stderr 对应于 `/dev/stderr`

在 shell 命令或脚本中常用的是 1 和 2。因此在上面的例子中，是将命令 `ls data.txt` 产生的标准输出重定向至 `log.txt` 中，将产生的标准错误信息重定向至 `/dev/null` 中。

**/dev/null**

`/dev/null` 是 linux 中的一个特殊设备，作用是接受输入并丢弃。在 shell 命令中的作用一般是用来接受输出信息，避免在屏幕上显示，同时也不希望使用文件对输出进行接收。

#### 例 2：管道、curl、grep、cut 结合使用

```bash
$ curl --head --silent baidu.com | grep -i content-length | cut --delimiter=' ' -f2
81
```

curl 命令用来请求 Web 服务器。其名字的含义即为客户端（client）的 URL 工具。具体用法可以参照[阮一峰博客](http://www.ruanyifeng.com/blog/2019/09/curl-reference.html)

> 它的功能非常强大，命令行参数多达几十种。如果熟练的话，完全可以取代 Postman 这一类的图形界面工具。——阮一峰博客《curl 的用法指南》

grep 的 `-i` 参数表示匹配时忽略大小写

cut 命令用于切分字符串，有若干种用法：取出第 $$m$$ 个到第 $$n$$ 个字符；按分隔符取出第 $$k$$ 个字符串。此处 cut 命令中用 `--delimiter=' '` 指定分割符为空格，`-f2` 表示取出以该分割符分割产生的第二项

#### 例 3：source、export

通常情况下，执行如下语句

```
./a.sh
```

实际发生的事情是：创建一个子进程，在子进程中运行 `a.sh`，然后回到当前 shell 中。注意子进程是用 fork 的方式产生的，因此子进程的环境变量与当前的 shell 是完全一致的。因此 `a.sh` 中设置的环境变量不会影响到当前的 shell。例子如下：

```bash
# a.sh的内容
export FFFF=ffff
echo $ABCD
echo $FFFF
```

```bash
$ export ABCD=abcd
$ export -p | grep ABCD
declare -x ABCD="abcd"
$ ./a.sh
abcd
ffff
$ echo $FFFF  # 没有输出
```

source 的作用是在当前 shell 中运行脚本内容， 使得脚本中设置的环境变量会影响到当前的 shell。例如：

```bash
$ source a.sh
# 也等价于
$ . a.sh
```

备注：对于 `source a.sh` 这种执行方式来说，脚本中的 `$0` 为终端本身，例如：`/bin/bash`，而 `./a.sh` 这种执行方式，脚本中的 `$0` 将会是 `./a.sh`

#### 例 4：带颜色的终端输出

```
echo -e "\e[44;37;5mabcs\e[0m"
```

- `\e[<...>m` 表示 `<...>` 中的部分为设置输出格式（字体颜色，背景颜色，是否加粗，是否产生闪烁效果等），`\e` 也可以用 `\033` 代替。

- `44;37;5`：`44` 表示设置背景色为蓝色，`37` 表示设置前景色（也就是字体颜色）为白色，`5` 表示字体产生闪烁效果 。

  <font color=red>备注</font>：这些数字实际上被称为 SGR 参数（Select Graphic Rendition） ，这些数字的顺序是不重要的，完整的列表可以参见 [ANSI escape code - Wikipedia](https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_(Select_Graphic_Rendition)_parameters)，简短版的说明可以参见 [简书](https://www.jianshu.com/p/bba963125f1a)。

- `\e[0m` 表示设定回终端的默认值

#### 例 5：排除某些子目录的复制

```bash
$ ls ./src | grep -v 'logs\|images' | xargs -i cp -r ./src/{} ./dst
```

排除 `logs` 与 `images` 子目录从 `src` 复制文件至 `dst`

```
ROOT
  - src
    - logs/
    - images/
    - models/
    - main.py
  - dst
```

- `grep -v` 表示排除，`'logs\|images'` 表示或的关系
- `xargs -i` 表示将前一步的结果放在 `./src/{}` 的 `{}` 处。

#### 例 6：grep、xargs

```bash
find . -name "*.py" | xargs grep -n "Model"
```
查找当前目录及子目录下所有 `.py` 文件含有关键词 `Model` 的行及行号。


#### 例 7：修改屏幕亮度

```bash
$ echo 5000 | sudo tee /sys/class/backlight/intel_backlight/brightness
# 不能使用 sudo echo 5000 > /sys/class/backlight/intel_backlight/brightness
```

#### 例 8：依据 csv 格式文件执行命令（read 命令）

`cpfiles.txt` 文件如下，希望按照 csv 进行文件拷贝

```
a.txt,a1.txt
b.txt,b1.txt
```

```bash
$ cat cpfiles.txt | while IFS="," read src dst; do cp $src $dst; done
$ cat cpfiles.txt | while IFS="," read -a row; do cp ${row[0]} ${row[1]}; done
```

#### 例 9：生成序列（seq 命令）

```bash
$ for i in $(seq 100000); do echo ${i} >> x.txt; done
```

#### 例 10：打印 Git Objects

```bash
$ find .git/objects/ -type f|awk -F"/" '{print $3$4}' | while read -r obj; do echo =========; echo ${obj} $(git cat-file -t ${obj}); echo $(git cat-file -p ${obj}); done;
```

输出
```
=========
08585692ce06452da6f82ae66b90d98b55536fca tree
100644 blob 78981922613b2afb6025042ff6bd878ac1994e85 a.txt
=========
4b825dc642cb6eb9a060e54bf8d69288fbee4904 tree

=========
78981922613b2afb6025042ff6bd878ac1994e85 blob
a
=========
db28edd91114108e88d430f317c46f87e9cb2896 commit
tree 08585692ce06452da6f82ae66b90d98b55536fca author xxx <xxx@qq.com> 1663866229 +0800 committer xxx <xxx@qq.com> 1663866229 +0800 add a.txt
```

### 杂录

#### 大杂烩

- 一般用户的命令提示符为 `$`，而 root 用户的命令提示符为 `#`。
- `Ctrl+L` 快捷键用于清除屏幕

#### linux系统目录

**/proc 目录**

```bash
$ cat /proc/23512/environ | tr "\0" "\n"
```

`/proc` 目录中按进程 id 存放着进程的相关信息，特别地，上述命令用于查看该进程运行时的环境变量。

#### 反引号、单引号、双引号

单引号的用于忽略所有特殊字符的特殊含义，双引号忽略大多数特殊字符，但不会忽略 `$`、`\`、反引号。反引号的作用是命令替换

```bash
$ a=abc
$ echo "$a"
abc
$ echo '$a'
$a
```

#### 进程替换与命令替换

命令替换（command substitution）的写法如下：

```bash
$ $(CMD)
$ `CMD`
$ # 例子
$ begin_time=`date`
```

其运行逻辑是将 `date` 视为一条命令执行，将输出替换掉 ``` `date` ```。

备注：

- 反引号的写法在任何类型的 shell 中都是通用的，但 `$(CMD)` 的写法只在 bash 中有效。但 `$()` 允许嵌套，而反引号的写法不允许嵌套。

- 如果被替换的命令输出有多行或者有连续的多个空格，那么使用 echo 命令进行输出时，若不用双引号将变量名包裹起来，那么换行符以及连续的空格将会被替换为系统默认的空白符，例如：

  ```bash
  $ a=`ls`
  $ echo a
  abc def
  $ echo "a"
  abc
  def
  ```

进程替换（process substitution）的写法如下：

```
diff <(ls a) <(ls b)
```

其运行逻辑是：运行 `ls a`，将结果存入一个临时文件，并用临时文件名替换掉 `<(ls a)`，也就是相当于：

```
ls a > tmp1.txt
ls b > tmp2.txt
diff tmp1.txt tmp2.txt
```

#### `&&`、`||`、`;`

这三者统称为 list operators for separating shell commands

```
command1 && command2  # command1正常执行时, command2才被执行
command1 || command2  # command1执行异常时, command2才被执行
command1 ; command2   # 无论command1执行是否正常, command2d
```

## 第 2 课：shell 脚本

### 基础语法

#### 怎么运行脚本

假定 `script.sh` 的内容如下：

```bash
#!/bin/bash
echo "hello"
```

**赋予执行权限**

读、写、运行的权限分别为 4、2、1，`chmod` 命令的三个数字依次代表拥有者、用户组、其他人的权限，权限数字为三项权限之和。因此 `744` 代表拥有者具有读、写、运行权限，用户组具有读权限，其他人具有读权限。

```bash
chmod 744 script.sh
```

**运行**

第一种运行方式为：`解释程序名+脚本名`，这种方式下当前用户对脚本不需要有可执行权限。

```bash
sh script.sh
```

第二种运行方式为：`脚本名`，这种方式下当前用户对脚本必须有可执行权限。

注意使用这种运行方式时，解释程序由第一行的 `#!/bin/bash` 决定，这一特殊行被称为 shebang 行。

```bash
./script.sh
```

因此，对于 python 脚本来说，也可以使用第二种方式运行。shebang 行的最佳实践写法是：

```bash
#!/bin/usr/env python
```

**调试**

可以使用 `bash -x 脚本名` 的方式进行调试，也可以使用类似下面的方式进行自定义调试：

```bash
#!/bin/bash
#test.sh
function DEBUG(){
  [ "$_DEBUG" == "on" ] && $@ || :
}

for i in {1..10}
do
  DEBUG echo $i
done
```

```bash
$ _DEBUG=on ./test.sh  # 调试模式
$ ./test.sh  # 非调试模式
```

#### 变量

注意点：

- 定义变量或者给变量赋值时，等号的左右两端不能有空格

  - 使用算术运算的形式为：`$[var1+var2]`

  - 使用命令的执行结果为变量赋值

    ```
    a=$(CMD)
    s=`CMD`
    ```

- 变量的值可以都是字符串

- 引用变量的形式为：`$var` 或者 `${var}`，两种写法是一样的，为了避免出错，建议使用后者

```bash
a=1
b=2
c=$[a+b]  # c=3
c=$a$b"34"  # c=1234，字符串拼接操作
c=$(wc -l a.py | cut -d " " -f1)  # 计算 a.py 文件的行数并存入变量 c 中
echo "$a+$b=${c}"
unset c  # 删除变量 c
```

#### 特殊变量

shell 脚本与其他脚本的特殊之处在于 shell 脚本中有许多特殊的预设变量

- `$0`：执行脚本名（不算做脚本参数）。注意：在调用函数时，指的是 `函数名.sh`
- `$1` - `$9`：脚本的参数，第 1-9 个参数。注意：在调用函数时，指的是函数的参数
- `$@`：脚本的所有参数（不包括 `$0`）。注意：在调用函数时，指的是函数的所有参数
- `$#`：脚本的参数个数（不包括 `$0`）
- `$?`：前一条命令的返回值（上一条命令如果正确执行了，返回值为 0，否则不为 0）
- `$$`：当前脚本的进程号
- `!!`：完整的上一条命令
- `$_`：上一条命令的最后一个参数
- `${@:2}`：指的是第 `$2` 至之后所有的参数（包含 `$2`），此语法只在用 bash 执行时有效，使用 sh 执行时会报错

#### 数组变量

数组的定义与使用的方式如下

```bash
$ arr1=(1 2 3 4 5 6)
$ arr2[0]=a
$ arr2[1]=b
$ echo ${arr1[1]}
$ echo ${arr2[1]}
$ echo ${arr1[*]}  # 打印数组中所有元素
```

bash 4.0 以后，引入了关联数组，即：字典。

```bash
$ declare -A ass_arr  # 必须先声明为关联数组
$ ass_arr=([a]=1 [b]=2)
$ echo ${ass_arr[a]}
```

列出所有元素/索引

```bash
$ echo ${arr1[*]}  # 列出数组所有元素
$ echo ${!arr1[*]}; echo ${!arr1[@]}  # 列出索引
$ echo ${!ass_arr1[*]}; ${!ass_arr1[@]}  # 列出索引
```



#### 条件语句

**test 命令**

test 的作用是检测某个条件是否成立，例如：

```bash
$ test -f a.txt  # 判断 a.txt 是否存在且为常规文件
$ test $a -eq $b  # 判断变量 a 与变量 b 是否相等
```

需要注意的是，`test` 命令没有输出，当条件成立时，返回状态码为 0；条件不成立时，返回状态不是 0。

test 命令还有一种“语法糖”的形式更为常见，<font color=red>注意左右中括号的空格是不能少的</font>：

```
$ [ -f a.txt ]
$ [ $a -eq $b ]
```

**条件语句**

```bash
#!/bin/bash
file=log.txt
if [[ -f $file ]];then  # 若不存在 file 时，取值为 true
	echo "$file already exists"
else
	touch $file
	echo "new $file"
fi
num=10
if (( $num < 9 )); then
	echo '$num is less than 9'  # 注意这里用的是单引号
else
	echo '$num is not less than 9'
fi
```

类似于 `[[ -f $file ]]` 这种写法：

- `[ -a FILE ]`: 如果 FILE 存在则为真
- `[ -d FILE ]`: 如果 FILE 存在且为目录则为真
- `[ -r FILE ]`: 如果 FILE 存在且为可读文件则为真
- `[ -w FILE ]`: 如果 FILE 存在且为可写文件则为真
- `[ -x FILE ]`: 如果 FILE 存在且为可执行文件则为真
- `[ -z STRING]`: 如果 STRING 的长度为 0 为真
- `[ -n STRING]`: 如果 STRING 的长度大于 0 为真
- `[ STRING ]`: 字符串不空为真, 类似于 -n
- `[ INT1 -le INT2 ]`: 数值上 INT1 小于 INT2 为真
- `[ ! EXPR ]`: 非
- `[ EXPR1 -a EXPR2]`: 与
- `[ EXPR1 ] && [ EXPR2 ]`: 与
- `[ EXPR1 -o EXPR2]`: 或
- `[ EXPR1 ] || [ EXPR2 ]`: 或

备注：双圆括号一般用于数值比较，写法上更方便，例如：`[ 1 -le 2 ]` 等价于 `(( 1<2 ))`；双中括号有时也是为写法的便利，例如：`[[ $a != 1 || $b = 2 ]]` 这种写法等价于 `[ $a != 1 ] || [ $b = 2 ]`，注意前者必须用双中括号而不能用单中括号。另外，注意要在**中括号内部的左右两边各留一个空格**。

#### 循环语句

```bash
#!/bin/bash
for param in $@; do
  echo $param
done
```

#### 函数

函数定义

```bash
function fname(){
  statements;
  # 可以有返回值
  return val;  # 返回值，可以用 $? 接收上一条的返回值
}
# 或者
fname(){
  statements;
  return val;
}
```

函数使用

```bash
fname;  # 无参数
fname arg1 arg2;  # 带参数
```

#### set

`set` 命令用于改变 shell 脚本的一些执行逻辑，一般写在脚本的开头。例如在默认情况下，遇到未定义的变量名，shell脚本不会报错，而是使用空字符串代替，为此，可以在脚本的最开头设置：`set -u`，使得当在使用未定义的变量时将会直接报错，并退出脚本。类似地，在 `bats` 测试中（参考[博客](https://sipb.mit.edu/doc/safe-shell/#:~:text=%20set%20-o%20pipefail%20causes%20a%20pipeline%20%28for,exit%20if%20any%20command%20in%20a%20pipeline%20errors.)）：

```bash
set -euo pipefail
```

- `set -e` 表示 shell 脚本遇到出错的命令就立刻中止脚本运行（备注：存在一些例外情况，见前面的链接或man手册），如果在设置了需要允许某行命令报错也能正常执行，则可以使用 `<error-command> || true` 或者 `<error-command> || :` 来使得命令出错时也能继续运行。
- `set -u` 表示 shell 脚本在使用未定义的变量时就报错并中止运行
- `set -o` 表示打开特殊选项，对应地，`set +o` 表示关闭特殊选项，此处 `set -o pipefail` 是针对 shell 的一个诡异行为（`set -e` 的例外情况）：在使用管道时，只有在管道的最后一条命令出错时，才认为整条命令出错。而使用了 `set -o pipefail` 表示在管道的任意位置出错则报错
- `set -x` 表示执行时会打印出脚本中的每条语句，并且变量名将被替换为其实际内容（debug 时推荐使用）

### 脚本例子

#### 例 1：定时计数

```bash
#!/bin/bash
echo -n Count:
tput sc  # 保存当前光标位置

count=0
while true; do
  if [ $count -lt 15 ];then
    let count++
    sleep 1
    tput rc  # 将光标返回到上一个存储位置
    tput ed  # 清空当前光标到结尾的所有字符
    echo -n $count;
  else exit 0;
  fi
done
```

#### 例 2：输入密码

```bash
#!/bin/bash
echo -e "Enter password"
stty -echo  # 抑制输出
read password
stty echo  # 显示输出
echo Password read
echo "password is $password"
```

#### 例 3：移动文件

```bash
for subdir in $(ls ./train);
do
	for filename in $(ls ./train/${subdir});
	do
	mv ./train/${subdir}/${filename} ./temp/${filename};
	done
done
```

#### 例 4：修改文件后缀

```bash
for file in $1/*.dat;
do
	mv "$file" "${file%.*}.txt";
done
```

## 第 5 课：命令行环境

### 5.1 Job Control

#### Signal

一般来说，可以使用 `ctrl+c` 来中断程序，又或者使用 `kill -9 <pid>` 杀死进程。这些操作本质上是在给进程发信号（signal）。每个进程在运行中如果接收到信号，那么它必须处理这些信号。而 `kill` 命令的作用就是给进程发信号。

```bash
$ kill -SIGINT <pid>  # 等价于 kill -2 <pid>，也等价于按快捷键ctrl+c，表示发送中断信息
$ kill -SIGKILL <pid>  # 等价于 kill -9 <pid>，表示发送杀死进程的信息
```

完整表格可参见 [man-pages](https://man7.org/linux/man-pages/man7/signal.7.html)

| Signal  | 快捷键 | 默认行为                                   | x86/ARM 所代表的值 |
| ------- | ------ | ------------------------------------------ | ------------------ |
| SIGHUP  |        |                                            | 1                  |
| SIGINT  | ctrl+c | 终止程序运行                               | 2                  |
| SIGKILL |        | 杀死进程                                   | 9                  |
| SIGSTOP | ctrl+z | 暂停进程运行                               | 19                 |
| SIGCONT |        | 继续进程运行                               | 18                 |
| SIGTERM |        | 终止程序运行                               | 15                 |
| SIGQUIT | ctrl+\ | 终止程序运行，且终止程序前会进行 core dump | 3                  |

备注：

- core dump 又叫核心转储，当程序运行过程中发生异常，程序异常退出时，由操作系统把程序当前的内存状况存储在一个 core 文件中。但是否会生成该文件还需要打开一项设定：

  ```bash
  $ ulimit -c 100  # 表示设定允许的core文件的最大大小为100KB
  $ ulimit -c unlimited  # 无限
  $ ulimit -a  # 显示设定，第一行即为core文件大小的设定
  ```

某些信号允许程序自己定义如何进行处理，例如：

```python
#!/usr/bin/env python
import signal, time

def handler(signum, time):
    print("\nI got a SIGINT, but I am not stopping")

signal.signal(signal.SIGINT, handler)
i = 0
while True:
    time.sleep(.1)
    print("\r{}".format(i), end="")
    i += 1
```

使用 `ctrl+c` 发送 `SIGINT` 无法中断程序

```bash
$ python sigint.py
24^C
I got a SIGINT, but I am not stopping
26^C
I got a SIGINT, but I am not stopping
30^\[1]    39913 quit       python sigint.py
```

### 



#### `nohup`、`&`、`jobs` 命令

```
nohup python main.py &
```

- **nohup**：在关闭终端时，会对所有由该终端运行的进程发送 `SIGHUP` 信号，默认情况下，所有进程均会退出。但使用了 nohup 启动命令后，该程序将无法得到 stdin 的输入，并且将 stderr 与 stdout 重定向到 nohup.out 文件中。并且在关闭终端时，该进程不会接受到 `SIGHUP` 信号，也就不会终止。

- **&**：表示将程序放在后台运行（这种进程可由 jobs 命令查看到），输出进程 ID。终端可继续执行其他命令。然而，这种进程依然会对终端进行输出，这可以通过重定向来避免。例如：

  ```
  nohup jupyter-lab --allow-root > jupyter.log 2>&1 &
  ```

  备注：这里的 `> jupyter.log 2>&1` 即为重定向，其中 `> jupyter.log` 是 `1 > jupyter.log` 的简写，表示将标准输出（stdout）重定向至 `jupyter.log` 文件中，而 `2>&1` 表示将标准错误（stderr）重定向至标准输出中，而前者已被重定向，因此全部被重定向至 `jupyter.log` 中。此处的 `2>&1` 中的 `&` 如果不写，则表示将标准输出重定向到一个名为 `1` 的文件中

- **jobs**：`jobs` 命令用于查看当前终端放入后台运行的进程的运行情况。

  ```bash
  $ python sigint.py  # 使用ctrl+Z发送SIGSTOP信号
  12^Z
  $ jobs  # 此处的1为job_id，后续可借由这个值将
  $ # jobs -l 可以查看到这些job的进程id
  [1]+  Stopped                 python sigint.py
  # $ bg 1  # 将job_id为1的进程放在后台继续运行
  # $ fg 1  # 将job_id为1的进程放在前台继续运行
  ```

#### `SIGINT`、`SIGTERM`、`SIGQUIT`、`SIGKILL`：

参考[博客](https://www.baeldung.com/linux/sigint-and-other-termination-signals)，大略意思如下：四者都可用于终止程序运行，`SIGKILL` 是结束进程的强制手段，程序不能改变接受到此信号的行为。而其余三者都可以由程序决定如何处置这些信号。默认行为下，推荐使用 `SIGTERM` 结束进程。

### 5.2 Terminal Multiplexers (tmux)

`ctrl+b` + `alt+方向键`：控制面板大小

`ctrl+b` + `z`：将当前面板扩大到整个窗口（或退出这一模式）

#### 命令环境

按下 `ctrl+b` 后，按下 `:` 进入底线命令模式

```
:resize-pane -D 10  # 向下减少10个单元
:resize-pane -U 10  # 向上增加10个单元
:resize-pane -R 10  # 向右增加10个单元
:resize-pane -L 10  # 向左增加10个单元
```





### 5.3 Alias and Dotfiles

### 5.4 Remote Macheines

## Linux

### Linux 目录结构

Linux 发行版的目录结构由 [Filesystem Hierarchy Standard](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard) 所规定，以 Ubuntu 18.04 为例，大致如下：

```
/
  - bin/
  - sbin/
  - usr/ 
  	- local/
  	  - bin/
  	  - sbin/
  	- bin/
  	- sbin/
  - root/
  - home/
  - dev/  # 设备文件, 例如硬盘：/dev/disk0
  - lib/
  - include/
  - etc/  # editable text config, 存放软件的配置文件
  - var/  # 在使用过程中一直在发生变化的文件，例如日志，临时文件等
  - boot/  # 系统启动所需的文件
  - opt/  # 可选的软件
  - tmp/  # 系统重启时将不被保留的文件
  - proc/  # 一个虚拟的目录, 不被存在磁盘上, 可用于查看进程的信息
```

说明：

- 默认的 `PATH` 环境变量的值通常情况下为

  ```
  /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  ```

  其中，`bin` 与 `sbin` 的区别在于前者适用于所有用户，后者适用于普通用户。而 `/bin` 目录存放的应该是系统所必须的命令或软件，`/usr/bin` 存放的是非系统必须的软件，一般而言，Linux 发行版的“软件商店”（例如 Ubuntu 的apt）里的软件会安装在此处。手动编译或者通过其他渠道获取的适用于所有用户的软件一般放在 `/usr/local/bin` 目录下。

### 管理用户相关

```bash
$ sudo useradd -m -N -s /bin/bash someone
# -m: 自动建立用户的登入目录，默认为/home/someone
# -N：不创建同名群组
# -s：指定shell，如安装了zsh，可指定为/bin/zsh
$ sudo useradd -d /d/ -m -N -s /bin/bash someone
$ passwd someone  # 设定用户密码
```

### 用户登录

[链接1](https://thecodecloud.in/what-happens-when-we-login-logout-in-linux/)，[链接2](https://www.stefaanlippens.net/bashrc_and_others/#:~:text=.bash_profile%20is%20for%20making%20sure%20that%20both%20the,if%20you%20would%20omit.bash_profile%2C%20only.profile%20would%20be%20loaded.)，[链接3](https://bencane.com/2013/09/16/understanding-a-little-more-about-etcprofile-and-etcbashrc/)

#### 各种类型的 shell

根据 [链接4](https://linuxize.com/post/bashrc-vs-bash-profile/) 中所指出的，

> A shell can be interactive or non-interactive

所谓的 interactive shell 即为与用户进行交互的 shell。而 non-interactive shell 指的是不会与用户进行交互的 shell，例如在 shell 中执行

```bash
sh run.sh
```

其执行逻辑是新开一个 non-interactive shell，在该 shell 中执行 `run.sh` 中的命令

进一步，interactive shell 又分为 interactive login shell 与 interactive non-login shell，例子是：使用 `ssh` 远程连接时得到的 shell，使用 `--login` 参数启动的 `bash` 是 login shell；而 non-login shell 则必须在 login shell 中才能打开，例如在一个 shell 中直接输入

```
bash
```

那么将会进入一个全新的 non-login shell（虽然可能不容易察觉，但使用 `ctrl+D` 快捷键便可以退出这个 non-login shell，从而回到原来的 shell）。

备注：各种类型的 shell 也可以在 `/etc/passwd` 文件中可见一斑：

```
buxian:x:1000:1000:buxian,,,:/home/buxian:/bin/bash
systemd-coredump:x:999:999:systemd Core Dumper:/:/usr/sbin/nologin
nvidia-persistenced:x:127:134:NVIDIA Persistence Daemon,,,:/nonexistent:/usr/sbin/nologin
```

**总结如下**

- non-interactive shell：不接受交互式操作的 shell；
- interactive shell：交互式 shell；
  - login shell：需要登陆的shell，例如使用 ssh 登陆；
  - non-login shell：不需要登陆的shell，例如在一个交互式（login 或者 non-login）的 shell 中输入 bash 所得到的 shell。

#### shell 的配置文件

对于 login shell，则登陆时首先查看 `/etc/profile` 是否存在并执行该文件，接下来，按顺序依次查找 `~/.bash_profile`，`~/.bash_login`，`~/.profile` 这三个文件是否存在并且有可读权限，只执行找到的第一个则停止。

对于 non-login shell，依次执行 `/etc/bash.bashrc`（ubuntu 为`/etc/bash.bashrc`，Red Hat 为 `/etc/bashrc`） 以及 `.bashrc`。

【存疑】：对于 non-interactive shell，以上所有的配置文件均不会被执行

备注：

（1）一般而言，`~/.bash_profile` 里会包含这种语句

```
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi
```

（2）`~/.bash_profile` 仅仅适用于 bash shell，而 `~/.profile` 适用于所有的 shell。

备注：bash 的 manpage 中节选如下：

>     When bash is invoked as an interactive login shell, or as a non-interactive shell with the --login option, it first reads and executes commands from the file /etc/profile, if that file exists. After reading that file, it looks for ~/.bash_profile, ~/.bash_login, and ~/.profile, in that order, and reads and executes commands from the first one that exists and is readable. The --noprofile option may be used when the shell is started to inhibit this behavior.
>     
>     When an interactive shell that is not a login shell is started, bash reads and executes commands from /etc/bash.bashrc and ~/.bashrc, if these files exist. This may be inhibited by using the --norc option. The --rcfile file option will force bash to read and execute commands from file instead of /etc/bash.bashrc and ~/.bashrc.

## Ubuntu

## 工具

### ranger

安装
```
apt install ranger
```
使用
```
ranger
```
按上下键在当前目录浏览，enter键进入，esc键退出文件。q键退出ranger


### screenkey

将按键显示在屏幕上

## 杂录（未被划分清楚的记录到这里）

### windows 上传文件中文乱码

```
unzip -O gbk filename.zip
```

