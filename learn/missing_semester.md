# CS课程中缺失的一课

课程主页：https://missing.csail.mit.edu/

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

#### dirname

```bash
$ dirname <file|dir> # 返回文件或目录的父目录
```

#### tee

```bash
$ tee a.txt b.txt  # 同时向 a.txt 和 b.txt 中写入相同的内容，输入这行命令后，需要继续输入要写入的内容，以 Ctrl+Z 结束。注意 tee 命令写入的东西同时也会打印至屏幕上。
```

一般使用管道的方式进行使用，例如

```bash
$ echo "abc" | tee a.txt b.txt
abc  # 写入文件并同时将写入的信息输出至标准输出流
$ echo "abc" | tee a.txt > b.txt  # 用重定向的方式同时写入两个文件，并且不显示在屏幕上
```

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

cut 命令用于切分字符串，有若干种用法：取出第 $$m$$ 个到第 $$n$$ 个字符；按分隔符取出第 $$k$$ 个字符串。此处 cut 命令之前的

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

#### 例 6：grep

#### 例 7：修改屏幕亮度

```bash
$ echo 5000 | sudo tee /sys/class/backlight/intel_backlight/brightness
# 不能使用 sudo echo 5000 > /sys/class/backlight/intel_backlight/brightness
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

## 第 3 课：vim

备注：<font color=red>按键严格区分大小写</font>

### vim 的编辑模式

主要操作模式为：

- 正常模式：在文件中四处移动光标进行修改的模式。进入vim时处于的模式
- 插入模式：插入文本
- 替换模式：替换文本
- 可视化模式：进一步分为一般、行、块模式，主要是移动光标选中一大块文本
- 命令模式：用于执行命令

#### 模式的切换方式

以正常模式为“中心模式”，使用 `<ESC>` 键从任何其他模式返回正常模式。在正常模式下：使用 `i` 键进入插入模式，使用 `R` 键进入替换模式，使用 `:` 键进入命令模式，使用 `v` 键进入可视化（一般）模式，使用 `V` 键进入可视化（行）模式，使用 `Ctrl+v` 进入可视化（块）模式。

### vim 界面：缓存（cache），标签页（tab），窗口（window）

> Vim 会维护一系列打开的文件，称为“缓存”。一个 Vim 会话包含一系列标签页，每个标签页包含一系列窗口（分隔面板）。每个窗口显示一个缓存。跟网页浏览器等其他你熟悉的程序不一样的是， 缓存和窗口不是一一对应的关系；窗口只是视角。一个缓存可以在多个窗口打开，甚至在同一 个标签页内的多个窗口打开。这个功能其实很好用，比如在查看同一个文件的不同部分的时候。
>
> Vim 默认打开一个标签页，这个标签也包含一个窗口

`:sp [filename]` 表示在下方新建一个窗口，打开 `filename` 文件

`ctrl+w` + `h/j/k/l` 切换至左/下/上/右的窗口

`:wq [filename]` 表示关闭当前窗口，并将当前窗口的文件更名为 `filename`

`:tabnew [filename]` 表示新建一个标签页，打开 `filename` 文件

`gt` 表示切换至下一个标签页



### 各模式下的基础操作

#### 正常模式

正常模式下，光标的显示方式为块状。

| 命令   |                |      |
| ------ | -------------- | ---- |
| u      | 撤销操作       |      |
| ctrl+r | 重做上一个操作 |      |
| .      | 重复上一个操作 |      |

第一类操作为**移动**，也被称为**名词**。

- 使用 `hjkl` 分别代表左、下、上、右移动光标，当然也可以使用方向键；

- 词：`w` 表示移动到下一个词首，`b` 表示移动到当前词的词首，`e` 表示移动到当前词的词尾；
  
  - 备注（不重要的细节）：如果当前光标停在词尾，那么 `e` 键将会移动到下一个词的词尾；`b` 键同理。
  
- 行：`0` （数字零）表示移动到行首，`^` 键移动到该行第一个非空格位置，`$` 键移动改行的行尾；

- 屏幕：`H` 表示屏幕首行，`M` 表示屏幕中间，`L` 表示屏幕底部；

- 翻页：`Ctrl+u` 表示上翻一页，`Ctrl+d` 表示下翻一页；

- 文件：`gg` 表示移动到文件开头，`G` 表示移动到文件结尾；

- 行数：`{数字}G` 表示移动到某一行，例如：`10G` 表示移动到文件的第10行

- Find: `f{character}`, `t{character}`, `F{character}`, `T{character}`

  - find/to forward/backward {character} on the current line
  - `,` or `;` for navigating matches

  `f/F` 表示在当前行查找下一个字符/上一个字符 character。`t/T`表示查找下一个字符character的前一个字符/前一个字符 character 的后一个字符。

- Search: `/{regex}`, `n` or `N` for navigating matches。从当前光标位置寻找符合正则表达式 `regex` 的字符串，`n` 表示寻找下一个，`2n` 表示寻找往后第二个，`N` 表示向前查找

以下操作暂时不明白，先摘录

- Misc: `%` (corresponding item)

#### 插入模式

插入模式下，光标的显示方式为块状。键入字符表示在光标停留位置增加一个字符，而原始光标处及该行之后的字符往后移动一位，且新光标位置也向后移动一位。

例子：

```
abcd
```

假定当前光标位置在 c 处，键入 `f`，那么会变为

```
abfcd
```

且光标位置依然在 c 处。

#### 替换模式

插入模式下，光标的显示方式为块状。键入字符表示将光标停留位置做字符替换，新光标位置向后移动一位。

例子：

```
abcd
```

假定当前光标位置在 c 处，键入 `f`，那么会变为

```
abfd
```

且光标位置移动至 d 处。

#### 命令模式

备注：保存/退出等都是针对当前窗口（window）的

- `:w` 表示保存
- `:q` 表示退出（前提是所有更改都已经保存了），`:q!` 表示不保存（不保存从上一次保存之后的更新）退出
- `:wq` 表示保存并退出
- `:e! [filename]` 表示放弃从上一次保存之后的更新，[并打开 `filename` 文件进行编辑]。

#### 可视化模式

待补充

### IDE



## 第 6 课：Git

本节课的讲法是先大致讲清 Git 的数据模型（实现），再讲操作命令。个人认为很受用，强烈推荐。这里仅记录 Git 数据模型等内部相关的东西，关于 Git 的使用参见[这里](../tools/git.md)。

一个数据模型的例子如下：

```
<root> (tree)
|
+- foo (tree)
|  |
|  + bar.txt (blob, content = "hello world")
|
+- baz.txt (blob, content = "git is wonderful")
```

### blob、tree、commit、object、reference

注：这些东西都可以在 `.git` 目录中探索。

在 Git 中，文件被称作 blob。目录被称作 tree。commit 包含了父 commit，提交信息、作者、以及本次提交的 tree 等信息。object 则可以是前三者的任意一个。进一步，对每个 object 进行哈希，用哈希值代表每个 object。注意 object 是不可变的，而 reference 是一个映射（指针），字符串（例如 master）映射到 object 的哈希值。所以，一个仓库可以看作是一堆 object 与一堆 reference（即objects 与 references），Git 命令大多数是在操作 object 或者 reference，具体地说，是增加 object ；增加/删除/修改 reference（即改变其指向）。伪代码如下：

```
type blob = array<byte>;
type tree = map<string, blob | tree>;
type commit = struct {
	parent: array<commit>
	author: string
	message: string
	snapshot: tree
}
type object = blob | tree | commit
objects = map<string, object> // objects[hash(object)] = object
// 只提供用 hash 值查找 object 以及对 objects 增加的接口
references = map<string, string> // 前一个 string 表示指针名，例如：master, 后一个 string 表示 object 的哈希值，这里的指针名的例子是：分支名、HEAD
```

### workspace、stage、version

以下参杂个人理解：这三者实际上都是一个版本/快照，而所谓的快照基本上等同于一个 tree/commit 对象。

workspace 表示工作区，也就是 `.git` 目录以外的所有内容，workspace 可以看作是一个快照；

stage 是为了方便用户使用的一个机制，比如说开发了一个新特性，将其放入缓冲区，之后再增加了一些调试代码，那么提交时可以不提交调试代码。stage 中的信息被保存在了 `.git/INDEX` 文件内，stage 可以看作是一个快照；

version 则是历史提交的版本，因此实际上是若干个快照。

许多命令例如：`git add`，`git diff`，`git restore` 实际上就是利用上述三者之一修改/比较另外一个或多个快照。

### branch、HEAD

每个 branch 都是一条直线的提交序列（一系列的 commit 对象），每个 branch 都有一个名字，例如 master、dev 等，它们指向其中的一次提交。`git commit` 命令实际上的作用是生成一个 commit 对象后，将生成的 commit 对象的父亲设置为当前的分支名指向的提交，而后将分支名指向的对象设置为新生成的对象，伪代码如下：

```python
def git_commit(branch_name):
	new_commit = make_commit(stage)
	old_commit = branch_name.current_commit
	new_commit.parent = old_commit
	branch.current_commit = new_commit
```

而 HEAD 指的是当前状态下的最近一次的提交。通常情况下，HEAD 总是与某个分支名的指向相同。但在某些情况下，HEAD 指向的提交不与任何的分支名的指向一致，称为 detached 的状态，例如使用类似如下的方式切换分支：

```bash
git checkout a7141d0
git checkout HEAD~3
git checkout master~3
```

本质上而言，分支名与 HEAD 都是 reference 对象中的元素，存储的都是一个特定的 commit_id。而整个版本库存放的东西无外乎是一堆 commit 对象（每个 commit 对象包含指向其父亲的指针以及一个 tree 对象）。

### `.git` 目录

依次执行

```bash
git config --global user.name "BuxianChen"
git config --global user.email "541205605@qq.com"
git init
echo "abc" > a.txt
git add .  # add_1
git commit -m "a"  # commit_1
mkdir b
echo "def" > b.txt
git add .  # add_2
git commit -m "b"  # commit_2
git branch dev  # branch_1
```

得到如下目录（省略了一些目录及文件）

```shell
.git
│  HEAD  # 文件内容是 refs/heads/<分支名>
│  index  # add_1, add_2
│  ...
├─logs
│  ...
├─objects
│  ├─24
│  │      c5735c3e8ce8fd18d312e9e58149a62236c01a  # blob (./b/b.txt), add_2
│  ├─3e
│  │      bc756fee46dfcb9410ab7f07980a8ff0e71d82  # commit, commit_2
│  ├─43
│  │      8e5d5f895ccf4910e1a463ff5f31e52c28df3c  # tree (./), commit_2
│  ├─83
│  │      edaf0d7f419929b1b0b84c8a7550f38daf97ac  # tree (./b), commit_2
│  ├─8b
│  │      3d54f8c5d0ebd682ea6e83386451e96a541496  # tree (./), commit_1
│  │      aef1b4abc478178b004d62031cf7fe6db6f903  # blob (./a.txt), add_1
│  ├─f7
│  │      496edd08d97d10773a6a76eabd9d24d96785c2  # commit, commit_1
└─refs
    ├─heads
    │      dev  # branch_1, 文件内容是某个 commit 的哈希值
    │      master  # 文件内容是某个 commit 的哈希值
    └─tags
```

注释项代表该条命令运行之后生成了该文件。除了 `objects` 目录以及 `index` 文件外，其余均为文本文件。为了获取 `objects` 目录及 `index` 文件的内容，可以使用以下两行命令：

```bash
git cat-file -p 24c5735c3e8ce8fd18d312e9e58149a62236c01a  # 查看 objects 目录下的文件内容
git ls-files -s  # 查看当前缓冲区内容, 即 .git/index 中的内容
```

结论：

- `git add` 命令只会生成 blob 对象
- `git commit` 命令会同时生成 tree 和 commit 对象
- `HEAD` 指向某个分支名，`git checkout <分支名>` 会同时修改 `HEAD` 及 `index` 的内容，并且切换分支时 `git status` 的结果必须为 `clean`，否则无法执行。

### object 的 hash 值

一个 blob 对象的hash值的计算过程如下：

假定 `readme.txt` 文件内容为`123`，它被 `git add` 的时候，`objects` 目录下会增加一个以二进制序列命名的文件

```text
d8/00886d9c86731ae5c4a62b0b77c437015e00d2
```

一共40位（加密算法为SHA-1），其中前两位为目录名，后38位为文件名

使用 python 可以用如下方式计算出来：

```python
import hashlib
# `header`+内容计算
# `header` = 文件类型+空格+文件字节数+空字符
hashlib.sha1(b'blob 3\0'+b'123').hexdigest() # d800886d9c86731ae5c4a62b0b77c437015e00d2

hashlib.sha1(b'blob 5\0'+'12中'.encode("utf-8")).hexdigest() 
# ec493cf5f7f9a5a205afbc80d7f56dbb34b10600

# len('12中'.encode("utf-8"))
# '12中'.encode("utf-8")
```

## 补充 1：Docker

参考链接：https://yeasy.gitbook.io/docker_practice/

### 基本概念：镜像、容器、仓库

待补充

### 针对镜像的操作命令：

**docker image ls**

列出本地镜像

```bash
$ docker image ls nginx
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
nginx                       latest              e43d811ce2f4        5 weeks ago         181.5 MB
```

**docker pull**

`docker pull` 用于将远程 Docker Registry 的镜像下载到本地（对比 Git 命令：`git clone https://github.com/BuxianChen/notes.git`）

```bash
$ docker pull [选项] [Docker Registry 地址[:端口号]/]仓库名[:标签]
# docker pull 172.22.24.223/username/softwarename:v1
```

`172.22.24.223` 为 Docker Registry 地址；端口号为默认值；仓库名一般为两段式的，`<用户名>/<软件名>`；标签名为 `v1`。

```bash
$ docker pull ubuntu:18.04
18.04: Pulling from library/ubuntu
92dc2a97ff99: Pull complete
be13a9d27eb8: Pull complete
c8299583700a: Pull complete
Digest: sha256:4bc3ae6596938cb0d9e5ac51a1152ec9dcac2a1c50829c74abd9c4361e321b26
Status: Downloaded newer image for ubuntu:18.04
docker.io/library/ubuntu:18.04
```

注意观察输出信息的最后一行的。此处不指定 Docker Registry 地址，则默认为 docker.io，此处没有指定用户名，对于 docker.io 来说，默认为 library。

**docker tag**

`docker tag` 命令的作用是为镜像重命名

```bash
$ docker tag 镜像的旧名字/镜像ID 镜像的新名字
```

常见的使用场景是更名后用于推送镜像：

```bash
$ docker tag ubuntu:18.04 username/ubuntu:18.04
$ docker login
$ docker push username/ubuntu:18.04
```

**docker login**

```bash
$ docker login  # 登录以获取拉取/推送镜像的权限
```

**docker push**

```bash
$ docker push 镜像ID
```

将镜像推送至远端 Docker Registry。

### 针对容器的操作命令

**docker run**

`docker run` 用于利用已有的本地镜像创建容器并运行容器。容器具有运行和终止两种状态。命令形式为：

```bash
$ docker run [参数列表] 镜像名/镜像ID [命令]
```

表示启动后容器运行的命令（**Docker 容器的哲学是一个 Docker 容器只运行一个进程**）。若不指定命令，默认为镜像创建的 Dockerfile 中的最后一个 `CMD` 语句或 `ENTRYPOINT` 语句（`CMD` 与 `ENTRYPOINT` 语句在 Dockerfile 中只能有一句，出现多句则以最后一条为准），默认情况下（不使用 `-d` 参数时），运行完命令后容器就会进入终止的状态。`docker run` 命令的例子如下：

使用以下命令运行完后会自动终止容器

```bash
$ docker run ubuntu:18.04 /bin/echo 'Hello world'
Hello world
```

使用以下命令运行后会启动一个终端，进入交互模式。其中，`-t` 选项让 Docker 分配一个伪终端（pseudo-tty）并绑定到容器的标准输入上， `-i` 则让容器的标准输入保持打开。进入交互模式后，使用 `exit` 命令或者 `Ctrl+d` 快捷键会终止容器。

```bash
$ docker run -t -i ubuntu:18.04 /bin/bash
```

与上一条命令不同的是，添加了 `--rm` 参数后，此时推出交互模式不仅会终止容器，还会将容器删除。

```bash
$ docker run -it --rm ubuntu:18.04 bash
```

**最常见的使用情形是：需要让 Docker 在后台运行而不是直接把执行命令的结果输出在当前宿主机下。**此时，可以通过添加 `-d` 参数来实现。注意：`-d` 参数与 `--rm` 参数含义刚好相反，因此不能同时使用。

```bash
$ docker run -d ubuntu:18.04 /bin/sh -c "while true; do echo hello world; sleep 1; done"
77b2dc01fe0f3f1265df143181e7b9af5e05279a884f4776ee75350ea9d8017a
```

使用 `-v` 参数可以实现宿主机与容器内部目录的挂载，注意挂载的目录在执行 `docker commit` 命令时不会被保存。

**docker container ls**

使用 `-d` 参数启动后会返回一个唯一的 id，也可以通过 `docker container ls` 命令来查看容器信息。

```bash
$ docker container ls
CONTAINER ID  IMAGE         COMMAND               CREATED        STATUS       PORTS NAMES
77b2dc01fe0f  ubuntu:18.04  /bin/sh -c 'while tr  2 minutes ago  Up 1 minute        agitated_wright
```

**docker container logs**

要获取容器的输出信息，可以通过 `docker container logs` 命令。

```bash
$ docker container logs [container ID or NAMES]
hello world
hello world
hello world
...
```

**docker container start/restart/stop**

重新启动已经终止的容器/将一个运行态的容器关闭并重新启动它/将一个运行态的容器终止

```bash
$ docker container start [container ID or NAMES]
$ docker container restart [container ID or NAMES]
$ docker container stop [container ID or NAMES]
```

**docker attach/exec**

进入一个正在运行的容器。

```bash
$ docker run -dit ubuntu
243c32535da7d142fb0e6df616a3c3ada0b8ab417937c853a9e1c251f499f550
$ docker attach 243c
root@243c32535da7:/#
```

注意：使用 `docker attach` 时，退出这个终端时，该容器会终止。

```bash
$ docker run -dit ubuntu
69d137adef7a8a689cbcb059e94da5489d3cddd240ff675c640c8d96e84fe1f6
$ docker exec -it 69d1 bash
root@69d137adef7a:/#
```

注意：使用 `docker exec` 时，该容器不会因为终端的退出而终止。

**docker stats**

以下命令用于查看容器的内存占用等情况

```bash
$ docker stats 容器ID
```

**docker commit**

```bash
$ docker commit -a "author_name" -m "description" 容器ID 镜像名
$ # docker commit 172.22.24.223/username/softwarename:v1
```

将容器的当前状态提交为一个新的镜像，注意挂载目录不会被提交到新镜像内。使用 docker commit 得到镜像的工作流程为：

```bash
$ docker run -it -v 本地目录绝对路径:挂载至容器内的目录 镜像ID --name 自定义容器名字 /bin/bash
$ # 在容器内修改文件, 安装相关的包等
```

修改完毕后，新打开一个终端（也许可以直接退出容器，直接在当前终端操作）

```bash
$ docker commit 自定义容器名字 镜像名
```

**注意：不推荐使用 docker commit 的方式得到镜像，应尽量使用 Dockerfile 制作镜像。**

### 使用 Dockerfile 制作镜像

**例子**

假定本机的 ngnix 镜像如下：

```bash
$ docker image ls nginx
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
nginx                       latest              e43d811ce2f4        5 weeks ago         181.5 MB
```

编写 `Dockerfile` 文件，其内容为

```
FROM nginx
RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
```

进入到 `Dockerfile` 文件所在目录，执行如下命令进行构建

```bash
$ docker build -t nginx:v3 .
Sending build context to Docker daemon 2.048 kB
Step 1 : FROM nginx
 ---> e43d811ce2f4
Step 2 : RUN echo '<h1>Hello, Docker!</h1>' > /usr/share/nginx/html/index.html
 ---> Running in 9cdc27646c7b
 ---> 44aa4490ce2c
Removing intermediate container 9cdc27646c7b
Successfully built 44aa4490ce2c
```

其输出内容的解释如下：这里的 `e43d811ce2f4` 为基础镜像 nginx 的镜像 ID，而后利用该镜像运行了一个容器 ID 为 `9cdc27646c7b` 的容器，之后运行命令，创建好新的镜像，其镜像 ID 为 `44aa4490ce2c`，并删除了刚刚运行的临时容器 `9cdc27646c7b`。

备注：构建命令的最后一个 `.` 被称为上下文路径，其作用与准确理解参见[这里](https://yeasy.gitbook.io/docker_practice/image/build)。

可以用如下命令以刚刚创建的镜像构建一个容器并运行该容器，并将这个运行的容器取名为 `web3`，`-p 81:80` 表示将宿主机的端口 `81` 与容器端口 `80` 进行映射，`-d` 表示保持容器在后台一直运行。

```bash
$ docker run --name web3 -d -p 81:80 nginx:v3
```

这样可以使用浏览器访问 `<宿主机IP地址>/81`。

备注：`docker run` 实际等效于 `docker start` 加上 `docker exec` 两条命令

## 补充 2：Linux

### 用户相关

```bash
$ sudo useradd -m -N -s /bin/bash someone
# -m: 自动建立用户的登入目录，默认为/home/someone
# -N：不创建同名群组
# -s：指定shell，如安装了zsh，可指定为/bin/zsh
$ sudo useradd -d /d/ -m -N -s /bin/bash someone
$ passwd someone  # 设定用户密码
```

