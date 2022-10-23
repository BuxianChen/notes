# shell cheat sheet

## bash 命令

### sed（待补充完整）

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

### dirname

```bash
$ dirname <file|dir> # 返回文件或目录的父目录
```

### tee（待补充）

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

### printf

```bash
$ printf "%-5s %-10s %-4.2f\n" 123 acb 1.23
```

printf 命令仿照 C 语言的 printf 函数，用于格式化输出，格式控制符 `%-5s` 表示左对齐（不加 `-` 则表示右对齐），**最少**使用 5 个字符长度，以字符串的形式输出。格式控制符 `%-4.2f` 表示最少使用 4 个字符长度，小数点后保留 2 位，以浮点数形式输出。

### !

在 shell 中，! 被称为 *Event Designators*，用于方便地引用历史命令。

- `!20` 表示获取 history 命令中的第 20 条指令；
- `!-2` 表示获取 history 命令中的倒数第 2 条指令；
- `!!` 是 `!-1 `的一个 alia；
- `!echo` 表示最近地一条以 `echo` 开头的指令；
- `!?data` 表示最近的一条包含 `data` 的指令

### echo、stty

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

### alias

别名相当于自定义命令，可以使用 alias 命令实现，也可以定义函数实现。此处仅介绍 alias 命令。

```
$ alias myrm='rm -rf'
$ myrm data/
```

alias 命令产生的别名只在当前 shell 有效

### du 与 df（待补充）

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

### watch

```bash
# 每秒钟执行一次nvidia-smi命令以监控GPU使用情况，按ctrl+c退出监控
$ watch -n 1 nvidia-smi
```

### sort

```bash
$ docker images | sort -n -k 7
```

sort 命令的作用是以行为单位进行排序。`-n` 选项表示把字符当作数字进行排序，`-k 7` 选项表示选择第 7 列进行排序。可以使用 `-t :` 来指定列的分割符为 `:`，可以使用 `-r` 选项进行降序排列（默认是升序排列）

### paste

```bash
paste -d= a.txt b.txt > c.txt
```
将 `a.txt` 与 `b.txt` 按行进行拼接，拼接字符为 `=`，即:

```bash
paste -sd+ a.txt | bc
```
`-s`表示把 `a.txt` 的每行进行拼接，拼接字符为 `+`，此条命令用于对一列数字求和


### history（待补充）

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

### mount/umount

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

### ps（待补充）

用于列出所有进程

[参考链接](https://www.jianshu.com/p/943b90150c10)

### nmap

打印本机打开的所有端口信息

```bash
$ # apt install nmap
$ nmap 127.0.0.1
```

### ln

```bash
$ # ln -s 原始路径 目标路径
$ ln -s /home/to/directory /data  # 得到/data/directory
```

### file -i

```bash
$ file -i 文件名  # 查看编码格式
```


### command

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

### getopt

[stackoverflow](https://unix.stackexchange.com/questions/85787/invoking-shell-script-with-option-and-parameters)


### rsync

参考[阮一峰博客](https://www.ruanyifeng.com/blog/2020/08/rsync.html)

```
rsync -anv source/ destination  # 测试, 不实际执行
rsync -av source/ username@remote_host:destination
rsync -av --delete source/ destination  # 确保两个目录完全一致
```

注: 
- 此处 `source` 后面的斜杠如果省略, 则目标位置将会形成 `destination/source` 的目录结构, 这可能不是所期望的
- rsync 命令默认会同步以 `.` 开头的隐藏目录


### awk


**示例1**

注意: 如果原始数据中以`\t`作为分割符, 但某些列可能为空字符串, 则必须指定`\t`
```
aaaa  bbbbb
  ccccc
```

```python
awk -F'\t' '{print $1"\t"$2}' data.csv
```

**示例2: 引入外部变量**

```python
var="123"
awk -v v=${var} '{print var" "$0}' data.csv
```

**示例3: if**
```python
awk '{if($2==23) {print $0}}' data.csv
```


## shell命令组合例子

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

#### 例 11: 跳过前 k 行

```
# tail -n +<N+1> <filename>
tail -n +3 a.txt  # 跳过前2行
```

## Shell 脚本示例 

### 例 1：定时计数

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

### 例 2：输入密码

```bash
#!/bin/bash
echo -e "Enter password"
stty -echo  # 抑制输出
read password
stty echo  # 显示输出
echo Password read
echo "password is $password"
```

### 例 3：移动文件

```bash
for subdir in $(ls ./train);
do
	for filename in $(ls ./train/${subdir});
	do
	mv ./train/${subdir}/${filename} ./temp/${filename};
	done
done
```

### 例 4：修改文件后缀

```bash
for file in $1/*.dat;
do
	mv "$file" "${file%.*}.txt";
done
```