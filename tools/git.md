# Git

用例子解释git命令

注意, 测试2.1与2.2表示的是在测试1的基础上尝试两种做法的结果

例子1（本例着重解释了`git status/diff/restore`三个命令）：

```text
测试流程，一共有4条路径
1 --- 2.1
  --- 2.2
  --- 2.3 --- 2.3.1
          --- 2.3.2
```

```text
git init
# 新建一个README.txt, 添加内容line one
git add README.txt
git commit -m "first commit"
# 在README.txt中添加内容line two
git add README.txt
# 在README中添加内容line three
```

测试1

```text
$ git status
On branch master
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.txt

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.txt

$ git diff
diff --git a/README.txt b/README.txt
index 4c00b39..fa58e34 100644
--- a/README.txt
+++ b/README.txt
@@ -1,2 +1,3 @@
 line one
-line two
\ No newline at end of file
+line two
+line three
\ No newline at end of file

$ git diff --staged
diff --git a/README.txt b/README.txt
index 017bf5c..4c00b39 100644
--- a/README.txt
+++ b/README.txt
@@ -1 +1,2 @@
-line one
\ No newline at end of file
+line one
+line two
\ No newline at end of file
```

总结:

`git status`命令的输出分为两部分:

* `Changes not staged for commit`: 显示工作区相对暂存区的变动记录
* `Changes to be committed`: 显示暂存区相对最近一次提交的变动记录

相应地, `git diff`命令地解释如下:

```text
git diff # 显示工作区相对于暂存区地修改
git diff --staged # 显示暂存区相对最近一次提交的修改
```

测试2

总结: 接下来, 在操作之前先解释`git restore`的两种用法

```text
git restore --staged <file> # 将<file>的暂存区记录删除, 即暂存区恢复至最近提交状态
git restore <file> # 将工作区<file>恢复至暂存区的状态
```

测试2.1

```text
$ git restore README.txt # 此时工作区的README回到只有两行文字的状态
$ git status
On branch master
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.txt
```

测试2.2

```text
# 注意工作区文件不变, 暂存区文件回到最近提交的状态
$ git restore --staged README.txt

$ git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.txt

no changes added to commit (use "git add" and/or "git commit -a")

# 注意到显示的修改是文件增加了两行内容
$ git diff
diff --git a/README.txt b/README.txt
index 017bf5c..fa58e34 100644
--- a/README.txt
+++ b/README.txt
@@ -1 +1,3 @@
-line one
\ No newline at end of file
+line one
+line two
+line three
\ No newline at end of file

# 执行完毕后, README.txt恢复至一行内容的状态
$ git restore README.txt
```

测试2.3

总结: `git restore`还有第三种用法如下

```text
# 将<file>恢复至版本库中的某个版本
git restore --source/-s 7173808e <file>
git restore --source/-s HEAD <file>
```

测试如下:

```text
# 执行完后工作区的README.md只有一行内容
$ git restore --source HEAD README.txt

54120@DESKTOP-7LQFJM3 MINGW64 ~/Desktop/git_test (master)
$ git status
On branch master
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.txt

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   README.txt

# 注意git diff显示的是工作区相对暂存区的差异, 这里表示工作区比暂存区少了一行
$ git diff
diff --git a/README.txt b/README.txt
index 4c00b39..017bf5c 100644
--- a/README.txt
+++ b/README.txt
@@ -1,2 +1 @@
-line one
-line two
\ No newline at end of file
+line one
\ No newline at end of file

# 注意git diff --staged显示的是工作区相对最近提交的差异, 这里表示工作区比最近提交多了一行
$ git diff --staged
diff --git a/README.txt b/README.txt
index 017bf5c..4c00b39 100644
--- a/README.txt
+++ b/README.txt
@@ -1 +1,2 @@
-line one
\ No newline at end of file
+line one
+line two
\ No newline at end of file
```

测试2.3.1

```text
$ git restore --staged README.txt
$ git status
On branch master
nothing to commit, working tree clean
```

测试2.3.2

根据上面的总结, 解释如下过程:

```text
$ git restore README.txt

$ git status
On branch master
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.txt
# 没有输出
$ git diff

$ git diff --staged
diff --git a/README.txt b/README.txt
index 017bf5c..4c00b39 100644
--- a/README.txt
+++ b/README.txt
@@ -1 +1,2 @@
-line one
\ No newline at end of file
+line one
+line two
\ No newline at end of file

$ git restore --staged README.txt
$ git restore README.txt
# 恢复至工作区, 暂存区均只有一行内容的状态
$ git status
On branch master
nothing to commit, working tree clean
```

例子2（本例着重解释`git rm`命令）

备注: 本机已经准备好了如下环境

```text
git init
# 新建一个README.txt, 添加内容line one
git add README.txt
git commit -m "first commit"
```

## git/github/gitlab使用

先明确几者的关系: ${\color{red} 待补充}$

### 本地git

以下概念非常重要:

* **工作区\(Working Directory\)**: 除`.git`文件夹以外文件
* **版本库\(Repository\)**: `.git`文件夹之内的文件
  * **暂存区\(stage/index\)**: `git add`命令是将**文件修改**从工作区提交到暂存区; `git commit`是把**文件修改**从暂存区提交到当前分支, 并将暂存区的文件修改清空
  * ...

简单理解: 工作区里存放着一个某个特定的版本\(注意不一定是最新版本\), 而版本库里存放着所有已经提交过的版本.

```text
# 注意: 所有与git有关的命令最好用git bash打开, 当然, 将git.exe所在目录加入到了path环境变量后
# 使用普通的shell(例如: cmd, powershell, bashrc等基本也都没问题)

# 建立身份信息
git config --global user.name "Your Name"
git config --global user.email "email@example.com"

# 初始化git
git init
# 其效果是为当前目录建立一个.git目录, 正常情况下不要去修改这个文件夹下的任何内容, 可以这样理解:
# 之后的每一条以git开头的命令执行后, git.exe会依据命令内容对.git目录下的文件进行修改
# 注意, git init命令不一定需要在空目录, 另外若.git目录已存在, 使用git init命令的结果是:
# ##待补充##


git add readme.txt


git commit -m "add readme file"

# 查看仓库的状态
git status

# 
git diff
# ##待补充: 查看工作区与版本库中最新版本的区别[工作区--add-->暂存区--commit-->当前分支]?##
git diff HEAD -- readme.txt

# 显示历史信息
git log
git log  --pretty=oneline  # 单行显示历史信息
# 注意输出顺序从上到下: 由新版本到旧版本
# 输出:
# 5ef5acd712370e5a8688e07576ff8f19cfe8a1cd (HEAD -> master) append GPL
# 3b34b144575c6cac0a69e4da91e1f7a6244ee16a add distributed
# 271efb47fac32ff15443aac275782613d6153830 wrote a readme file

# 回退版本
# 注意: head是指当前分支的当前版本
git reset --hard HEAD^  # 注意此时, 若输入git log, 就只有两个历史版本了
git reset --hard 5ef5  # 

# 查看历史命令, reflog译为回流
# 注意输出的每一行前面的数字串是执行该条命令后的版本号
# 注意输出顺序由上到下: 由最新的命令到旧命令,
# 注意不是所有的命令都会保存, 只保留引起head发生变化的命令
git reflog
# 输出
# 5ef5acd (HEAD -> master) HEAD@{0}: reset: moving to 5ef5
# 3b34b14 HEAD@{1}: reset: moving to HEAD^
# 5ef5acd (HEAD -> master) HEAD@{2}: commit: append GPL
# 3b34b14 HEAD@{3}: commit: add distributed
# 271efb4 HEAD@{4}: commit (initial): wrote a readme file

# 将工作区回退到暂存区或版本库其中之一, 哪个最新就回退到哪个
git checkout -- readme.txt
# 将暂存区的修改删除
git reset HEAD readme.txt
```

### github使用

### git远程仓库

### gitlab使用

### 高阶?

一个文件的hash值的计算, 假定`readme.txt`文件内容为`123`, 它被`git add`的时候, `objects`目录下会增加一个以二进制序列命名的文件

```text
d8/00886d9c86731ae5c4a62b0b77c437015e00d2
```

一共40位\(加密算法为SHA-1\), 其中前两位为目录名, 后38位为文件名

使用python可以用如下方式计算出来

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

**例子**

由于被各种命令搞晕, 于是决定干脆打开`.git`目录一探究竟, 难免会有许多错误, 待日后修改

当所有操作仅限于本地时, `.git`目录大概长这样:

```text
.git/
|-- hooks/        # 暂且不需要管, 用处大致是在执行git命令时可以自动执行一些附加操作
    |-- applypatch-msg.sample    # (文本文件)一些shell脚本代码
    |-- commit-msg.sample        # (文本文件)一些shell脚本代码
    |-- ...                        # (文本文件)一些shell脚本代码
|-- info/
    |--exclude                    # (文本文件)感觉与.gitignore有关
|-- logs/
    |-- refs/
        |-- heads/
            |-- dev                # (文本文件)记录了dev分支的历史提交信息, 格式见下面的说明
            |-- master            # (文本文件)记录了master分支的历史提交信息, 格式见下面的说明
    |-- HEAD                    # (文本文件)记录了HEAD指针的历史提交信息, 格式见下面的说明
|-- objects/
    |-- e6/                        # 大概是散列技术的索引
        |-- 9de29bb2d1d6434b8b29ae775ad8c2e48c5391    # (字节码文件) 对应于一个文件的提交状态
        |-- c15228661d6bfce44a215fb3fbaaea1397059c    # (字节码文件) 对应于一个文件的提交状态
    |-- ee/
        |-- 2c1ea0cbf0f242452802fbf32b1ae0abe92467    # (字节码文件) 对应于一个文件的提交状态
    |-- ...
    |-- info/                    # 不清楚
    |-- pack/                    # 不清楚
|-- refs/
    |-- heads/
        |-- dev                    # (文本文件)记录了dev分支的当前版本号?
        |-- master                # (文本文件)记录了master分支的当前版本号?
    |-- tags/                    # 未知
|-- COMMIT_EDITMSG                # (文本文件)似乎是"最近"一个提交的提交信息
|-- config                        # (文本文件)配置文件, 格式是标准配置文件格式
|-- description                    # (文本文件)大概是对整个项目的描述信息
|-- HEAD                        # (文本文件)文件内容示例: ref: refs/heads/dev
|-- index                        # (字节码文件)暂存区信息, 大概是当前暂存区的快照
|-- ORIG_HEAD                    # (文本文件)文件内容为某个版本号
```

以下为一个完整的测试\(我们主要关注`logs`, `objects`, `refs`文件夹以及`index`, `HEAD`, `ORIG_HEAD`文件的变化\)

**step 1**

```bash
# 在空目录下
git init
```

注意: 此时`logs`文件夹与`ORIG_HEAD`未被创建, `objects`目录下还没有字节码文件

`HEAD`文件的内容为

```text
ref: refs/heads/master
```

`refs/heads/master`文件还未被创建

**step 2**

在工作区增加一个文件`show_git.py`

```bash
git add show_git.py
```

此时, `objects`目录下新增了一个`./74/ba2162d599d4e44dcf3d7811cbbd84d43e911d`字节码文件\(对应于新增的`show_git.py`文件\), `index`文件夹也做了更新.

`refs/heads/master`文件还未被创建

备注: `objects`目录下计算出的16进制值只与文件内容与文件名有关

**step 3**

```bash
git commit -m "add show_git.py 0.1.0 version"
```

此时`.git`目录的变化为:

* `log`目录被创建, 目录结构如下

  ```text
  |-- logs/
      |-- refs/
          |-- heads/
              |-- master            # (文本文件)记录了master分支的历史提交信息
      |-- HEAD                    # (文本文件)记录了HEAD指针的历史提交信息
  ```

  master与HEAD的文件内容如下

  ```text
  # master
  0000000000000000000000000000000000000000 92628406280a94f4efbdcbf59dcb60a5b44ab124 BuxianChen <541205605@qq.com> 1599265251 +0800    commit (initial): add show_git 0.1.0 version

  # HEAD
  0000000000000000000000000000000000000000 92628406280a94f4efbdcbf59dcb60a5b44ab124 BuxianChen <541205605@qq.com> 1599265251 +0800    commit (initial): add show_git 0.1.0 version
  ```

  若此时使用`git flag pretty=oneline`命令, 输出结果为:

  ```text
  92628406280a94f4efbdcbf59dcb60a5b44ab124 (HEAD -> master) add show_git 0.1.0 ver
  sion
  ```

* `index`目录做更新
* `refs`目录下的`heads/master`文件被创建, 内容如下

  ```text
  92628406280a94f4efbdcbf59dcb60a5b44ab124
  ```

  注意: 这个值大概与时间有关\(怀疑是uuid\)

* `objects`目录新增了两个文件`7b/9a4ced4ae9fc215a7cbb2e1c52b6fce706e051`, `92/628406280a94f4efbdcbf59dcb60a5b44ab124`

  不知道为什么要新增两个文件, 有一个疑惑是`git reflog`命令需要的输出保存在哪, 是否与这个有关?

* `HEAD`文件内容保持不变

  ```text
  ref: refs/heads/master
  ```

## Git操作手册

在家整理git与github部分

```text
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

```text
git pull <远程主机名> <远程分支名>:<本地分支名>
# 例子: git pull origin dev:release  #表示将
git push <远程主机名> <本地分支名>:<远程分支名>
# git push origin release:dev
```

```text
git branch <待创建的分支名>
git checkout <待切换的分支名>
```

```text
git rm <file> # 从工作区与缓冲中删除文件
git rm -f <file> # 修改过<file>后, 使用了git add <file>, 此时希望将文件从工作区与缓冲区删除
git rm --cached <file> # 只删除缓冲区中的<file>
```

```text
touch README.txt
git add README.txt
git commit -m "First Commit"
```

情况1:

```text
git restore [--worktree]/[-W] README.md # 工作区文件内容发生变动, 撤销相对于暂存区的修改
git restore --staged/-S README.md # 工作区内的文件内容不变, 但文件状态处于没有添加到暂存区的状态
git restore -s HEAD~1 README.md # 将工作区的文件内容恢复到最近提交的上一个提交版本
git restore -s dbv231 README.md # 将工作区恢文件内容恢复到特定提交版本
```

```text
git status --short/-s
# 以下为输出结果
#  M README                    
# MM Makefile                
# A  lib/git.rb
# M  lib/simplegit.rb
# ?? LICENSE.txt
```

左边的M表示文件修改了并且放入了暂存区, 右边的M表示文件修改了但是没有放入暂存区. 此处表示`README`修改了, 但是还没有使用`git add README`放入暂存区; `lib/simplegit.rb`被修改后放入了工作区, 之后未被修改过; `Makefile`在工作区被修改后放入了暂存区, 而后工作区又做了修改. 左边的A表示`lib/git.rb`是工作区新增的文件, 并已经放入了暂存区, `??`表示`LICENSE.txt`是工作区新增的文件, 但没有放入暂存区中

```text
git log --pretty=format:"%h %s" --graph
```

