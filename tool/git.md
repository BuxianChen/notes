# Git

参考：

- https://git-scm.com/docs
- https://missing.csail.mit.edu/
- [pro git](https://git-scm.com/book/en/v2)
- https://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html
- ...

## 术语

- stage/cache/index/缓冲区/暂存区都是指的同一个东西

## Git 命令简介

所有与 Git 有关的命令最好用 git bash 打开。当然，将 git.exe 所在目录加入到了 path 环境变量后，使用普通的 shell（例如：cmd, powershell, bashrc 等）基本也都没问题。所有的 Git 命令除了 `git init` 与 `git clone` 外，一般都要在 `.git` 的同级目录下执行。

### git init

```bash
git init
```

其效果是为当前目录建立一个 `.git` 目录，正常情况下不要去修改这个文件夹下的任何内容，可以这样理解：之后的每一条以 git 开头的命令执行后，git.exe 会依据命令内容对 `.git` 目录下的文件或依据 `.git` 目录内的文件对工作区进行修改。注意：

- git init 命令不一定需要在空目录，是否在空目录下执行产生的 .git 目录内容是一样的。
- 若 .git 目录已存在, 使用 git init 命令的结果会重新初始化，一般不会这样用

### git config

```bash
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com
```

### git log

以下命令用于显示所有的提交信息

```bash
git log --all --graph --decorate --oneline
git log --pretty=format:"%h %s" --graph
```

### git reflog

```bash
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
```

### git status

```bash
git status
```

输出分为两部分:

* `Changes not staged for commit`: 显示工作区相对暂存区的变动记录
* `Changes to be committed`: 显示暂存区相对最近一次提交的变动记录

用下面的命令可以简化输出：

```bash
git status --short/-s
# 以下为输出结果
#  M README                    
# MM Makefile                
# A  lib/git.rb
# M  lib/simplegit.rb
# ?? LICENSE.txt
```

左边的 M 表示工作区文件修改了（相对最近一次提交）并且放入了暂存区, 右边的 M 表示工作区文件修改了但是没有放入暂存区。此处表示 `README` 修改了，但是还没有使用 `git add README` 命令将其放入暂存区；`lib/simplegit.rb` 被修改后放入了工作区，之后未被修改过；`Makefile` 在工作区被修改后放入了暂存区，而后工作区又做了修改。

左边的 A 表示 `lib/git.rb` 是工作区相对最近一次提交新增的文件，并已经放入了暂存区。

`??` 表示 `LICENSE.txt` 是工作区相对最近一次提交新增的文件，但没有放入暂存区中。

### git diff

```bash
git diff HEAD -- readme.txt
git diff # 显示工作区相对于暂存区的修改
git diff --staged # 显示暂存区相对最近一次提交的修改
```

### git add/rm

`git add` 的作用是为工作区产生变化的文件生成 blob，并将这些文件添加至暂存区

```bash
git add <filename>
```

- 如果 `filename` 在工作区存在，且与暂存区中的内容不一致或暂存区中没有该文件。具体执行过程为：首先为 `filename` 创建一个 object （blob）放在 `.git/objects` 下，之后将该 object 放入暂存区。
- 如果 `filename` 在工作区中不存在，且在暂存区中存在，那么效果等同于 `git rm <filename>`。具体执行过程为：将暂存区中相应的 object 删除

`git rm` 的主要作用是在暂存区中删除文件，可以通过不同的参数选择是否也删除工作区中的文件

```bash
git rm <file> # 从工作区与缓冲区中同时删除文件
git rm -f <file> # 修改过<file>后, 使用了git add <file>, 此时希望将文件从工作区与缓冲区删除
git rm --cached <file> # 只删除缓冲区中的<file>
```

### git commit

```bash
git commit
git commit -m "xxx"
```

git commit 命令表示将当前的暂存区放入版本库中，前者会打开 Git 默认的文本编辑器（可以使用 git config 进行设置）供开发者添加描述信息，在公司里开发项目推荐用这种方式。后者一般用于添加简略的描述信息，适用于不那么正式的个人项目中使用。

### git reset

回退版本

```bash
git reset HEAD readme.txt  # 将暂存区对于HEAD的修改
git reset readme.txt  # 与上一条命令含义相同
```

### git branch

显示已有分支

```bash
git branch  # 显示本地分支
git branch -r  # 显示远程分支
git branch -a  # 显示远程与本地分支
```

创建删除分支

```bash
git branch <待创建的分支名>  # 创建分支
git branch -d <待删除的分支名>
git branch -D <待删除的分支名>  # 强制删除，即使被删除的分支还未被合并
```

为分支设定 upstream 分支，在 Git 中，每个分支至多只能有一个 upstream 分支。注意：这里的 upstream 与数据模型中的 parent 是不同的概念。设定后可以不加参数地使用 git pull/push/fetch。

```bash
git branch --set-upstream-to <远程仓库名>/<远程仓库分支名> <本地分支名>
```

### git checkout/switch/restore

git checkout 命令用于切换分支以及文件的版本切换作用

**git checkout 用于切换分支**

本质上是修改 HEAD 指向的 commit_id

```bash
git checkout <branch_name/commit_id>
git checkout --detach <branch_name/commit_id>
```

注意两条命令只有一些微妙的区别：第一条命令如果切换后 HEAD 与现有的某个分支名的指向一致，则 HEAD 的状态为非 detached 状态，否则为 detached 状态；第二条命令切换后必然处于 detached 状态。所谓 detached 状态指的是切换后相当于处于匿名分支上，如果在匿名分支上发生了提交，之后又切换到别的分支，如果还想切换回匿名分支，那么只能用 commit_id 来切换回去。

**git checkout 用于文件版本切换**

```bash
git checkout -- readme.txt # 将工作区回退到暂存区或版本库其中之一, 哪个最新就回退到哪个
```

较新版本的 Git 引入了两个命令将 checkout 的两大功能进行了分离。其中 git switch 用于分支切换，git restore 用于文件的版本切换。

**git switch**

```bash
git checkout <branch_name>  # 注意：此处不能用commit_id进行切换，因此切换后必不为 detached 状态
git checkout --detach <branch_name/commit_id>  # 切换后为 detached 状态
```

**git restore**

```bash
git restore [--worktree]/[-W] README.md # 工作区文件内容发生变动, 撤销相对于暂存区的修改
git restore --staged/-S README.md # 工作区内的文件内容不变, 撤销暂存区相对最近一次提交的修改，等价于 git reset README.md
git restore -s HEAD~1 README.md # 将工作区的文件内容恢复到最近提交的上一个提交版本
git restore -s dbv231 README.md # 将工作区恢文件内容恢复到特定提交版本
```

### git merge

分支合并的一般流程为：

```bash
git merge <branch_name>  # 将 branch_name 分支合并至当前分支
# 手动解决冲突
git add .
git merge --continue  # 填写好提交信息后就完成了合并
```

### git rebase

**原理**

从合并的文件上看与 merge 效果一样，但提交历史有了改变。

假定分支情况为：

```
c1 <- c2 <- c3 <- c4 <- c5  # f1分支
         <- c6 <- c7  # dev分支
```

使用 `git rebase` 的流程为：

```bash
git checkout f1
git rebase dev
# 手动解决冲突
git add xxx
git rebase --continue
```

效果是 f1 分支的提交历史变为

```
c1 <- c2 <- c6 <- c7 <- c3 <- c4 <- c5
```

个人理解：所谓 rebase 的直观含义是将 f1 的“基” 从 c2 修改为了 dev 分支的 c7。使用变基得到的另一个好处是切换回 dev 分支后将 f1 分支进来就不用解决冲突了。

### git clone

```bash
git clone git@github.com:username/repository_name.git
git clone git@github.com:username/repository_name.git -b dev
```

上述两条命令内部的详细过程为：两者都会将 `.git` 中的所有内容（远程仓库的所有分支）下载到本地。下载后，第二条命令本地的默认分支 dev 由远程分支 origin/dev 产生。

### git remote

### git fetch/pull/push

```bash
git pull <远程主机名> <远程分支名>:<本地分支名>
# 例子：git pull origin dev:release  #表示将
git push <远程主机名> <本地分支名>:<远程分支名>
# 例子：git push origin release:dev
git push -u <远程主机名> <本地分支名>:<远程分支名>
# 之后可以直接使用 git push，不加其余参数
```

### * git cat-file

查看 `.git/objects` 目录下的文件内容

```bash
git cat-file -p 24c5735c3e8ce8fd18d312e9e58149a62236c01a  # 查看 objects 目录下的文件内容
```

### * git ls-files

```bash
git ls-files -s  # 查看当前缓冲区内容, 即 .git/index 文件中的内容
```

## 疑难杂症

```bash
# 忽略权限修改
git config core.filemode false
# 查看git配置
cat .git/
# 忽略某些文件的修改, gitignore只能忽略untracked的文件
git update-index --assume-unchanged [<file> ...]
# 取消忽略
git update-index --no-assume-unchanged [<file> ...]
```

## 详例

注意, 测试2.1与2.2表示的是在测试1的基础上尝试两种做法的结果

### 例 1（待清晰化）

本例着重解释了`git status/diff/restore`三个命令

```text
测试流程，一共有4条路径
1 -> 2.1
1 -> 2.2
1 -> 2.3 -> 2.3.1
1 -> 2.3 -> 2.3.2
```

```bash
git init
# 新建一个README.txt, 添加内容line one
git add README.txt
git commit -m "first commit"
# 在README.txt中添加内容line two
git add README.txt
# 在README中添加内容line three
```

**测试 1**

```bash
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

```bash
git diff # 显示工作区相对于暂存区的修改
git diff --staged # 显示暂存区相对最近一次提交的修改
```

**测试 2**

总结: 接下来, 在操作之前先解释`git restore`的两种用法

```bash
git restore --staged <file> # 将<file>的暂存区记录删除, 即暂存区恢复至最近提交状态
git restore <file> # 将工作区<file>恢复至暂存区的状态
```

**测试 2.1**

```bash
$ git restore README.txt # 此时工作区的README回到只有两行文字的状态
$ git status
On branch master
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        modified:   README.txt
```

**测试 2.2**

```bash
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

**测试 2.3**

总结: `git restore`还有第三种用法如下

```bash
# 将<file>恢复至版本库中的某个版本
git restore --source/-s 7173808e <file>
git restore --source/-s HEAD <file>
```

测试如下:

```bash
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

**测试 2.3.1**

```bash
$ git restore --staged README.txt
$ git status
On branch master
nothing to commit, working tree clean
```

**测试2.3.2**

根据上面的总结, 解释如下过程:

```bash
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

### 例 2（待补充）

本例着重解释 `git rm` 命令

备注: 本机已经准备好了如下环境

```text
git init
# 新建一个README.txt, 添加内容line one
git add README.txt
git commit -m "first commit"
```

### 例 3（待删减）

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

## Git 合作模式

模式一：

master 分支只用作合并，且合并过程自动完成，无需解决冲突。dev 分支用做开发人员的公共基库，各开发人员（例如：f1，f2 分支）完成相应的开发后，在 dev 分支上完成手动解决冲突后的合并。最后将 dev 分支合并至 master 分支。

```bash
git branch dev
git checkout dev
git branch f1  # A: feature 1
git branch f2  # B: feature 2
# do some commit in f1, f2...
git checkout dev
git merge f1 f2
# 手动解决冲突...
git add .
git merge --continue
git checkout master
git merge dev
```

## Git hooks

hooks 通常译为“钩子”，Git hooks 本质上是位于 `.git/hooks` 下的一些脚本，它们会在特定的事件触发时（也就是某些特定的命令被执行时）被自动运行，例如：执行 `git commit` 命令时。其文件名是固定的（对应着相应的事件），git 默认为每个仓库都提供了默认的 hooks，它们的扩展名均为 `.sample`，如果需要启用 hooks，只需要将相应脚本的扩展名删除即可。hooks 的特点是在 `git clone` 时，这些脚本不会被克隆下来，另外默认 hooks 的语言为 shell 脚本，但也可以使用其他脚本语言例如 Python，只需要修改文件的 shebang 行即可。

一个看起来还不错的[教程](https://www.atlassian.com/git/tutorials/git-hooks)。

注意：不要为了加 hooks 而加 hooks，它只是一个工具。

## github、gitlab

### 

### 

