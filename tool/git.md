# Git

参考：

- https://git-scm.com/docs
- https://missing.csail.mit.edu/
- [pro git](https://git-scm.com/book/en/v2)
- https://www.ruanyifeng.com/blog/2015/12/git-cheat-sheet.html
- ...

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

如果需要查看两个 commit 中某个文件的差异，可以使用如下

```
git diff <source_commit_id> <target_commit_id> -- <filename>
```

例子及解释如下

- source_commit_id 的行用 `-` 进行标识，target_commit_id 的行用 `+` 进行标识，没有变化的行用空格进行标识。

- 显示方式按差异块的形式呈现：

  ```
  @@ -6,12 +6,8 @@ from ..modules import Module
  ```

  表示的是 source_commit_id 的第 [6, 6+12-1] 行的内容与 target_commit_id 的第 [6, 6+8-1] 行的内容有差异。

```bash
$ # pytorch 源码
$ git diff v1.9.1 v1.6.0 -- torch/nn/parallel/data_parallel.py
diff --git a/torch/nn/parallel/data_parallel.py b/torch/nn/parallel/data_parallel.py
index d85d871a5d..86d2cf801d 100644
--- a/torch/nn/parallel/data_parallel.py
+++ b/torch/nn/parallel/data_parallel.py
@@ -6,12 +6,8 @@ from ..modules import Module
 from .scatter_gather import scatter_kwargs, gather
 from .replicate import replicate
 from .parallel_apply import parallel_apply
-from torch._utils import (
-    _get_all_device_indices,
-    _get_available_device_type,
-    _get_device_index,
-    _get_devices_properties
-)
+from torch.cuda._utils import _get_device_index
+

 def _check_balance(device_ids):
     imbalance_warn = """
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

参考 [Pro Git 7.7 节](http://git-scm.com/book/en/v2/Git-Tools-Reset-Demystified#_git_reset)

`git reset` 操作当前分支的三种操作如下

```bash
# commit_id 可以是git commit id或者分支名，或者用类似HEAD^来代表
git reset --soft <commit_id>
git reset --mixed <commit_id>  # 加不加--mixed都一样
git reset --hard <commit_id>
```

下面分别说明上面三条命令在做什么，首先假定当前分支为 `foo`，那么此时 `HEAD` 也指向了 `foo`。

- `git reset --soft HEAD^` 表示的意思是将 `foo` 的指针指向次新一次的提交，而 `HEAD` 依然指向 `foo`
- `git reset --mixed HEAD^` 表示的意思是在上一步的基础上，使用 `foo` 指向的提交更新暂存区
- `git reset --hard HEAD^` 表示在执行前两步后，用暂存区的内容再更新工作区，这样三个区域便完全一致了

备注：这里有一个用法可以用在代码审查中，不确定是否为最佳实践。

VSCode中**工作区相对于暂存区**的修改在代码行的左侧有 gutter indicators 进行标识，现在假设原始仓库的地址为：`https://github.com/base/project`，而开发者foo将此代码库进行了 fork 操作，仓库地址为：`https://github.com/foo/project`。经过代码修改过，首先更新了自己仓库的 `feature` 分支，之后提出 Pull Request 合并至原始仓库的 `dev` 分支。此时可以用如下方式在本地看出代码修改了哪些部分

```bash
# git clone https://github.com/base/project  # 原始仓库
git checkout dev  # 注意严格地说这里需要切换到与PR相匹配的远程dev分支的commit处
git checkout -b foo-feature dev  # 建立新的分支，并切换至foo-feature分支
# 备注：这种方式代码审查人可以自己手动将PR中不合理的地方进行改正
git pull https://github.com/foo/project feature  # 将PR分支并入本地foo-feature

# 如果只希望利用VSCode的gutter indicators查看修改处，则可以临时使用如下命令
git reset dev  # 将foo-feature直接回退至dev分支，并且暂存区也将更新

# 确认/修改后先回到原始的PR状态
git reset <origin_feature_commit>  # 可以通过git reflog命令查询

# git add and commit ...

# 审查完后可以将foo-feature删除
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

类似地，可以使用如下命令来新建本地分支，并与远程分支建立联系。同样地，后续也可以不加参数地使用 git pull/push/fetch

```
git branch --track dev origin/dev  # 新建本地分支 dev，并建立与远程origin/dev分支间的联系
git branch --track origin/dev  # 将当前分支与远程的origin/dev关联
```

### git stash

git stash 用于暂存一些文件，但不进行提交。参见例 4。

```bash
git stash  # 暂存
git stash pop stash@{0}  # 将暂存的东西取出, 并且不保留该份存储
git stash drop stash@{0}  # 丢弃某份存储
git stash apply stash@{0}  # 将暂存的东西取出, 并且保留该份存储
git stash clear  # 清除所有的暂存
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

当工作目录发生了修改，但此时希望切换到另一个分支，并且要切换到的分支与当前工作目录的版本存在冲突时，使用 `git checkout` 命令将会失败，此时如果使用

```bash
git checkout --force brach_name
```

那么等效于先丢弃工作区的全部修改，再进行分支切换

**git checkout 用于文件版本切换**

```bash
git checkout -- readme.txt # 将工作区回退到暂存区或版本库其中之一, 哪个最新就回退到哪个
```

较新版本的 Git 引入了两个命令将 checkout 的两大功能进行了分离。其中 git switch 用于分支切换，git restore 用于文件的版本切换。

**git switch**

```bash
git switch <branch_name>  # 注意：此处不能用commit_id进行切换，因此切换后必不为 detached 状态
git switch --detach <branch_name/commit_id>  # 切换后为 detached 状态
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
# step 1:
git merge <branch_name>  # 将 branch_name 分支合并至当前分支

# step2: 手动解决冲突(即修改好发生冲突的文件)

# step3: 将解决好了的冲突文件进行添加
git add .

# step 4: 继续合并
git merge --continue  # 填写好提交信息后就完成了合并
# 或者直接使用 git commit 也是ok的
```

### git rebase（待研究）

**原理**

从合并的文件上看与 merge 效果一样，但提交历史有了改变。

假定分支情况为：

```
c1 <- c2 <- c3 <- c4 <- c5  # master分支
         <- c6 <- c7  # dev分支
```

使用 `git rebase` 的流程为：

```bash
git checkout dev
git rebase master
# 手动解决冲突
git add xxx
git rebase --continue
# git checkout master
# git merge dev  # fast-forward
```

效果是 dev 分支的提交历史变为

```
c1 <- c2 <- c3 <- c4 <- c5 <- c6’ <- c8
```

所谓 rebase 的直观含义是将 dev 的“基” 从 c2 修改为了 master分支的 c5。使用变基得到的另一个好处是切换回 master 分支后将 dev 分支合进来就不用解决冲突（不产生任何 git object，仅仅是修改了 git ref）

**rebase 与 merge 的区别**

上述过程如果只用 merge，流程为

```bash
git checkout dev
git merge master
# 手动解决冲突
git add xxx
git merge --continue
# git checkout master
# git merge dev  # fast-forward
```

效果是

```
c1 <- c2 <- c3 <- c4 <- c5  <- c8(dev/master)
         <- c6 <- c7		
```

实际上，merge c5 和 c7 的过程为，对 c5 和 c7 对应的 tree 进行合并，解决冲突后使用 `git add` 命令时，会得到一些新的 blob，使用 `git merge --continue` 时，会用 `.git/INDEX`（暂存区）里对应的 tree 写入 `.git/objects` 目录，并得到一个新的 commit 对象 c8，也写入 `.git/objects` 目录。注意：c8 这个 commit 对象的 parent 有两个，即 c5 和 c7。

而 rebase 的过程为：对 c5 和 c7 进行合并后得到的 commit 对象 c8 的 parent 仅有 c5 一个。另外，还会产生一个新的 commit 对象 c6’，其提交信息与 c6 一致，但其 parent 与 c6 不同，并且所对应的 tree 对象也有所不同。

### git cherry-pick

```
# git checkout dev
git cherry-pick <commit-id>
```

上述命令的作用是，将 `<commit-id>` 相对于它前一次提交的修改，作用到当前分支 `dev` 上，并形成一次新的提交。注意：假设 `<commit-id>` 对应于另一个分支例如 `master`，新的提交依旧可能与 `master` 分支有合并冲突。

### git clone

```bash
git clone git@github.com:username/repository_name.git
git clone git@github.com:username/repository_name.git -b dev
```

上述两条命令内部的详细过程为：两者都会将 `.git` 中的所有内容（远程仓库的所有分支）下载到本地。下载后，第二条命令本地的默认分支 dev 由远程分支 origin/dev 产生。

### git remote



### git submodule

参考：[pro-git 7.11](https://git-scm.com/book/en/v2/Git-Tools-Submodules)

参考：[git-tower](https://ww.git-tower.com/learn/git/ebook/en/command-line/advanced-topics/submodules)（一个不错的教程，感觉比 pro-git 还要清晰）

有时候，项目开发时需要引入另一个项目，并希望同时保留两个项目各自的提交历史，此时需要使用 git submodule 命令

例如：在项目 `a` 中需要引入 `https://github.com/example/b.git` 作为子模块，可以使用：

```bash
# 当前目录 a/
git submodule add https://github.com/example/b.git
```

此时，`a/` 目录下会多出一个 `.gitmodules` 文件，并且多出一个 `a/b` 文件夹，文件目录类似如下：

```
a/
  - .git/
  - .gitmodules
  - ...
  - b/
    - .git/
    - ...
```

此时 `a/.git` 目录不记录 `b` 目录的具体修改（只关心在对 `a` 提交时 `a/b` 的 commit id 是否发生变化）。`git` 命令在 `a/` 目录与 `a/b` 目录下分别只对 `a/.git` 与 `a/b/.git` 进行修改，并且只关心各自部分的文件修改历史。

如果需要克隆一个带有 submodule 的仓库，可以使用如下几种方式进行：

```bash
# 一步步操作
git clone https://github.com/example/a
cd a
git submodule init
git submodule update
# 后两步也可以合为一步
# git submodule update --init

# 一步到位
git clone --recurse-submodules https://github.com/example/a
```

### git fetch/pull/push

```bash
git pull <远程主机名> <远程分支名>:<本地分支名>
# 例子：git pull origin dev:release  #表示将
git push <远程主机名> <本地分支名>:<远程分支名>
# 例子：git push origin release:dev
git push -u <远程主机名> <本地分支名>:<远程分支名>
# 之后可以直接使用 git push，不加其余参数
```

git pull 的具体行为是：首先 git fetch 指定远程分支的更新，之后在指定的本地分支上进行 git merge 的操作，如果当前的 HEAD 刚好位于指定的本地分支上，则移动 HEAD 的指向到 merge 后的位置。

git fetch 命令只会将远程分支的修改更新到 `.git` 目录内部，但不会新建分支，也不会对本地的工作目录进行修改。

参考 [stackoverflow](https://stackoverflow.com/questions/10312521/how-to-fetch-all-git-branches) 问答，可以按如下方式拉取所有远程分支的更新：

```bash
git branch -r | grep -v '\->' | while read remote; do git branch --track "${remote#origin/}" "$remote"; done  # 建立与远程分支同名的本地分支，并一一关联
git fetch --all  # 等价于 git remote update, 作用是拉取全部分支的远程更新
git pull --all  # 更新全部的本地分支
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

### git lfs

```
# 不下载大文件的方式进行下载
GIT_LFS_SKIP_SMUDGE=1 git clone xxx.git
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

关于代理引发的 git clone 失败问题，参考[链接](https://blog.csdn.net/shaopeng568/article/details/114919318)

重置代理
```
git config --global  --unset https.https://github.com.proxy
git config --global  --unset http.https://github.com.proxy
```

根据实际端口情况修改
```
git config --global http.https://github.com.proxy http://127.0.0.1:7890
git config --global https.https://github.com.proxy https://127.0.0.1:7890
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

### 例 4（git stash 与 git merge 综合实例）

适用场景如下，例如：master 分支为线上分支，现在需要开发一个新功能，则基于该分支创建一个 dev 分支，但还没修改完毕并且不想提交时，发现 master 分支上出现了 bug，需紧急修复。此时直接使用 `git branch master` 会报错，此时可以使用 git stash 命令将改动的文件暂存，这样便可以正常切换分支。完整过程如下

```bash
# 在dev分支上做了一些还不想提交的修改
git stash  # 暂存所有修改过的文件，但不产生提交
git checkout master
git checkout -b bug

# 修复bug后, 将bug与master合并
git merge master
git checkout master
git merge bug  # Fast-forward, 不会有冲突
git branch -d bug  # 删除bug分支

git checkout dev
git stash list  # 展示暂存的东西
git stash pop stash@{0}  # 将暂存的东西取出, 并且不保留该份存储
# 对dev分支修改完毕后
git commit
git merge master
# 解决冲突
git add .
git merge --continue
git checkout master
git merge dev  # Fast-forward, 不会有冲突
git branch -d dev
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

