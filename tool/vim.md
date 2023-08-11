# Vim

## 第一部分: VIM的基本使用(第 3 课: VIM)

备注：<font color=red>按键严格区分大小写</font>

### vim 的编辑模式

主要操作模式为：

- 正常模式：在文件中四处移动光标进行修改的模式。进入vim时处于的模式
- 插入模式：插入文本
- 替换模式：替换文本
- 可视化模式：进一步分为一般、行、块模式，主要是移动光标选中一大块文本
- 命令模式：用于执行命令

以正常模式为“中心模式”，使用 `<ESC>` 键从任何其他模式返回正常模式。在正常模式下：使用 `i` 键进入插入模式，使用 `R` 键进入替换模式，使用 `:` 键进入命令模式，使用 `v` 键进入可视化（一般）模式，使用 `V` 键进入可视化（行）模式，使用 `Ctrl+v` 进入可视化（块）模式。

### vim 界面：缓存（cache），标签页（tab），窗口（window）

> Vim 会维护一系列打开的文件，称为“缓存”。一个 Vim 会话包含一系列标签页，每个标签页包含一系列窗口（分隔面板）。每个窗口显示一个缓存。跟网页浏览器等其他你熟悉的程序不一样的是， 缓存和窗口不是一一对应的关系；窗口只是视角。一个缓存可以在多个窗口打开，甚至在同一 个标签页内的多个窗口打开。这个功能其实很好用，比如在查看同一个文件的不同部分的时候。
>
> Vim 默认打开一个标签页，这个标签也包含一个窗口


**window**

`:sp [filename]` 表示在下方新建一个窗口，打开 `filename` 文件

`:vsp [filename]` 表示在右边新建一个窗口，打开 `filename` 文件

`ctrl+w` + `h/j/k/l` 切换至左/下/上/右的窗口

`ctrl+w` + `+/-` 当前窗口增加/减少一行

`ctrl+w` + `10` + `+` 当前窗口增加十行

`:resize +5`：上下方向增加5行

`:vertical resize -5`：左右方向减少5行

`:wq [filename]` 表示关闭当前窗口，并将当前窗口的文件更名为 `filename`

**tab**

`:tabnew [filename]` 表示新建一个标签页，打开 `filename` 文件

切换至左边的标签页：`:-tabnext` 或 `gT` 或 `:tabp`（previous）

切换至右边的标签页：`:+tabnext` 或 `gt` 或 `:tabn`（next）


**vim swap 文件**

若出现多个窗口编辑同一个文件，使用 E 表示对原始文件进行编辑，此时会新建一个缓存。以编辑 `a.py` 为例，第一个缓存的名字为 `.a.py.swp`，第二个缓存的名字为 `.a.py.swo`（第三个缓存的名字为 `.a.py.swn`）。备注：若使用 `E` 进入，则表示基于 `a.py` 里的内容新建缓存 `.a.py.swo`。若使用 `R` 进入，则表示基于 `.a.py.swp` 文件新建缓存 `.a.py.swo`。

- `E` 表示利用原始文件新建缓存
- `R` 表示利用缓存文件新建缓存，若已有多个缓存，则需要按照屏幕提示选择以哪个已有缓存新建缓存

### 各模式下的基础操作

#### 正常模式

正常模式下，光标的显示方式为块状。

| 命令   |    作用            |      |
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

- 行数：`{数字}G`或者`{数字gg}`或者`:{数字}` 表示移动到某一行，例如：`10G` 表示移动到文件的第10行

- Find: `f{character}`, `t{character}`, `F{character}`, `T{character}`

  - find/to forward/backward {character} on the current line
  - `,` or `;` for navigating matches

  `f/F` 表示在当前行查找下一个字符/上一个字符 character。`t/T`表示查找下一个字符character的前一个字符/前一个字符 character 的后一个字符。取消搜索高亮需要在普通模式下输入`:nohlsearch`或`:noh`。

- Search: `/{regex}`, `n` or `N` for navigating matches。从当前光标位置寻找符合正则表达式 `regex` 的字符串，`n` 表示寻找下一个，`2n` 表示寻找往后第二个，`N` 表示向前查找

- Misc: `%` (corresponding item)：匹配左括号或右括号

- `Ctrl+o` 回到上一次光标位置，`Ctrl+i` 回到下一次光标为置。（这里的上一次与下一次是指缓存）

第二类操作为**操作**，也称为**动词**

- 进入到插入模式存在多种方法
  - `i` 在当前方块的左侧进入插入模式（insert）
  - `I` 在当前方块行首插入（insert）
  - `a` 在当前方块的右侧进入插入模式（append）
  - `A` 在当前方块行尾插入（append）
  - `o` 在当前行的下一行新增一个新行并进入插入模式
  - `O` 在当前行的上一行新增一个新行并进入插入模式
- d{motion}：删除（实际上是剪切）
  - `dw`：删除单词
  - `dd`：删除当前行
  - `d3w`：删除从当前单词开始的3个单词
  - `d3l`：删除从当前位置开始的3个字符（向右）
  - `di"`：删除双引号内部的全部内容（delete in `"`）
  - `da"`：删除双引号及双引号中的内容（delete around `"`）
- y{motion}：复制，与`d{motion}`类似
  - `yy`：复制当前行
- x：删除当前字符，等价于`dl`
- c{motion}：改变，即删除并进入插入模式
- u：撤销上一次改变
- `<Ctrl-r>`：重做
- p：粘贴

特殊操作：

- `.`：重复上次的操作
- `zz`：将当前所在行居于屏幕中央显示
- `<ctrl+a>`：将当前数字加1，`<ctrl+x>`：将当前数字减1。

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

普通模式下按 `R` 进入替换模式（此模式比较少用）

替换模式下，光标的显示方式为块状。键入字符表示将光标停留位置做字符替换，新光标位置向后移动一位。

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

保存、退出相关的命令

- `:w` 表示保存
- `:q` 表示退出（前提是所有更改都已经保存了），`:q!` 表示不保存（不保存从上一次保存之后的更新）退出
- `:wq` 表示保存并退出
- `:e! [filename]` 表示放弃从上一次保存之后的更新，[并打开 `filename` 文件进行编辑]。

写完文件时发现没有写入权限：
- `:w !sudo tee %`：此处`%`表示当前文件，一步解决。接下来输入密码确定后，使用`:q!`退出即可
- `:w ~/x.txt`：即先保存到一个有写入权限的位置，再退出vim进行文件copy

分屏
- `:vsplit`：左右分屏
- `:split`：上下分屏
- `<C-W> {hjkl}`：切换屏幕

其他命令：
- `:pwd`：显示当前打开文件的目录
- `:Vex`：在左侧打开目录树，按方向键选择，按回车键进入，`:q`退出窗口。`:Sex` 在上方打开目录树
- `:r !cat x.txt`：将`x.txt`的文件内容读取进来
- `:%TOhtml`：将当前代码转换为html页面

#### 可视化模式

增加注释的操作：`ctrl+v` 后选中若干行，按下 `I` 进入插入模式，输入 `# `，按下 `ESC`，则选中行将加上注释

取消注释的操作：`ctrl+v` 后选中若干行的 `# ` 块，输入 `d` 或 `x` 即可


### 操作实例

#### 例1：对所有行进行相同的操作

原始文件
```python
def foo():
    return 1
```
目标文件
```
# def foo():
#     return 1
```

- 方法一：按`shift+v`进入可视行模式，选中两行，输入`:normal0i# `
- 方法二：按`ctrl+v`进入可视块模式，选中两行的开头，输入`I# `，再输入`ESC`

### VIM Cheatsheet

本部分仅记录原生的VIM快捷键

参考: https://vim.rtorr.com/

#### 重复上次的命令

`.`用于重复上一条命令

#### 制表符
参考[博客](https://linuxhint.com/tab-multiple-lines-vim/)

普通模式下: `>>`插入一个制表符, `<<`向左侧删除一个制表符
visual模式下(三种块模式均可):
- 按下`>`插入一个制表符,继续按`.`插入第2个制表符
- 按下`<`插入一个制表符,继续按`.`插入第2个制表符

#### 撤销与重做

普通模式下: `u`撤销上一次操作, `ctrl+r`重做操作

#### 分屏

`:vsp filename <CR>`: 左右分屏打开文件
`:sp filename <CR>`: 上下分屏打开文件

#### 上下移动

`ctrl+u`上翻半页, `ctrl+d`下翻半页

#### 宏

```
abc
defg
h
ijk
```

现在希望在每行前面加上一个单引号, 结尾加上一个单引号和一个逗号, 可以使用宏来解决

录制宏: 首先回到第一行行首并进入 normal 模式, 按下 `qai'<ESC>$',<ESC>q`, 其中 `qa` 代表录制一个名为 `a` 的宏, 最后的 `q` 代表宏录制结束
应用宏: 对每行应用刚才录制的操作, 使用 `<shift+v>` 进入行可式模式, 然后按 `G` 选中至结尾, 接下来输入 `:normal @a` 应用宏



## 第二部分: 自定义VIM

### VIMRC、VIMSCRIPT

`~/.vimrc` 文件

#### 语法

```
" comment
set number
set syntax on
set laststatus=2
nnoremap S :w<CR>
nnoremap <C-W> :w<CR>
nnoremap R :source $MYVIMRC<CR>
inoremap <Down> <ESC>:echoe "Use j"<CR>
inoremap <Up> <Nop>
nnoremap <C-Right> 5l
```

`set laststatus=2` 为VIM命令，即可以在普通模式下按下`:`再输入`laststatus=2`，可以起到等同于在`~/.vimrc`文件中写`set laststatus=2`一样的效果。此条命令的含义是将当前文件名显示在底部

`nnoremap`：`n`+`nore`+`map`含义分别为`normal（普通模式）`，`no recuresive（非递归）`，`map（映射）`，`nnoremap S :w<CR>` 指的是在普通模式下，按`S`建将等同于按下`:w`与回车键，也就是保存当前文件。类似的语法还有如下：
- nnoremap、nmap：普通模式非递归/递归映射
- inoremap、imap：插入模式非递归/递归映射

`nnoremap <C-W> :w<CR>`表示将`ctrl+w`键映射为保存文件

`nnoremap R :source $MYVIMRC<CR>`：中source为VIM命令，此条的含义为普通模式下按下`R`等同于重新加载`~/.vimrc`文件（这里假定环境变量`MYVIMRC`已定义为了`~/.vimrc`）

同理，`inoremap <Down> <ESC>:echoe "Use j"<CR>` 表示的是将插入模式下的下方向键映射为退出到普通模式，并输出提示：`Use j`。`inoremap <Up> <Nop>`中的`Nop`代表无操作，即按上方向键无任何实际效果。


#### 推荐设置
```
" 设置tab宽度并将tab键替换为若干个空格
set ts=4
set expandtab
set autoindent

" 在底部显示文件名
set laststatus=2

" 当前行底部显示一根下划线
set cursorline

" 在界面右下方显示命令
set showcmd

" 每行的显示内容不会越出窗口的宽度
set wrap

" 语法高亮
set nocompatible
syntax on

" 显示行号/相对行号
set number
set relativenumber

" 搜索结果高亮
set hlsearch

" 搜索时高亮（一边输入一边高亮）
set incsearch

" 忽略大小写搜索与智能大小写搜索
set ignorecase
set smartcase

" 映射取消搜索高亮，默认情况下leader键为反斜杠键。映射后按下leader键，接着再按下回车键即为取消搜索高亮
" let mapleader=" "
noremap <LEADER><CR> :nohlsearch<CR>
```

### VIM 插件

通常而言可以使用一些第三方插件管理器，然后再指定需要的插件进行安装。但其实VIM已经自带了“插件管理器”[参考博客](https://distro.tube/guest-articles/vim-plugins-without-manager.html)。在不使用第三方插件管理器的情况下，可以按如下方法安装插件：

```
mkdir -p ~/.vim/pack/default/start
# mkdir -p ~/.local/share/nvim/site/pack/default/start  # for neovim
cd ~/.vim/pack/default/start
git clone https://github.com/vim-airline/vim-airline.git  # 在底部显示更多信息
```

#### vim-airline

#### nerdtree




## 第三部分: Neovim



关于配置插件的一些“原理”：

- `init.lua`/`init.vim`为用户配置的入口
- `.lua`/`.vim`的搜索路径与`$VIMRUNTIME`变量相关（此变量是nvim中的变量，并非系统环境变量）。关于启动 neovim 时的具体过程可参考：
  - neovim 官方文档：https://neovim.io/doc/user/starting.html#starting
  - 博客：https://thevaluable.dev/vim-runtime-guide-example/
- 关于自定义配置时 lua 脚本放在哪，可参考[官方文档](https://neovim.io/doc/user/lua-guide.html#lua-guide)
  - 这几个目录是特殊的 `~/.config/nvim/lua`，`~/.config/nvim/plugin`，`~/.config/nvim/after`

本部分以用Neovim配合各种插件替代VSCode为主线介绍Neovim的使用


常用快捷键:

- 文件自动保存
- 代码增加/取消注释:
- 切换标签页:
- 转到定义:
- 文件夹导航窗格:
- 在文件夹中查找关键词:
  - VSCode: `ctrl+shift+f`
- 在当前文件中查找关键词:
  - VSCode: `ctrl+f`
- 按文件名查找文件:
  - VSCode: `ctrl+p`
- 底部终端:
- 端口转发:
- git修改显示:
- git解决冲突时选择use theirs or user ours:
  VSCode: 高亮+鼠标选择
- 代码缩进/取消缩进
- markdown预览
- 从剪切版复制内容粘贴至命令模式