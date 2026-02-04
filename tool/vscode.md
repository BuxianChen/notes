# VSCode

## 快捷操作

**块状选择修改**

将如下快速改为

```
a -> b
b -> c
c -> d
```

```
a <- b
b <- c
c <- d
```

方法为：将鼠标放在第一个 `->` 处的 `-` 前面，按住 `alt+shift` 移动鼠标至第三个 `->` 的 `>` 右侧，之后输入 `<-`，即可替换成功，再按下 `esc` 退出此模式。（备注：类似于 vim 的块选择模式）

## VSCode 离线安装扩展 (TODO, 还没有好方案)

有时会出现VSCode版本与扩展不兼容的问题, 此时需要手动下载vsix包进行下载, 下面的[问答](https://stackoverflow.com/questions/69398500/vscode-download-older-version-of-an-extension)解释了一些关于怎么下载历史版本的jupyter扩展的问题, 顺带也有介绍一些GitHub流水线的知识, 供参考.

古老版本的 VSIX: [https://www.vsixhub.com/](https://www.vsixhub.com/)

亲测更推荐这个(有插件对应的VSCode版本要求): [https://open-vsx.org/](https://open-vsx.org/)

**更新**

问题场景如下: 假设内网机器上已经安装了 VSCode, 且通过顶部菜单栏得知了 VSCode 版本号, 但由于网络受限无法联网安装其他插件, 需要在外网机器上先下载相关的插件再拷贝进内网.

[Portable](https://code.visualstudio.com/docs/editor/portable) 版 VSCode 的安装解决这个问题更合理.

假定内网的 VSCode 版本为 1.97.2, 大致原理及操作步骤如下:

在外网环境, 首先下载 Portable 对应版本的 VSCode

```
https://update.code.visualstudio.com/1.97.2/win32-x64-archive/stable
```


## VSCode 设置项

VSCode设置`File->Preferences->Settings`中的`用户`优先级低于`工作区`再低于文件夹, 实际上三者的可以设置的参数名是相同的. 优先级会决定哪个发挥作用. _**也许逻辑是这样的, 用户的设置会体现在vscode安装目录下, 三者计算的结果会存在当前工作区的.vscose目录下**_. 有关于python的默认参数及含义可参见[官网链接](https://code.visualstudio.com/docs/python/settings-reference)

## C++

**VSCode写C++代码 (Window MSVC)**

```text
// tasks.json 注意链接外部头文件的写法
{
    "version": "2.0.0",
    "tasks": [
      {
        "type": "shell",
        "label": "cl.exe build active file",
        "command": "cl.exe",
        "args": [
          "/I",
          "C:/Users/54120/Documents/code",
          "/Zi",
          "/EHsc",
          "/Fe:",
          "${fileDirname}\\${fileBasenameNoExtension}.exe",
          "${file}"
        ],
        "problemMatcher": ["$msCompile"],
        "group": {
          "kind": "build",
          "isDefault": true
        }
      }
    ]
  }
// launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "cl.exe - Build and debug active file",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${fileDirname}\\${fileBasenameNoExtension}.exe",
            "args": [],
            "stopAtEntry": true,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "preLaunchTask": "C/C++: cl.exe build active file"
        }
    ]
}
// c_cpp_properties.json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**",
                "C:/Users/54120/Documents/code"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "windowsSdkVersion": "10.0.18362.0",
            "compilerPath": "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.28.29333/bin/Hostx64/x64/cl.exe",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "msvc-x64"
        }
    ],
    "version": 4
}
```