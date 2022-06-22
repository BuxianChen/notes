# Docker

参考链接：https://yeasy.gitbook.io/docker_practice/

## 基本概念：镜像、容器、仓库

待补充

## 针对镜像的操作命令：

### docker image ls

列出本地镜像

```bash
$ docker image ls nginx
REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
nginx                       latest              e43d811ce2f4        5 weeks ago         181.5 MB
```

### docker pull

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

备注：执行 docker pull 的时候，本地有镜像但由于各种原因（例如跟镜像库网络不通）无法读取镜像库时，docker pull 会报错。猜想原因可能是 docker pull 需要校验镜像库的镜像ID与本地的镜像ID，如果本地存在此ID，则不进行拉取。因此总是需要查远程库，所以会报错，而不会因为本地存在同名镜像而什么都不做。

### docker tag

`docker tag` 命令的作用是为镜像新增一个别名

```bash
$ docker tag 镜像的旧名字/镜像ID 镜像的新名字
```

常见的使用场景是更名后用于推送镜像：

```bash
$ docker tag ubuntu:18.04 username/ubuntu:18.04
$ docker login
$ docker push username/ubuntu:18.04
```

### docker login

```bash
$ docker login  # 登录以获取拉取/推送镜像的权限
```

### docker push

```bash
$ docker push 镜像ID
```

将镜像推送至远端 Docker Registry。

### docker rmi

```bash
$ docker rmi 镜像ID
```

删除镜像

### docker image history

```bash
$ docker image history 镜像ID --no-trunc
```

查看镜像是怎么创建的(Dockerfile)

## 针对容器的操作命令

### docker run

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

### docker container ls

使用 `-d` 参数启动后会返回一个唯一的 id，也可以通过 `docker container ls` 命令来查看容器信息。

```bash
$ docker container ls
CONTAINER ID  IMAGE         COMMAND               CREATED        STATUS       PORTS NAMES
77b2dc01fe0f  ubuntu:18.04  /bin/sh -c 'while tr  2 minutes ago  Up 1 minute        agitated_wright
```

### docker container logs

要获取容器的输出信息，可以通过 `docker container logs` 命令。这条命令尤其适用于使用了 `-d` 参数进行启动的容器。

```bash
$ docker container logs [container ID or NAMES]
hello world
hello world
hello world
...
```

### docker container start/restart/stop

重新启动已经终止的容器/将一个运行态的容器关闭并重新启动它/将一个运行态的容器终止

```bash
$ docker container start [container ID or NAMES]
$ docker container restart [container ID or NAMES]
$ docker container stop [container ID or NAMES]
```

### docker attach/exec

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

### docker stats

以下命令用于查看容器的内存占用等情况

```bash
$ docker stats 容器ID
```

### docker commit

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

## 使用 Dockerfile 制作镜像

### 例子

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

### 利用已有镜像反推 Dockerfile

参考 [CSDN](https://blog.csdn.net/qq_43228739/article/details/115718191)

```bash
$ docker history --format {{.CreatedBy}} --no-trunc=true <镜像ID> |sed "s?/bin/sh\ -c\ \#(nop)\ ??g"|sed "s?/bin/sh\ -c?RUN?g" | tac
```

### 例子：深度学习 Dockerfile

#### 标准镜像规划

| 标签                                                   | Python/Conda | cuda&cudnn | nvcc&gcc | pytorch | 用途                                                  |
| ------------------------------------------------------ | ------------ | ---------- | -------- | ------- | ----------------------------------------------------- |
| official-nvidia-cuda-10.1-cudnn7-devel-ubuntu18.04     | no           | yes        | yes      | no      | 暂无，开发时不必从头安装python                        |
| official-nvidia-cuda-10.1-cudnn7-runtime-ubuntu18.04   | no           | yes        | no       | no      | 最小体积上线，pytorch版本不在下面的列表中时用这个上线 |
| official-pytorch-pytorch-1.6.0-cuda10.1-cudnn7-devel   | yes(conda)   | yes        | yes      | yes     | 用于开发（官方镜像）                                  |
| official-pytorch-pytorch-1.6.0-cuda10.1-cudnn7-runtime | yes(conda)   | yes        | no       | yes     | 用于上线代码（Pytorch版本最好不要动）                 |
| official-pytorch-pytorch-1.9.0-cuda10.1-cudnn7-devel   | yes(conda)   | yes        | yes      | yes     | 用于开发（官方镜像）                                  |
| official-pytorch-pytorch-1.9.0-cuda10.1-cudnn7-runtime | yes(conda)   | yes        | no       | yes     | 用于上线代码（Pytorch版本最好不要动）                 |

备注：上述 devel 均包含有 nvcc 及 gcc，而 runtime 均不包含这些编译工具。所有镜像均不包含 vim，git 等软件。

使用说明：

- 为追求最小体积上线，上线时可以直接用官方镜像，将代码复制进去即可（若想
- 开发环境可以使用一些脚本



#### shell 脚本 & Dockerfile

**shell script（待补充：`./docker/dl/plus.sh`）**

**Dockerfile（待补充：`./docker/dl/Dockerfile`）**

下面是细节：

**更改时区（似乎不是必要的）**

参考 [stackoverflow](https://stackoverflow.com/questions/44331836/apt-get-install-tzdata-noninteractive) 问答

**vim 与 git**

备注：pytorch 官方镜像包含 conda

```
RUN apt update && apt -y upgrade && apt install -y vim git tmux
```

**更改 pip/conda/apt 源**

待补充

**安装 pre-commit**

待补充

**代码支持中文注释**

在 `/etc/vim/vimrc` 中添加

```
set encoding=UTF-8
"set langmenu=zh_CN.UTF-8
"language message zh_CN.UTF-8
set fileencoding=UTF-8
```

```
RUN echo -e "set encoding=UTF-8\nset fileencoing=UTF-8" > /etc/vim/vimrc
```

**环境变量**

官方镜像设置的 cuda 相关的环境变量似乎有些诡异，例如：`$LD_LIBRARY_PATH` 里显示的路径不存在，可能需要修改。

#### 实操

待补充

若要使用 GPU：Windows 下使用 WSL2 需要去 [Nvidia](https://developer.nvidia.com/cuda/wsl) 官网下载驱动即可。之后启动镜像时需要增加参数 `--gpus all` 即可：

```
docker run -it --rm --gpus all pytorch/pytorch:1.9.0-cuda10.2-cudnn-devel /bin/bash
```

## docker compose

旧版本的使用方式为 `docker-compose ...`, 新版本的使用方式为 `docker compose ...`，docker compose的作用是管理一组容器来实现某个功能。例如一个应用需要依赖于一个数据库，正确的做法是将应用程序代码放在一个容器中，而数据库用另一个容器启动，并且通过建立网络使得这两个容器能进行互相访问。docker compose 可以通过写一个配置文件自动进行搭建网络，构建所需要的镜像，并启动相应的镜像。docker compose 的实现逻辑是调用了 Docker 服务提供的 API 来对容器进行管理。

### 一个示例

例子摘录自 [Docker从入门到实践](https://yeasy.gitbook.io/docker_practice/compose/usage) 并在此基础上完善

```
learn-docker
    web/
        app.py
        Dockerfile
    docker-compose.yml
```

各文件内容如下
```python
# web/app.py
from flask import Flask
from redis import Redis

app = Flask(__name__)
redis = Redis(host='redis', port=6379)

@app.route('/')
def hello():
    count = redis.incr('hits')
    return 'Hello World! 该页面已被访问 {} 次。\n'.format(count)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
```

```dockerfile
# web/Dockerfile
FROM python:3.6-alpine
ADD . /code
WORKDIR /code
RUN pip install redis flask
CMD ["python", "app.py"]
```

```yaml
version: '3'
services:

  web:
    build:  # 也可以直接写"build: ./web", 后面的context与dockerfile可省略不写
      context: ./web
      dockerfile: Dockerfile  # 注意此处的dockerfile是相对于context而言的
    image: "web_web:v1"
    ports:
      - "5000:5000"

  redis:
    image: "redis:alpine"  # 注意此处并未将6379端口映射出去,但web容器是可以访问该端口的,但无法在本机直接访问数据库
```

运行方法为
```bash
# -f docker-compose.yml可省略不写
# docker compose up是先打镜像(如果没有), 再建立网络, 最后启动容器
docker compose -f docker-compose.yml up

# docker compose build  # 可强制重新打镜像
# docker compose up
```

docker compose 命令的一般格式为
```
docker compose [-f docker-compose.yml文件路径] 命令
```


## 杂录

### Docker Desktop (Windows 11) 使用记录

安装基本是傻瓜式的，从略。配置镜像加速可以注册一个阿里云账号，搜索“容器镜像加速服务”按上面的指引修改 Docker Desktop 的配置即可。



