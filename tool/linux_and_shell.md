# linux_and_shell

```bash
sudo useradd -m -N -s /bin/bash someone
# -m: 自动建立用户的登入目录，默认为/home/someone
# -N：不创建同名群组
# -s：指定shell，如安装了zsh，可指定为/bin/zsh
sudo useradd -d /d/ -m -N -s /bin/bash someone
passwd someone
# 设定用户密码
```

