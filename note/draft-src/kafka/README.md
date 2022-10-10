使用docker-compose启动kafka服务, 并使用kafka-python包使用

主要参考资料：
- 镜像来源: https://hub.docker.com/r/bitnami/kafka
- 参考博客: https://towardsdatascience.com/kafka-docker-python-408baf0e1088

操作步骤如下:

## 依赖

- docker, docker-compose: 用于启动kafka服务
- `pip install kafka-python`

## 运行方法

首先启动kafka服务
```
docker compose up
```

往kafka里写数据
```
python my_producer.py
```

从kafka里取数据
```
python my_consumer.py
```