# Python 杂录

## 异常处理

正常来说, 一个全部组件都完整的异常处理包含一个 `try`, 多个 `except`, 和一个 `finally`. 正常的执行顺序是:

(1) 如果 try 里没有任何异常. 那么执行顺序是: 先执行 `try`, 再执行 `except`, 再执行 `finally`
(2) 如果 try 里发生异常, 并且有对应的 `except` 捕获了这个异常, 并且这个 `except` 执行完也没有引发新的异常, 那么执行顺序是: 先执行 `try`, 再执行 `except`, 再执行 `finally`

有如下特殊情况:

(1) try 里发生异常, 且没有对应的 except 将其捕获, 执行顺序是: 先执行 try, 再执行 finaly
(2) try 里发生异常, 有对应的 except, 但 except 里又引发了新的异常, 那么执行顺序是: 先执行 try, 然后执行 except, 然后执行 finally, 最后将新异常抛出
(3) try 里发生异常, 有对应的 except, except 里引发了新异常, 而 finally 里也引发了新异常, 那么执行顺序是: 先执行 try, 然后执行 except, 然后执行 finally, 最后将 finally 的异常抛出


```python
def bar():
    try:
        try:
            return 1 / 0
        except ZeroDivisionError as e:
            print("Enter except")
            raise ValueError("1")
        finally:
            print("Enter finaly")
            raise ValueError("123")
    except Exception as e:
        print(e.args[0])

bar()
# 输出:
# Enter except
# Enter finaly
# 123
```

除此以外, `break` 和 `return` 也同样遵循 `finally` 覆盖 `except` 覆盖原始 `try` 的逻辑.

注意: finally 语句块种不能使用 yield 语法

另外, 有一个这种特殊模式, 这种模式是**常见且实用**的

```python
def foo():
    try:
        yield db
    finally:
        db.close()

def main():
    db = next(foo())
    # 使用 db 去做别的事情
    # ...
    # 退出 main 时, 会自动执行 finally 语句块的内容

# 也可以像这样更明确地写, 但比较笨拙, 不优雅, 不常见
def main():
    gen = foo()
    db = next(gen)
    # ...
    gen.close()
```

具体的理解如下: 在 main 函数中, `next(foo())` 会让生成器执行到 `yield db` 为止, 而退出 main 函数时, `foo()` 这个生成器会被销毁, 而销毁生成器会触发生成器的 close 方法, 而 close 方法做的事情是在生成器当前为止引发一个 `GeneratorExit`, 这导致生成器执行 `finally`, 也就执行了 `db.close()`, 随后这个 `GeneratorExit` 被抛出, 能被生成器的 close 方法正确处理


## 使用生成器/上下文管理器处理数据库连接

注意: 两种方式的底层是一样的, 本质上是生成器的 close 起作用, 前者更底层些. 但在使用者来说, 前者需要使用 `next(gen)` 的语法, 后者用 `with` 语法, 后者会更友好些. 但从内部实现来说, 前者理解起来难度反而更小些(后者如果要细细研究还需要知道 `__enter__`, `__exit__`, `contextlib.contextmanager`)

**使用生成器**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./dialogues.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def main():
    db = next(get_db())
    # TODO: labeling_item
    db.add(labeling_item)
    db.commit()
    db.refresh(labeling_item)
```

**使用上下文管理器包装(更常见)**

```python
from contextlib import contextmanager

# 函数体和之前没有任何变化
@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def main():
    with get_db() as db:
        # TODO: labeling_item
        db.add(labeling_item)
        db.commit()
        db.refresh(labeling_item)
```

## sys.path

```bash
# 假设当前目录的绝对路径是 <root>
python app/run.py  # 那么sys.path包含的搜索路径是 <root>/app
python -m app.run  # 那么sys.path包含搜索路径是 <root>
PYTHONPATH=<root>/app python -m app.run  # 那么sys.path包含搜索路径是 <root> (最优先) 和 <root>/app (次优先), 因此可能会出现冲突
```
