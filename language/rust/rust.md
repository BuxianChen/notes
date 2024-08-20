# Rust

推荐资料:

- [The Rust Programming Language](https://doc.rust-lang.org/beta/book): [GitHub](https://github.com/rust-lang/book), [中文版](https://rustwiki.org/zh-CN/book/), [中文版 GitHub](https://github.com/rust-lang-cn/book-cn)
- [Rust By Example](https://doc.rust-lang.org/rust-by-example/)

**hello world**

代码

```rust
// main.rs
fn main() {
    println!("Hello, world!");  // println! 里后面的感叹号!表示这是一个宏, 而不是一个函数
}
```

编译与运行

```bash
rustc main.rc
./main  # 编译出的文件是可执行文件
```

<table>
<tr>
    <td> C++ </td>
    <td> Python </td>
    <td> Rust </td>
</tr>
<tr>
    <td> 函数 </td>
    <td> 函数 </td>
    <td> 函数 </td>
</tr>
<tr>
    <td> 类 </td>
    <td> 类 </td>
    <td> 结构体 </td>
</tr>
<tr>
    <td> 成员函数 </td>
    <td> 方法 </td>
    <td> 方法 </td>
</tr>
<tr>
    <td> 静态成员函数 </td>
    <td> 用 classmethod 装饰器装饰的方法 </td>
    <td> 关联方法 </td>
</tr>
</table>

