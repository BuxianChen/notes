**Objective-C、C、C++ 之间的关系**

Objective-C（后面简称：OC）完全兼容 C 语言，也就是说：纯 C 代码是符合 OC 语法的；在编写 OC 代码时可以直接使用 C 语法，引入 C 的包等等，即 OC 代码与 C 代码可以混用。

OC 代码与 C++ 代码只能在 .mm 文件中混合使用。关于 OC 中的文件类型参见后文。

**hello world**

```objective-c
// main.m 文件内容
#import <Foundation/Foundation.h>
#include "stdio.h"

@interface SampleClass:NSObject
- (void) sampleMethod;
@end

@implementation SampleClass
- (void) sampleMethod {
    NSLog(@"Hello, World! \n");
    printf("Print from C");
}
@end

/* 注释的写法与C语言一致 */
int main() {
   SampleClass *sampleClass = [[SampleClass alloc]init];
   [sampleClass sampleMethod];
   return 0;
}
```

- 定义接口（相当于 C++ 的类）的方法是

  ```objective-c
  @interface SampleClass:NSObject
  - (void) sampleMethod;
  @end
  ```

  表示定义了一个名为 `SampleClass` 的类，继承自 `NSObject`，`NSObject` 是 OC 中所有类的基类（类似于 Python 中的 Object，Java 中的 object）。

  类定义中的实例方法（

**文件类型**

与C

```
#import <xxx/yyy>
```







**XCode/Mac 相关**

Mac 键盘：[参考](https://www.jianshu.com/p/d3815f2bd3d1)

Mac 中删除与退格键是同一个键，删除文件的快捷键

快捷键方面，Mac 中的 `command` 键相当于 Windows 中的 `Ctrl` 键的作用，例如：

| 功能 | Mac         | Windows  |
| ---- | ----------- | -------- |
| 复制 | `command+c` | `ctrl+c` |
| 粘贴 | `command+v` | `ctrl+v` |

Mac 中使用三指上滑进入任务视图，可以方便地切换窗口，Windows 中可以用 `Win+Tab` 键实现。对于打开的多个 XCode 窗口，也可以使用 `Command+~` 进行切换。

XCode 中可以使用如下方式将代码规范化：`command+A` 选中后使用 `Ctrl+I` 将代码规范化，但对不会删除多余的空行。
