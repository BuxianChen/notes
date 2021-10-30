参考资料：

- [objective-c-tutorial](https://www.tutorialspoint.com/objective_c/index.htm)

- [菜鸟教程（讲解不够清晰）](runoob.com/w3cnote/objective-c-tutorial.html)

### 术语表

|      |          |
| ---- | -------- |
| ivar | 实例变量 |
|      |          |
|      |          |



### Objective-C、C、C++ 之间的关系

Objective-C（后面简称：OC）完全兼容 C 语言，也就是说：纯 C 代码是符合 OC 语法的；在编写 OC 代码时可以直接使用 C 语法，引入 C 的包等等，即 OC 代码与 C 代码可以混用。

OC 代码与 C++ 代码只能在 .mm 文件中混合使用。关于 OC 中的文件类型参见后文。

### 编程环境配置

在 Windows 上，需要安装 GNUsetup，安装完成后，编译命令示例为（详情参考：[.gitbook/asset/oc/test01.m](../.gitbook/assets/oc/test01.m)）：

若使用普通的命令行，编译指令为：

```shell
gcc -o test01.exe test01.m -I c:/GNUstep/GNUstep/System/Library/Headers -L c:/GNUstep/GNUstep/System/Library/Libraries -std=c99 -lobjc -lgnustep-base -fconstant-string-class=NSConstantString
```

若使用 GNUsetup 命令行（在 Windows 所有程序中选择 GNUsetup 下的 Shell），编译命令为：

```
gcc `gnustep-config --objc-flags` 
-L /GNUstep/System/Library/Libraries hello.m -o hello -lgnustep-base -lobjc
```

注意用 GNUsetup 命令行打开时，根目录所对应的磁盘目录为：`C:\GNUstep\msys\1.0`

### 语言特性注意事项

- OC 运行入口为 main 函数

- OC 中“函数特征标”与 C++ 具有显著的不同，OC中不以参数类型来区分函数特征标，例如：

  ```objective-c
  // 某个类的@interface内部的函数声明
  - (void)setRGB:(int)red Green:(int)green Blue:(int)blue; 
  ```

  其特征标为：

  ```
  setRGB:Green:Blue
  ```

  因此，在 OC 中不能支持 C++ 意义上的函数重载。例如如下写法是错误的：

  ```objective-c
  - (void)setX:(int) x;
  - (void)setX:(double) x;
  ```

  但是可以用这种写法进行“函数重载”（备注：实质上不是，这两个函数的特征标实际上是不一样的）：

  ```
  - (void)setXY:(int)x oldY:(int)y;
  - (void)setXY:(int)x newY:(int)y;
  ```

- OC 只支持单继承（与 Java 相同）

  ```
  OC(interface) ~ C++(class) ~ Java(class)
  OC(protocol) ~ Java(interface)  // 用来“多继承”
  ```

- OC 中的方法只能在 `@interface` 与 `@end` 之间声明，在 `@implement` 与 `@end` 中定义。也就是说方法必须依附于一个类（类似于 Java，所不同的是，OC 兼容 C 语言，因此可以使用 C 的函数定义与实现的写法，并且在 OC 中，main 函数的写法与 C 语言一致）。

### 例子0：hello world

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
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    SampleClass *sampleClass = [[SampleClass alloc]init];
    [sampleClass sampleMethod];
    [pool drain];
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

  **待续**

### 文件类型

C/C++ 中，头文件一般需要使用如下方式防止头文件被多次引入，从而产生重定义的错误。

```c++
#ifndef HEADER
#define HEADER
// code
#endif
```

在 OC 中，推荐使用如下语法引入头文件，并且相应的头文件内可以不写上述繁琐的 `ifndef`，`define`，`endif` 编译预处理指令。

```objective-c
#import <xxx/yyy.h>  // 例：#import <Foundation/Foundation.h>
```

关于 OC 的程序结构可以参考如下链接（不够深入）：[.h与.m文件里应该放什么](https://victorleungtw.medium.com/connection-between-h-and-m-files-in-objective-c-eaf6b7366717)

### 字符串

OC 中可以沿用 C 语言中单引号表示字符，双引号表示字符串的表示方法，但更推荐这种方式表示字符串：`@"xxx"`。准确地说，这种字符串的类型为 `NSString`。

```objective-c
NSString *str1 = @"12se";
NSString *str2 = [[NSString alloc] init]; // 显式初始化空字符串
NSString *str3 = [[NSString alloc] initWithString:@"jack"]; //显式初始化并赋初值
NSString *str4 = [[NSString alloc] initWithFormat:@"%@ is %@ years old", name, age]; //显式初始化并赋初值
```

备注：NSString 是不可变的，可变字符串类为 `NSMutableString`。

### NSLog 格式控制

```objective-c
int a = 1, b = 2, c = 3;
NSLog(@"%d + %d = %d", a, b, c); // 与C语言中printf类似的格式控制
NSString *name = @"Bob";
NSString *time = @"night";
NSLog(@"Hello, %@, Good %@", name, time);  // %@表示NSString类型
```



### 内存管理

内存管理的相关函数有 alloc、init、retain、release、autorelease、retainCount、dealloc、@selector(retain)、@selector(release) 等。

备注：在使用 ARC （Automatic Reference Counting，自动引用计数）功能后，除了 alloc 与 init 以外，其余均不可使用。实际上，在开启 ARC 功能后，在程序的编译预处理阶段，编译器会自动加上例如 release、retain 等代码。是否启用 ARC 功能可以在 XCode 新建项目时进行选择或者在 Build Settings 中设置，而其本质是在编译选项中添加 `-fobj-arc`。

### 关于 @property 与 @synthesize

参考链接：

- [简书：关于 @property 的历史](https://www.jianshu.com/p/035977d1ba89)
- [简书：成员变量、实例变量、属性的含义](https://www.jianshu.com/p/f7b434534389)
- [stackoverflow：有了@property与实例变量应该用哪个](https://stackoverflow.com/questions/11478038/reason-to-use-ivars-vs-properties-in-objective-c)

简单地说，`@property` 与 `@synthesize` 是 “语法糖”，编译器会对这些写法进行转换。在 OC 早期版本中，@property 与 @synthesize 的作用如下：

@property 在头文件（.h 文件）中使用时，例如：

```objective-c
// test.h
@interface Test: NSObject {
    int count;  // 实例变量
}
@property int count;
@end
// 等效于如下写法
// @interface Test: NSObject {
//     int count;  
// }
// - (int) count;
// - (void) setCount: (int) newCount;
// @end
```

当然，@property 也可以在实现文件（.m 文件）中使用，同样也应该被 `@interface` 与 `@end` 包裹住。两者的区别在于：在头文件中使用时，两个自动生成的方法为 public；在实现文章中使用时，两个自动生成的方法为 private。

@synthesize 在实现文件中使用，例如：

```objective-c
@implementation Test
@synthesize count;
@end
// 等效于
// @implementation Test
// -(double) count { return count; }
// -(void) setCount: (double) _value {count = _value; }
// @end
```

备注：@synthesize 有如下的完整写法，表示将自动生成的两个方法“绑定”到特定的实例变量上。例如：

```objective-c
@synthesize xx = count;
// 等效于
// -(double) xx { return count; }
// -(void) setXx: (double) _value {count = _value; }
```

备注：正常情况下，新建对象后得到的东西是一个“指针”，因此使用 `->` 运算符访问实例变量。（注意：实际上这里的 count 是一个 protected 实例变量，因此可能会出现警告或者错误）

```objective-c
Test *test = [[Test alloc] init];
NSLog(@"%d", test -> count);
```

备注：OC 中，`.` 运算符只能适用于方法，不能用于实例变量，@property 的作用的微妙之处就在于此。并且当 `.` 运算符出现在等号右侧时，会自动调用 getter 方法，出现在左侧时，会自动调用 setter 方法。

**最佳实践**

在现在的 XCode 版本中，XCode 将默认的编译器由 GCC 改为 LLVM，在 .h 文件中只需写：

```
@interface Test: NSObject
@property int count;
@end
```

无需为属性声明实例变量（编译器在预编译时会默认生成一个以下划线开头的实例变量 `_count`）。并且在 .m 文件中也不需要使用 @synthesize 了。

更深入地，@property 可以设置属性的一些特性，详情可以参考**待补充**：[简书 ：Objective-C 属性(property)的特性(attribute) ](https://www.jianshu.com/p/035977d1ba89)

### XCode/Mac 相关

Mac 键盘：[参考](https://www.jianshu.com/p/d3815f2bd3d1)

Mac 中删除与退格键是同一个键，删除文件的快捷键

快捷键方面，Mac 中的 `command` 键相当于 Windows 中的 `Ctrl` 键的作用，例如：

| 功能 | Mac         | Windows  |
| ---- | ----------- | -------- |
| 复制 | `command+c` | `ctrl+c` |
| 粘贴 | `command+v` | `ctrl+v` |

Mac 中使用三指上滑进入任务视图，可以方便地切换窗口，Windows 中可以用 `Win+Tab` 键实现。对于打开的多个 XCode 窗口，也可以使用 `Command+~` 进行切换。

XCode 中可以使用如下方式将代码规范化：`command+A` 选中后使用 `Ctrl+I` 将代码规范化，但对不会删除多余的空行。

### public/protected/private

在 .h 文件中声明的实例变量默认为 protected



### IOS 开发相关：NS/UI/CG/CF

**NSObject**

**UIViewController**

参考链接：

- [简书：UIXXXCotroller](https://www.jianshu.com/p/99f37dac2e8c)
- [官方文档：UIViewController](https://developer.apple.com/documentation/uikit/uiviewcontroller?language=objc)

### 待琢磨

[objective-c private vs protected vs public](https://stackoverflow.com/questions/4869935/objective-c-private-vs-protected-vs-public/4870304)

[@property与实例变量](https://stackoverflow.com/questions/11478038/reason-to-use-ivars-vs-properties-in-objective-c)

Category——mixin编程

### 杂录

OC 中 `id` 类型表示万能指针，相当于 `NSObject *`，类似于 C++ 中的 `void *`，例如：

```objective-c
@interface Test {
	id addr;
    // ...
}
```

