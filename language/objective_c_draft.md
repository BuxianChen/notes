参考资料：

- [菜鸟教程](runoob.com/w3cnote/objective-c-tutorial.html)

**Objective-C、C、C++ 之间的关系**

Objective-C（后面简称：OC）完全兼容 C 语言，也就是说：纯 C 代码是符合 OC 语法的；在编写 OC 代码时可以直接使用 C 语法，引入 C 的包等等，即 OC 代码与 C 代码可以混用。

OC 代码与 C++ 代码只能在 .mm 文件中混合使用。关于 OC 中的文件类型参见后文。

**编程环境配置**

在 Windows 上，需要安装 GNUsetup，安装完成后，编译命令示例为（详情参考：[.gitbook/asset/oc/test01.m](../.gitbook/assets/oc/test01.m)）：

```shell
gcc -o test01.exe test01.m -I c:/GNUstep/GNUstep/System/Library/Headers -L c:/GNUstep/GNUstep/System/Library/Libraries -std=c99 -lobjc -lgnustep-base -fconstant-string-class=NSConstantString
```

```objective-c
// test01.m 源码
#import <Foundation/Foundation.h>

@interface SampleClass:NSObject
- (void) sampleMethod;
- (int) add:(int)a andB:(int)b;
@end

@implementation SampleClass
- (void) sampleMethod {
    NSLog(@"Hello, World! \n");
}

- (int) add:(int)a andB:(int)b {
	return a + b;
}
@end

int main() {
	NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
	SampleClass *sampleClass = [[SampleClass alloc]init];
	[sampleClass sampleMethod];
	int x = 1, y = 2;
	NSLog(@"%d + %d = %d", x, y, [sampleClass add:x andB:y]);
	[pool drain];
	return 0;
}
```



**语言特性注意事项**

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

  待续

**文件类型**

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



**字符串**

OC 中可以沿用 C 语言中单引号表示字符，双引号表示字符串的表示方法，但更推荐这种方式表示字符串：`@"xxx"`。准确地说，这种字符串的类型为 `NSString`。

```objective-c
NSString *str1 = @"12se";
NSString *str2 = [[NSString alloc] init]; // 显式初始化空字符串
NSString *str3 = [[NSString alloc] initWithString:@"jack"]; //显式初始化并赋初值
NSString *str4 = [[NSString alloc] initWithFormat:@"%@ is %@ years old", name, age]; //显式初始化并赋初值
```

备注：NSString 是不可变的，可变字符串类为 `NSMutableString`。

**NSLog 格式控制**

```objective-c
int a = 1, b = 2, c = 3;
NSLog(@"%d + %d = %d", a, b, c); // 与C语言中printf类似的格式控制
NSString *name = @"Bob";
NSString *time = @"night";
NSLog(@"Hello, %@, Good %@", name, time);  // %@表示NSString类型
```



**内存管理**





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

**IOS 开发相关**

**NSObject**

**UIViewController**

参考链接：

- [简书：UIXXXCotroller](https://www.jianshu.com/p/99f37dac2e8c)

- [官方文档：UIViewController](https://developer.apple.com/documentation/uikit/uiviewcontroller?language=objc)

