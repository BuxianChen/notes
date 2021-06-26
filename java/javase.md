# javase

## java语法

* System.out.println\(var\)当参数为引用类型时, 自动转换为System.out.println\(var.toString\(\)\). 其中System在java.lang中, 而System.out为System类的一个静态成员.
* java中没有函数默认值机制
* java成员函数的实现必须直接写在类体中
* java.lang目录下的"顶级"类无需import可以直接使用
* const与goto也是java关键字, 但目前没有具体含义\(保留关键字\). 注意, java中用final表示常量
* java源代码位置

```text
C:\Program Files\Java\jdk-14.0.1\lib\src\
```

```text
# System包的位置
C:\Program Files\Java\jdk-14.0.1\lib\src\java.base\java\lang\System
```

### java代码规范

违反如下规范不会报语法错误

#### 命名规范

变量名, 方法名采用小写字母开头的驼峰式命名方法

常量名全部大写中间加下划线

类名, 接口名使用大写字母开头的驼峰式命名方法

源代码\(.java\)文件名与类名一致, 且一个文件里只放一个外部类, 可以是public或没有修饰符

包名\(目录名\)全部小写使用倒序的网址名加上项目名, 例如`com.baidu.project1`

#### 注释规范

#### 其他

* IDEA中提示一个文件里只能有一个外部类\(备注: matlab里强制一个文件里只能有一个方法, 否则直接报语法错误, 有异曲同工之处\)

### java程序的执行顺序与内存结构

在JVM的存储空间主要包含三个重要区域: 方法区, 栈, 堆. java程序被运行时, 首先将代码片段存储至**方法区**中, 调用方法时将方法栈帧压入**栈区**, 使用new运算符时在**堆**上为对象分配内存. 静态变量在类加载时初始化, 所以存放在方法区. 每个对象内部除了自己的成员外, 还存储这一个特殊的引用型成员变量this\(存储在堆区\).

### 变量与常量

基本数据类型: long/int/short/byte/boolean/char/double/float

引用数据类型: 其余

### 循环与条件判断

### 方法

java中函数的参数传递只有值传递

### 类\(class\)

#### 静态代码块与实例代码块

语法例子:

```java
class Animal{
    private String name;
    static String classname = "Animal";
    static {
        System.out.println("static code block of Animal");
        System.out.println("classname:" + classname); // 静态代码块与静态变量初始化的执行顺序与代码顺序一致
    }
    {
        System.out.println("instance code block of Animal");
    }
    public static void main(String[] args){
        new Animal();
    }
}
```

静态代码块与实例代码块都用的不多, 静态代码块的执行时机为类加载时机, 实例代码块在每次调用构造方法前执行.

#### 构造方法\(new\)

* 当一个类没有编写构造方法时, java编译器会自动未知添加一个无参数构造方法, 此构造方法会将所有基本类型的变量赋值为0, 为引用类型的变量赋值为null. 当一个类编写了构造方法后, java将不会自动提供隐含的无参构造方法. 一个好的习惯是让每个类都写上默认的无参数构造方法
* 由于java中有垃圾回收机制, 所以没有析构函数这种东西
* java构造方法没有C++那样的冒号赋初值的语法

```java
// 调用构造方法必须使用new关键字
A a = new A();
```

#### this

* 每个对象都会隐含的带着一个实例变量this, 它是一个引用数据类型
* this只能出现在实例方法, 构造方法中. 当它出现在构造方法中时, 必须出现在第1行, 此时的使用方式为this\(...\)的形式.

#### 封装

* 一个好的编程习惯是将所有实例变量都设置为private变量, 并同时提供setAttr与getAttr方法

### 继承\(extends\)

~~继承实际上是将父类的东西除了构造方法/private方法/private变量/静态方法/静态变量以外的所有东西都复制过来一份~~

#### 注意事项

* java中只有单继承
* 没有写继承关系的类默认继承Object类
* 子类可以定义与父类同名的变量, 不会发生覆盖. 但是同名方法在满足返回类型与参数类型完全相同时会形成覆盖
* 子类无法直接访问父类的私有成员
* 父类的构造函数不会被继承

```text
class Animal{
    String name;

}
```

#### this与super对比

* this与super都只能出现在成员函数, 构造函数中
* super在构造方法中只能出现在第一行, 因此this与super在构造方法中有且仅有一个. 若一个构造方法既没有this\(...\)也没有super\(...\), 则自动补上super\(\).
* super的主要用于访问父类被覆盖的方法或与子类同名的变量.
* super不是引用, 必须使用super\(...\)或super.xxx两种形式之一.

#### 方法覆盖与多态

* 向上转型: 子类对象转换为父类对象. 可自动进行. 一般用这种方式实现多态

  ```java
  Animal a = new Dog();
  a.move()  // 调用的是Dog的move方法, 多态机制
  ```

* 向下转型: 父类对象转换为子类对象. 必须使用强制类型转换符

  ```java
  Animal a = new Dog();
  Dog d = (Dog) d;  // 向下转型需使用强制类型转换符
  ```

* 向上转型与向下转型的说法仅适用于引用数据类型, 基本数据类型应该使用术语: 自动类型转换与强制类型转换.
* 向下转型有风险, 一般需要使用类似于如下的代码段

  ```java
  Animal a = new Dog();
  // instanceof对于dog is Animal也会判断为true
  if ((a instanceof Dog))  // 避免java.lang.ClassCastException
  {Dog d = (Dog) d;}  // 向下转型需使用强制类型转换符
  ```

方法覆盖也被称为方法重载写 方法覆盖的注意事项为

* 子类方法的访问权限应高于或等于父类的访问权限, 例如父类为protected, 则子类可以时protected或public
* 子类方法不能抛出更多的异常
* 静态方法没有覆盖的概念\(不会被覆盖\)
* 父类的私有方法无法被覆盖
* 多态的含义是编译与运行时有着不一样的状态
* 覆盖后的返回值类型可以是父类方法的返回值类型的子类

### 访问权限修饰符\(private/protected/public\)

[参考链接](https://www.geeksforgeeks.org/protected-keyword-in-java-with-examples/)\(还未完全阅读完\)

#### 属性或方法的访问权限

| 修饰符 | 类内部 | 同个包的子类 | 同一个包的其他类 | 不同包的子类 | 不同包的其他类 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| public | Y | Y | Y | Y | Y |
| protected | Y | Y | Y | Y | N |
| 无修饰符 | Y | Y | Y | N | N |
| private | Y | N | N | N | N |

Note: 网上大部分关于无修饰符和protected的访问权限都是错误的, 代码可以参考`src/java/access`

#### 外部类的访问权限

* public: public修饰的类在其他包中可见, public修饰的类名必须与文件名一致
* 默认: 默认类在本包内可见, 在其他包不可见
* 无private与protected

#### 内部类的访问权限

内部类本身的访问权限可以是private/protected/默认/public四种, 但是IDEA工具会不建议使用protected修饰, 另外, 从设计上看, 内部类应该仅供外部类访问. 因此修饰符应该设为private.

内部类和外部类的访问是相互透明的, 不受内部/外部类成员的控制访问修饰符影响, 都可以相互直接访问, 见下例.

```java
class Outer {
    private int b = 2;
    private class Inner {  // 此处一般建议为private, 否则没必要定义为内部类
        private int a = 1;
        void change() {
            b = 3;
        }
    }
    private void foo() {
        Inner inner = new Inner();
        inner.a = 2;
    }
    public static void main(String[] args) {
        new Outer().foo();
        new Outer().new Inner().change();
    }
}
```

#### 外部接口/内部接口的访问权限

只能是public的, 所以会省略不写

### final关键字

* 修饰普通数据类型: 相当于C++的const
* 修饰方法: 该方法不能被重写
* 修饰类: 该类不能被继承

  ```java
  public final class A{}
  ```

* 修饰引用数据类型: 可以修改引用内部的值

  ```java
  final a = new A();
  a.data = 1; // OK
  ```

* 修饰成员变量

  ```java
  class Test02{
      final private int x;
      Test02(){} // 此行报错, java不会默认的为final修饰的成员变量赋值为0
      Test02(int x){this.x = x;}
      public static void main(String[] args){
          Test02 t = new Test02();
          System.out.println(t.x);
      }
  }
  ```

  final修饰成员变量时一般会与static连用, 表示常量. 命名规范: 全部大写, 单词之间用下划线连接. 常量一般用public修饰.

  ```java
  class Test02{
      public static final int CLASSNAME = "Test02";
  }
  ```

### package与import机制

#### 语法

```text
javac -cp ".:xx/yy/a.jar" -d . aa/bb/Myclass.java  # 设置classpath, 并以包的形式编译
java -cp ".:xx/yy/a.jar" aa.bb.Myclass  # 运行
```

package用于声明本文件所在的包

```java
/* 编译方式: javac -d . Test01.java
运行方式: java aa.Test01
注意: 现在这个类的类名是aa.Test01, 不能切换至aa目录下运行java Test01, 也不能切换后运行java aa.Test01
*/

/*注意import后仍然可以使用java.util.Scanner的写法*/
package aa;
//或者可以用 import java.util.*;
import java.util.Scanner;

public class Test01{
    public static void main(String[] args){
        java.util.Scanner s = new java.util.Scanner(System.in);
        String str = s.next();
        System.out.println("Your input:" + str);
    }
}
```

注意: `java.lang`这个package会被自动import, 例如`java.lang.String`, `java.lang.System`.

#### package命名规范

公司域名倒置+项目名+模块名，例如:

com.baidu.projectname.module1

### 接口\(interface\)

接口可以看作是特殊的抽象类, 接口中的方法必须都是抽象的\(抽象方法可以带有非抽象方法\), 语法如下

```java
// 接口中只能有常量和抽象方法, 且这些常量和方法都必须是public的, 可以省略这些修饰符
interface MyMath{
    /*public static final*/ double PI = 3.1415;
    /*public abstract*/ int add(int a, int b);
    /*public abstarct*/ int sub(int a, int b);
}
interface A{int multi(int a, int b);}
// 接口可以多继承接口
interface B extends A, MyMath{}
// 非抽象类实现接口时必须将所有接口都实现
class MyMathImple implements MyMath{
    public int add(int a, int b){return a + b;}
    public int sub(int a, int b){return a - b;}
}
// 一个非抽象类实现多个接口
class MyMathImple2 implements A, MyMath{
    public int add(int a, int b){return a + b;}
    public int sub(int a, int b){return a - b;}
    public int multi(int a, int b){return a * b;}
}
// extends在前, implements在后
class C extends MyMathImple implements A{
    public int multi(int a, int b){return 1;}
}
public class Test{
    public static void main(String[] args){
        Mymath mm = new MyMathImple();
        mm.add();  // 可以使用多态
        // 诡异: 语法上可以这么干, 编译不报错, 但是运行时会报错
        A a = (A) mm;
    }
}
```

备注: 接口使用的比较多, 抽象类使用的相对较少

#### default关键字

见Iterator与Iterable源代码

#### 常用的一些接口

**Iterable接口与Iterator接口**

注意, 由于java历史原因, Iterable在java.lang包内, 因此无需import, 而Iterator在java.util包内, 因此必须import.

Iterator的源代码如下

```java
package java.util;
import java.util.function.Consumer;
public interface Iterator<E> {
    boolean hasNext();
    E next();
    default void remove() {
        throw new UnsupportedOperationException("remove");
    }
    default void forEachRemaining(Consumer<? super E> action) {
        Objects.requireNonNull(action);
        while (hasNext())
            action.accept(next());
    }
}
```

Iterable的源代码如下

```java
package java.lang;
import java.util.Iterator;
import java.util.Objects;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.function.Consumer;
public interface Iterable<T> {
    Iterator<T> iterator();
    default void forEach(Consumer<? super T> action) {
        Objects.requireNonNull(action);
        for (T t : this) {
            action.accept(t);
        }
    }
    default Spliterator<T> spliterator() {
        return Spliterators.spliteratorUnknownSize(iterator(), 0);
    }
}
```

### 抽象类\(abstract\)

```java
abstract class Account{  // 抽象类
    public abstract void show();  // 抽象方法, 注意没有它方法体
    public void foo(){System.out.println("foo");}
}
class Credit extends Account{
    public void show(){System.out.println("credit");}
}

public class Main{
    public static void main(String[] args){
        Account a = new Credit();
        a.show(); // OK, 适用多态机制, 调用子类的方法
    }
}
```

抽象类的子类可以是抽象类或者是普通类

抽象类中可以没有抽象方法, 但是抽象方法必须出现在抽象类中.

抽象方法必须被重写.

抽象类中有构造方法

abstract与final不能一起用

注意: java中有些方法没有方法体, 但它们不是抽象方法\(native关键字修饰的方法调用底层C++实现\)

### 内部类\(inner class\)

内部类可以定义在方法中, 也可以定义在类中\(与成员变量同级\), 内部类的访问权限修饰符可以是public/private

内部类分为：

* 静态内部类，类似静态变量
* 实例内部类，类似实例变量
* 局部内部类，类似局部变量，匿名内部类属于局部内部类

使用内部类可读性差, 尽量不用

```java
class Test{
    static class Inner1{}  // 静态内部类 Test.Inner1 = new Test.inner1()
    class Inner2{}  // 实例内部类 Test.Inner2 inner = new Test().new Inner2()
    public void doSome(){
        int i = 100;
        class Inner3{}  // 局部内部类
        Inner3 inner = new Inner3()
    }
    public void doOther{
        // Inner3在这里不能使用
        // 匿名内部类的使用
        int a = new A(){
            int add(int a, int b){return a+b;}
        }.sum(1, 2);
    }
}

interface A{
    int add(int a, int b);
}
```

### 泛型机制

基本语法

```java
class MyType<T>{
    public T a;
}
public class Main{
    public static void main(String[] args){
        MyType<Integer> mt = new MyType<Integer>();
        System.out.println(mt.a);
    }
}
```

高级用法

### 异常

### 杂录

java中似乎不会区分声明与定义, 更仔细地说, 只有"声明", 也可以理解成声明时包含了定义. 这体现在编写类的时候函数的代码需写在类内部

java中只允许单继承, 多继承通过interface实现

只引入java.util.Iterator则可以使用Iterable标识符

has a关系 =&gt; 类组合

is a关系 =&gt; 类继承

like a关系 =&gt; 类实现接口

#### java关键字列表

基本类型相关: byte, short, int, long, float,double, boolean, char, strictfp, void

控制结构相关: if, for, else, while, do, continue, break, switch, case, default, return

访问权限相关: public, protected, private

异常相关: throw, throws, try, catch, finally

常量: final

包相关: package, import

调用构造函数: new

类相关: class, interface, static, enum, abstract, extends, implements, this, super, instanceof

小东西: assert

声明该函数底层用计算机相关语言实现: native

transient: 声明不用序列化的成员

synchronized: 声明一段代码同步执行

volatile: 声明多个变量同步发生变化

保留关键字, 没有实际含义: goto, constant

#### Object源码分析

重写finalize方法可以在垃圾回收时机做些事情, 垃圾回收器会自动调用这个方法. 但垃圾回收器的执行时机是不可控的. 它可能会垃圾太多了垃圾回收器才会启动

```java
// Object源代码, JDK在9.0版本后不能重写这个方法
@Deprecated(since="9")
protected void finalize() throws Throwable{}
```

```java
A a = new A();
a = null;
System.gc();  // 建议启动垃圾回收器, 但它依然可能不启动
```

#### 自动类型推断

JDK 8引入, 如下写法可以正常编译通过.

```text
List<Animal> ls = new List<>();
```

#### java内存分析

参考algorithm 4 page201-204分析.

在64位机器上,

* 每个对象一般有16字节的额外开销\(overhead\)
* 引用数据类型占用空间为8字节
* 每个对象所占字节数为8的整数倍\(字节对齐\)
* 实例内部类需要8字节的额外开销
* String对象需要40字节, 其内部的字符数组另外计算

例子1:

```java
public class Date{
    private final int month;
    private final int day;
    private final int year;
}
```

Date类的对象所占空间为32字节: 其中overhead为16字节, 3个整型数据各占4个字节, 再加4字节补齐至8的整数倍.

例子2

```java
public class Counter{
    private final String name;
    private int count;
}
```

Counter类的对象所占空间为32字节: 16\(overhead\)+8\(String是引用类型\)+4\(int类型\)+4\(overhead\)

例子3:

```java
public class Stack<Item> implements Iterable<Item>
{
    private Node first;
    private int N;
    private class Node{
        Item item;
        Node next;
    }
    public boolean isEmpty() { return first == null; }
    public int size() { return N; }
    public void push(Item item) {
        Node oldfirst = first;
        first = new Node();
        first.item = item;
        first.next = oldfirst;
        N++;}
    public Item pop() {
        Item item = first.item;
        first = first.next;
        N--;
        return item;}
}
```

占用内存为$32+64N$个字节

* 其中Stack类本身带有16字节的overhead, 8字节的Node引用, 4字节的整数, 加上4字节的padding. 因此在堆上每个Stack需要分配32个字节.
* 每个Node对象需要16字节的overhead, 由于是Node是内部类, 还需要8个字节的overhead, 再加上两个引用16个字节, 所以一个Node需要40个字节. 而每个Node所指向Integer类型需要24字节\(16字节的overhead+4字节的整型+4字节的padding\). 所以每次增加一个节点, 总共需要64字节的开销.

例子4:

```java
int[] a = new int[N];
```

a占用$24+4N+padding$的内存, 因为数组也是对象, 所以有16字节overhead, 加上记录长度的整数为4字节, 加上4字节padding为24字节. 存储每个整数需要4字节, 共$4N+padding$字节.

例子5:

```java
Date[] a = new Date[N];
```

数组本身分析同上, 需要24字节, 每个Date对象占32字节, 每个Date对象还需要一个额外的引用8字节. 因此一共为$24+40N$字节.

例子6:

```java
double[][] a = new double[M][N]
```

占用内存为$24+32M+8MN$, 首先数组需要24字节, 数组的每个元素都是一个数组, 每个需要8字节的引用, 所以需要$8M$字节, 而每个内层数组double \[N\]需要$24+8N$的大小, 所以总共需要$24+8M+M\(24+8N\)=24+32M+8MN$字节

例子7:

String类型, 本身作为对象有16字节开销, 内部数据为一个字符数组及3个整数, 需要8+12=20字节, 最后padding为4个字节. 因此总共为16+8+12+4一共40个字节. java内部实现时很多字符串可以共用一个内部的字符数组, 因此字符数组的开销不能简单记为$24+2N+padding$.

备注:

* 实例内部类才会有8 byte的额外开销, 静态内部类没有8 byte的额外开销

  参考: [stackoverflow](https://stackoverflow.com/questions/12193116/java-size-of-inner-class), [algs4-question](https://www.coursera.org/learn/algorithms-part1/discussions/weeks/2/threads/zXb8w75FEeuAwA5JxuJq6Q)

* JAVA虚拟机存在一个小整数池, 用以节约内存, 因此一个全为1的整数数组的开销可能会比较小, 参考[algs4-question](https://www.coursera.org/learn/algorithms-part1/discussions/weeks/2/threads/zXb8w75FEeuAwA5JxuJq6Q)

#### Iterator实现细节

当一个Iterator被创建后, 原始对象被修改了, 工程实现上应该抛出异常: [java.util.ConcurrentModificationException](http://docs.oracle.com/javase/8/docs/api/java/util/ConcurrentModificationException.html)

Notes:

在C++标准库中, 类似的情况也有, 尚不清楚STL是怎么处理这个问题的.

## Springboot

