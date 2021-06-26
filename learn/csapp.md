# CSAPP

## chapter 2

#### 整型数值表示

将负数$-x$表示为二进制的方式为: 首先用二进制表示出$+x$, 按位取反后加1即可.

反过来, 依据二进制计算其表示的数字的方法为: 首先依据最高位判断符号, 若最高位为$0$, 则直接利用二进制转十进制确定; 若最高位为$1$, 则确定符号为负, 其绝对值由减一后按位取反, 再用二进制转十进制确定.

**练习**

假定`INT`类型为1字节有符号整数, 则数值表示范围为$\[-2^7,2^7-1\]=\[-128,127\]$.

+10 = 00001010 =&gt; 11110101 =&gt; -10 = 11110110 =&gt; 256 + \(-10\) = 246 = 11110110

11110110 =&gt; - ~\(11110101\)=&gt;- 00001010 =&gt; -10

#### 浮点型数值表示

## chapter 3

```c
//mstore.c
long mult2(long, long);
void multstore(long x, long y, long *dest) {
long t = mult2(x, y);
*dest = t;
}
```

```text
$ gcc -Og -S mstore.c
$ gcc -c -Og -o mstore.o mstore.c
```

显示.o文件里的部分机器码:

```text
$ gdb mstore.o
(gdb) x/14xb multstore # 显示机器码0x0 <multstore>:    0x53    0x48    0x89    0xd3    0xe8    0x00    0x00    0x00
0x8 <multstore+8>:    0x00    0x48    0x89    0x03    0x5b    0xc3
```

显示.s文件里的内容:

```text
//mstore.s
    .file    "mstore.c"
    .text
    .globl    multstore
    .type    multstore, @function
multstore:
.LFB0:
    .cfi_startproc
    pushq    %rbx
    .cfi_def_cfa_offset 16
    .cfi_offset 3, -16
    movq    %rdx, %rbx
    call    mult2
    movq    %rax, (%rbx)
    popq    %rbx
    .cfi_def_cfa_offset 8
    ret
    .cfi_endproc
.LFE0:
    .size    multstore, .-multstore
    .ident    "GCC: (GNU) 5.2.0"
    .section    .note.GNU-stack,"",@progbits
```

反汇编mstore.o:

```bash
$ objdump -d mstore.o

mstore.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <multstore>:
   0:    53                       push   %rbx
   1:    48 89 d3                 mov    %rdx,%rbx
   4:    e8 00 00 00 00           callq  9 <multstore+0x9>
   9:    48 89 03                 mov    %rax,(%rbx)
   c:    5b                       pop    %rbx
   d:    c3                       retq
```

## 

