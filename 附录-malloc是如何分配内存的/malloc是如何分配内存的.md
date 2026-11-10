> 提出4个问题：
1. malloc如何分配内存？代码验证brk()从堆区分配内存？
2. malloc(1)会分配多大的内存？
3. free函数是如何知道要释放多少内存的？
4. free释放内存会立即归还给操作系统吗？

## 0.前提：进程虚拟地址空间的分布
在进入正式的内容之前，我们先了解一个重要的概念——进程描述符PCB。

每一个进程都有一个进程描述符（PCB），在进程创建的时候，生成PCB，在进程消亡的时候撤销PCB，PCB记录了操作系统所需的描述一个进程的所有信息，如打开的文件，挂起的信号量，进程状态以及地址空间等。
在Linux系统中，每一个进程都由task_struct数据结构来描述，task_struct就是我们平时所说的PCB，当我们fork()一个进程时，操作系统会为我们生成一个PCB，然后从父进程那里继承一些数据，并同时把新进程插入到进程树中，以待进行管理。

task_struct结构体中的内容：

+ 标识符：描述本进程的唯一标识符，又来区别本进程与其他进程。
+ 状态：进程的状态，退出信号等等
+ 优先级：相对于其它进程的优先级。
+ 程序计数器：存储着程序中下一条将要执行的指令的地址。
+ 内存指针(mm_struct)：包括程序代码及进程相关数据的指针，还有其它进程共享的内存块的指针。
+ 上下文数据：进程执行时处理器的寄存器中的数据。
+ IO状态信息：包括显示IO请求，分配给进程的IO设备和被进程使用的文件列表。
+ 记账信息：可能包括处理器时间总和，使用的时钟总和，时间限制，记帐号等。
+ 其他信息 

Linux操作系统通过task_struct感知进程的存在，task_struct中有一个成员变量内存指针（mm_struct），指向了进程虚拟地址空间。

以32位操作系统为例，我们知道，所谓的“位”指的是比特位，而32位，则代表着它有着32根地址线，每一根地址线都有着0和1两种状态，所以对应的，其地址可以由0x0000 0000表示到0xffff ffff，共计4G，其中用户空间占据了3G，内核空间占据了1G。每一个进程，都有它自己独有的进程虚拟地址空间。

![画板](https://cdn.nlark.com/yuque/0/2024/jpeg/28326176/1709619150823-3613dca5-01da-4b5f-afe2-6714aa4616e0.jpeg)

> 以上便是task_struct结构体中内存指针所指向的进程地址空间。

需要注意的是，对于每一个进程，操作系统都会给它们分配一个task_struct结构体，对应的，也会有一块4G的进程地址空间。然而，从硬件的角度讲，我们的CPU只有4G的内存……操作系统却给了每个进程4G的空间，这意味着操作系统给所有进程画的大饼实际上是n*4G的内存，这是远远大于我们真实可用的空间的。

那么，为什么在操作系统画的大饼远大于实际饼的大小的情况下，我们的电脑却能流畅的运行呢？

这就不得不提到进程地址空间的另一个名字了——进程虚拟地址空间。

也就是说，操作系统给每个进程都画了一张4G的饼，但毕竟一个进程是几乎不可能一下子使用4G的内存，绝大多数进程它们就算拼尽全力，所使用的空间，也是远小于4G的，这就给了操作系统很大的操作空间，它通过自己的方式精细规划，合理地给进程分配真正可以使用的物理内存，从而使计算机可以同时运行多个进程。

操作系统所存储数据地物理内存的地址我们是不知的，我们通过程序所得到的地址往往是操作系统给进程分配的进程虚拟地址空间中的虚拟地址。

```cpp
//   1）在父进程中，fork返回新创建子进程的进程ID；
//   2）在子进程中，fork返回0；
//   3）如果出现错误，fork返回一个负值
int main() {
    int i = 10;
    int fd = fork();
    if (fd > 0) {
        i = 5;
        sleep(1);
        printf("pid:%d, &i:%p, i= %d\n", getpid(), &i, i);
     } else {
        printf("pid:%d, &i:%p, i= %d\n", getpid(), &i, i);
     }
     return 0;
 }
```

![](https://cdn.nlark.com/yuque/0/2024/png/28326176/1709556062275-452f4677-71aa-453a-afa0-8539a8f9395f.png)

<font style="color:rgb(77, 77, 77);">不同的进程中，i的地址是相同的，但是它的值却是不同的。由此可以佐证，我们通过程序获取的并不是变量真实存储的物理内存。</font>

## 1.malloc是如何分配内存的
+ 通过brk()系统调用从堆分配
+ 通过mmap()系统调用在文件映射区分配内存
+ 两种方式的区分方式

### a.通过brk系统调用
![画板](https://cdn.nlark.com/yuque/0/2024/jpeg/28326176/1709557336482-9933f619-1df9-4416-a3a7-3d4a5364c8fd.jpeg)

### b.通过mmap文件映射
![画板](https://cdn.nlark.com/yuque/0/2024/jpeg/28326176/1709557825811-0ffbbed9-ec13-4e45-84ae-548f342410b0.jpeg)

_两种方式分配的空间都是虚拟地址空间，最终虚拟地址空间要和真实的物理地址空间通过段表或者页表进行映射，这里不再深入讨论。_

### c.如何区分两种内存分配方式？
```c
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

int main(){
    void *addr = malloc(1024);
    printf("申请出来1024Byte内存：%p\n", addr);
    while(1){
        sleep(1);
    }
    return 0;
}
```

![](https://cdn.nlark.com/yuque/0/2024/png/28326176/1709603340385-91aa81f3-9f67-49f6-b9bd-ce86103c3a78.png)

![](https://cdn.nlark.com/yuque/0/2024/png/28326176/1709603288689-97c53f23-c098-4637-bbfa-a0becf0dbaa3.png)需要看出返回的addr指针<font style="background-color:#FBDE28;">0x55dbbe3</font>a0260在名为[heap]的区间段[<font style="background-color:#FBDE28;">55dbbe3</font>a0000~<font style="background-color:#FBDE28;">55dbbe3</font>c1000]内。

```c
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
// （32位系统值128k）
int main(){
    void *addr = malloc(1024*128*1024);
    printf("申请出来1024*128*1024Byte内存：%p\n", addr);
    while(1){
        sleep(1);
    }
    return 0;
}
```

![](https://cdn.nlark.com/yuque/0/2024/png/28326176/1709604128279-40b125f3-d4c5-4504-bed9-1eab354bf834.png)这是addr指针<font style="background-color:#FBDE28;">0x7f624869</font>c010已经不在堆中了，而是在文件映射区 <font style="background-color:#FBDE28;">7f624869</font>c000-<font style="background-color:#FBDE28;">7f62</font>5069d000。 

## 2.malloc(1)会分配多大内存
从程序员角度malloc(1)只申请了1字节内存，但是从操作系统角度真正分配的内存需要进一步计算。

```c
int main(){
    void *addr = malloc(1);
    printf("申请出来1Byte内存：%p\n", addr);
    while(1){
        sleep(1);
    }
    return 0;
}
// 申请出来1Byte内存：0x563e91041260
```

![](https://cdn.nlark.com/yuque/0/2024/png/28326176/1709604724199-1bf2af9a-e383-474c-9a52-ce31259f3ea7.png)通过(563e91062000)<sub>16</sub> − (563e91041000)<sub>16</sub> = (21000)<sub>16</sub> = (135168)<sub>10</sub> = (132KB)<sub>10 </sub>可以看出操作系统实际把brk指针移动了128KB的空间。

## 3.free函数怎么知道释放多大内存的
注意到malloc返回的指针，和虚拟空间所分配的堆空间有一个差值，这个差值不同的系统可能会不同。

( 563e91041260)<sub>16</sub> −( 563e91041000)<sub>16</sub> = (260)<sub>16</sub> = (608)<sub>10 </sub>

struct malloc_chunk结构体

```c
// glibc源码：http://ftp.gnu.org/pub/gnu/glibc （2.17）
#define INTERNAL_SIZE_T size_t
#define MALLOC_ALIGNMENT           MAX (2 * sizeof(INTERNAL_SIZE_T),
struct malloc_chunk {

    INTERNAL_SIZE_T      prev_size;  /* Size of previous chunk (if free).  */
    INTERNAL_SIZE_T      size;       /* Size in bytes, including overhead. */

    struct malloc_chunk* fd;         /* double links -- used only if free. */
    struct malloc_chunk* bk;

    /* Only used for large blocks: pointer to next larger size.  */
    struct malloc_chunk* fd_nextsize; /* double links -- used only if free. */
    struct malloc_chunk* bk_nextsize;
};

```

各字段含义：.

+ prev_size: 如果上一个chunk块是空闲的时候，表示上一个chunk块的大小，如果上一个chunk块在使用中，则借给上一个chunk使用
+ size: 表示当前chunk的大小（申请的大小和头部的结构体大小）, 大小必须是 2 * SIZE_SZ 的整数倍（在32位系统中以8字节对齐，在64位系统中一般以16字节对齐）
    - 该字段的低三个比特位对 chunk 的大小没有影响，它们从高到低分别表示
    - NON_MAIN_ARENA，记录当前 chunk 是否不属于主线程，1 表示不属于，0 表示属于
    - IS_MAPPED，记录当前 chunk 是否是由 mmap 分配的。
    - PREV_INUSE，记录前一个 chunk 块是否被分配。
    - 一般来说，堆中第一个被分配的内存块的 size 字段的 P 位都会被设置为 1，以便于防止访问前面的非法内存。当一个 chunk 的 size 的 P 位为 0 时，我们能通过 prev_size 字段来获取上一个 chunk 的大小以及地址。这也方便进行空闲 chunk 之间的合并。
+  fd, bk.
    - chunk 处于分配状态时，从 fd 字段开始是用户的数据。chunk 空闲时，会被添加到对应的空闲管理链表中，这时其字段的含义如下：
    - fd 指向下一个（非物理相邻）空闲的 chunk。（指向同一个bin（程序）中的下一个free chunk）
    - bk 指向上一个（非物理相邻）空闲的 chunk。
    - 通过 fd 和 bk 可以将空闲的 chunk 块加入到空闲的 chunk 块链表进行统一管理
    - 在已分配的 chunk 中，该字段直接指向用户数据区。
+ fd_nextsize， bk_nextsize。也是只有 chunk 空闲的时候才使用，不过其用于较大的 chunk（large chunk
    - fd_nextsize 指向前一个与当前 chunk 大小不同的第一个空闲块，不包含 bin 的头指针。(同一个bin（程序）中的下一个free chunk)
    - bk_nextsize 指向后一个与当前 chunk 大小不同的第一个空闲块，不包含 bin 的头指针。
    - 一般空闲的 large chunk 在 fd 的遍历顺序中，按照由大到小的顺序排列。这样做可以避免在寻找合适 chunk 时挨个遍历。

一个已经分配的 chunk 的样子如下。我们称前两个字段称为 chunk header，后面的部分称为 user data。每次 malloc 申请得到的内存指针，其实指向 user data 的起始处。当一个 chunk 处于使用状态时，它的下一个 chunk 的 prev_size 域无效，所以下一个 chunk 的该部分也可以被当前 chunk 使用。这就是 chunk 中的空间复用。

![画板](https://cdn.nlark.com/yuque/0/2024/jpeg/28326176/1709619475095-a27bd02f-e1c0-4dcf-848b-c85d780f7b28.jpeg)

于是可以通过malloc返回的指针，减去偏移量8个字节就可以到size的地址

```c
int main(){
    void *addr = malloc(1024);
    printf("申请出来1024Byte内存：%p\n", addr);
    printf("该chunk大小：%ld\n", *((size_t*)(((char *)addr)-8)));
    printf("最低位为:%ld\n", ((*((size_t*)(((char*)addr)-8)))&1));
    while(1){
        sleep(1);
    }
    return 0;
}
// 申请出来1024Byte内存：0x55ee613752a0
// 该chunk大小：1041
// 最低位为:1
// (0 0 1)
// 1040 = 1024 + 16
```

## 4.free 释放内存，会归还给操作系统吗
```c
int main(){
    void *addr = malloc(1024);
    printf("申请出来1024Byte内存：%p\n", addr);
    getchar();
    free(addr);
    printf("释放了\n");
    while(1){
        sleep(1);
    }
    return 0;
}
// ./a.out 
// 申请出来1024Byte内存：0x5558a7585260
```



![](https://cdn.nlark.com/yuque/0/2024/png/28326176/1709605730108-4104e3e1-c608-4eff-85f7-ca3d59f032d8.png)图1![](https://cdn.nlark.com/yuque/0/2024/png/28326176/1709605818120-3cb12a0e-b5e9-4b0b-9e2c-66c94a57c081.png)图2

可以看出，free的地址空间没有还给操作系统。实际上是返回给了malloc进行管理。

但是如果改为malloc(1024*128*1024)由mmap分配，free后将立即还给OS。验证过程同理。

# 5.总结
+ malloc分配的是虚拟内存，在分配内存时，会预分配更大的空间作为内存池。
+ malloc(1)实际分配132K字节的内存。
+ brk()方式申请的内存，free释放内存时，没有把内存归还给操作系统而是缓存在malloc内存池中了，待下次使用；而mmap方式申请的内存会归还给操作系统。
+ 频繁使用mmap分配内存，每次都要进行运行态的切换，还会发生缺页中断，导致CPU消耗大。
+ 频繁使用brk分配内存，堆内将产生越来越多不可用的碎片。
+ free函数会对传进的内存地址向左偏移~~~~字节，分析出内存块的大小。



