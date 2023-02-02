本文基于《轻松称为设计高手——Verilog HDL实用精解》
# 3种描述方法
## 数据流描述：
采用assign语句，该语句被称为连续赋值语句
`assign #1 A_xor_wire=eq0^eq1;
在任意一个时刻,A_xor_wire线网的值是由eq0和eq1决定的,也可以说是由它们驱动的。
![[Pasted image 20221218165842.png]]
## 行为描述：
使用always或initial语句块，出现的语句被称为过程赋值语句。这种也是最常用的描述方法。
initial语句在0仿真时间执行,而且只执行一次;always语句同样在0仿真时间开始执行,但是它将一直循环执行。
```
`timescale1ns/1ns
moduleClockGen(Clock);
outputClock;
regClock;
initial//将Clock初始化为0
Clock=0;
always //每个5ns将Clock翻转一次
#5 Clock=~ Clock;
endmodule
```
### 时序控制
在行为描述中,有几种方式对设计模型进行时序控制,它们是:
- 事件语句(“@”);
- 延时语句(“#”);
- 等待语句。
### 过程赋值
1. 阻塞赋值
	阻塞赋值的语法如下:
	`寄存器变量= 表达式;`
	如果多个阻塞赋值语句顺序出现在begin...end语句中,前面的语句在执行时,将完全阻塞后面的语句,直到前面语句的赋值完成以后,才会执行下一句的右边表达式计算。
1. 非阻塞赋值
	非阻塞赋值的语法如下:
	`寄存器变量<= 表达式;`
	,如果多个非阻塞赋值语句顺序出现在begin…end语句中,前面语句的执行,并不会阻塞后面语句的执行。前面语句的计算完成,还没有赋值时,就会执行下一句的右边表达式计算。例如“beginm<=n;n<=m;end”语句中,最后的结果是将m 与n值互换了。
## 结构化描述：
实例化已有功能模块，主要由以下三种方法。
1. Module实例化：实例化已有的Module
2. 门实例化
3. 用户定义原语（UDP）实例化
# 基本语法
## 词法
Verilog区分大小写
//表示注释，/**/表示注释全部
## 模块和端口
![[Pasted image 20221218163854.png]]
Module是Verilog中基本组成单位。
通常module具有输入和输出端口,在module名称后面的括号中列出所有的输入、输出和双向的端口名称。
有些module也不包含端口。例如,在仿真平台的顶层模块中,其内部已经实例化了所有的设计模块和激励模块,是一个封闭的系统,没有输入和输出。一般这种没
有端口的模块都是用于仿真的,不用作实际电路。
在module内部的声明部分,需要声明端口的方向(input,output和inout)和位宽。按照Verilog的习惯,高位写在左边,低位写在右边。比如“`input[1∶0]A_in;`”就表示两位的总线。
模块内部使用的reg(寄存器类型的一种)、wire(线网类型的一种)、参数、函数以及任务等,都将在module中声明。
一般来说,module的input默认定义为wire类型,output信号可以是wire,也可以是reg类型(如果在always或initial语句块中被赋值)。而inout是双向信号,一般将其设为tri类型,表示其有多个驱动源,如无驱动时为三态。
## 逻辑值
- “X”表示未知值(unknown),或者不关心(don’tcare),“X”用作信号状态时表示未知,用在条件判断时(在casex或casez中)表示不关心;
- “Z”表示高阻状态,也就是没有任何驱动,通常用来对三态总线进行建模
- 0
- 1
## 常量
1. Verilog中的常量有3种:
- 整数型;
- 实数型;
- 字符串型。
2. 在基数表示法中,都是以如下格式写的:
	[长度]’数值符号数字
	例如：`4'b1010`
	其中长度可有可无,数值符号中,h表示十六进制,o表示8进制,b表示二进制,d表示十进制数据。如果长度比后面数字的实际位数多,则自动在数字的左边补足0,如果位数少,则自动截断数字左边超出的位数。
## 变量
### Wire型
表示电路间的物理连线。
未初始化时的值为“Z”。
线网类型主要用在连续赋值语句中,以及作为模块之间的互连信号。
### Reg型
寄存器类型变量在Verilog语言中通常表示一个存储数据的空间。
- reg:是最常用的寄存器类型数据,可以是1位或者多位,或者是二维数组(存
储器);
- integer:整型数据,存储一个至少32位的整数;
例：
`reg[3:0]ABC;//定义一个名为ABC的4位寄存器`
`reg[3:0]MEMABC[0:7]; //定义一个存储器,地址为0~7,每个存储单元是4位`
# 并发与顺序
与在处理器上运行的软件不同的是,硬件电路之间的工作是并行的。
但是,在语句块(always和initial)内部,则可以存在两种语句组:
- begin...end:顺序语句组;
- fork...join:并行语句组。
# 操作符
![[Pasted image 20221218165552.png]]