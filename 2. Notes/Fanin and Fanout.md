本文基于《数字集成电路：电路，系统，设计（第二版）》

# Fanout扇出
扇出表示连接到驱动门输出端的负载门的数目N，见下图。
![[Pasted image 20221219163908.png]]
增加一个门的扇出会影响它的逻辑输出电平。使负载门的输入电阻尽量大，并保持驱动门输出电阻尽量小可以减少影响。
因为输入电阻尽可能大代表输入电流最小，输出电阻较小代表减少负载电流对输出电压的影响。
当扇出较大时，所加的负载动态性能变差。
# Fanin 扇入
一个门的扇入定义为该门输入的数目。扇入较大的门会使动态和静态特性变差。
![[Pasted image 20221219164646.png]]