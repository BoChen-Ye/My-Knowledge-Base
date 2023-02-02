本文参考《Digital Integrated Circuits: A Design Perspective》
下图显示的是一个经典的静态CMOS反相器的电路图。
![[Pasted image 20221218155407.png]]
# 基本原理
通过[[MOSFET Device]]这篇文章我们了解MOSFET的工作原理，我们知道当Vin输入电压为高电平（1）时，下面的NMOS导通上面的PMOS截止，在地和Vout之间形成通路使得输出电压为低电平（0）。
同理，当输入电压为低电平时，下面的NMOS截止上面的PMOS导通，在VDD和输出电压之间形成通路使得输出电压为高电平。
在瞬态响应分析中，门从低到高的响应时间是由电阻$R_P$充电电容$C_L$所需要的时间决定的。因此，**一个快速门的设计是通过减小输出电容或者减少晶体管的导通电阻实现的。后者可以通过加大期间的$W/L$来实现。
# 静态特性
1. 输出的高电平为VDD，低电平为GND
2. 逻辑电平与器件的尺寸无关，所以晶体管可以采用最最小尺寸。  
	这种被称为无比逻辑。关于无比逻辑和有比逻辑请看[[Ratio Logic and Ratioless Logic]]
3. 在稳态时，VDD或GND与输出之间总是存在一条有限电阻的通路
	反相器具有低输出阻抗，使得对噪声和干扰**不敏感**
4. 在稳定工作状态下，电源和地线之间没有直接的通路。
	没有电流存在，没有静态功率(DC power)消耗。
5. 输入电路非常高。
	DC电流几乎为零
	单个反相器可以驱动很多的门（具有无穷扇出[[Fanin and Fanout]]）
	但是增加扇出会增加传播延迟（propagation delay），使瞬态响应变差。
## 噪声容限
关于噪声容限参考![[Noise Margin#逻辑电平]]
![[Noise Margin#噪声容限]]
## 电压传输特性（Voltage Transfer Characteristic）
![[Noise Margin#电压传输特性 VTC]]
## 稳定性
### 器件参数变化
设计静态CMOS电路时，若希望使噪声容限最大并得到对称的特性，建议使PMOS比NMOS宽来均衡晶体管的驱动强度。但是，在实际设计中使PMOS管的宽度小于完全对称时所要求的值是可以接受的，因为$V_M$对于器件比值的变化相对来说是不敏感的。
### 降低电源电压
反相器在过渡区的增益实际上随电源电压的降低而增大
# 动态特性
## 电容
- 米勒效应：一个在其两端经历大小相同但是相位相反的电压百富的电容可以用一个两倍与该电容值的接地电容来替代。
	![[Pasted image 20221221215126.png]]
	- 在实际中，由于米勒效应可能导致输出时放电过多。
	![[Pasted image 20221221215101.png]]
- 栅漏电容计算公式：$C_{gd}=2C_{GD0}W$。
## 传播延迟分析
- 由高到低的翻转延迟：$t_{pHL}=0.69R_{eqn}C_L$
	$R_{eqn}$是NMOS在所关注时间内的等效导通电阻。
- 由低到高的翻转延迟：$t_{pLH}=0.69R_{eqp}C_L$
	$R_{eqp}$是PMOS在所关注时间内的等效导通电阻。
- 传播延时定义为这两个值的平均值：$t_p=\frac{t_{pHL}+t_{pLH}}{2}=0.69C_L(\frac{R_{eqn}+R_{eqp}}{2})$
	常常希望一个门的上升和下降输入具有相同的传播延时，这一状况可以通过使PMOS和NMOS晶体管的导通电阻近似来实现。
![[设计技巧#1.减小一个门的传播延时]]

# 功耗
![[功耗]]