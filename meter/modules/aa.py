import turtle

A = '我喜欢你呀！方梦平！'  # 爱心中心显示的内容


def C():
    for i in range(200):
        turtle.right(1)
        turtle.forward(2)


turtle.setup(width=900, height=600)  # 画布大小
turtle.color('pink', 'red')  # 爱心图案中心颜色及边缘颜色
turtle.pensize(8)  # 画笔粗细
turtle.speed(10)  # 绘制速度
turtle.up()
turtle.hideturtle()
turtle.goto(0, -180)
turtle.showturtle()
turtle.down()
turtle.speed(10)
turtle.begin_fill()

turtle.left(140)
turtle.forward(224)
C()
turtle.left(120)
C()
turtle.forward(224)
turtle.end_fill()
turtle.pensize(5)
turtle.up()
turtle.hideturtle()
turtle.goto(0, 0)
turtle.showturtle()
turtle.color('blue')  # 爱心中心内容的颜色
turtle.write(A, font=('gungsuh', 30,), align="center")
turtle.up()
turtle.hideturtle()
window = turtle.Screen()  # 锁定画布屏幕范围
window.exitonclick()  # 单击退出
