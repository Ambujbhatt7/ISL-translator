import turtle
import colorsys

# Setup screen
screen = turtle.Screen()
screen.bgcolor("black")

# Create turtle
t = turtle.Turtle()
t.speed(0)
t.width(2)

h = 0

# Drawing loop
for i in range(16):
    for j in range(16):
        c = colorsys.hsv_to_rgb(h, 1, 1)
        t.pencolor(c)
        h += 0.005

        t.right(90)
        t.circle(150 - j * 6, 90)
        t.left(90)
        t.circle(150 - j * 6, 90)
        t.right(180)
        t.circle(40, 24)

# Finish
turtle.done()