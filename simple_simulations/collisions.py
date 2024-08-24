import numpy as np
import pygame as pg

windowWidth = 900
windowHeight = 700

class Ball:
    def __init__(self, position, velocity, radius=6) -> None:
        self.pos = np.array(position)
        self.vel = np.array(velocity)
        self.radius = radius

    def updatePosition(self, dt):
        self.pos += dt * self.vel

ballList = []
for i in range(7):
    ballList += [Ball([10. + j * 100., 10. + i * 100.], [50. * i + (-1) ** j * 100., 50. * i + 50.]) for j in range(9)]

def wallCollion(ball):
    if ball.pos[0] - ball.radius <= 0:
        ball.vel[0] = -ball.vel[0]
    if ball.pos[0] + ball.radius >= windowWidth:
        ball.vel[0] = -ball.vel[0]
    if ball.pos[1] - ball.radius <= 0:
        ball.vel[1] = -ball.vel[1]
    if ball.pos[1] + ball.radius >= windowHeight:
        ball.vel[1] = -ball.vel[1]

def ballCollision(ball):
    global ballList
    for otherBall in ballList:
        if ball == otherBall:
            continue
        u = ball.pos - otherBall.pos
        if np.linalg.norm(u) <= ball.radius + otherBall.radius:
            u = u / np.linalg.norm(u)
            v = np.dot(np.array([[0, -1], [1, 0]]), u) 
            rotationMatrix = np.array([[u[0], v[0]],
                                       [u[1], v[1]]])
            invRotationMatrix = np.linalg.inv(rotationMatrix)
            newBallVel = np.dot(rotationMatrix, ball.vel)
            newOtherBallVel = np.dot(rotationMatrix, otherBall.vel)
            newBallVel[0], newOtherBallVel[0] = newOtherBallVel[0], newBallVel[0]
            ball.vel = np.dot(invRotationMatrix, newBallVel)
            otherBall.vel = np.dot(invRotationMatrix, newOtherBallVel)

pg.init()
window = pg.display.set_mode((windowWidth, windowHeight))
clock = pg.time.Clock()
dt = 0
running = True

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    window.fill('black')
    totalEnergy=0
    for ball in ballList:
        pg.draw.circle(window, 'white', ball.pos, ball.radius)
        wallCollion(ball)
        ballCollision(ball)
        ball.updatePosition(dt)
        totalEnergy += 0.5 * np.dot(ball.vel, ball.vel)
    print(totalEnergy)
    pg.display.update()
    dt = clock.tick(90) / 1000

pg.quit()