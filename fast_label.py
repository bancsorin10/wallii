
import glob
import os
import pygame


images = glob.glob("images/*")
i = 0
surface = pygame.display.set_mode((1980, 1080))
pygame.display.set_caption("image")
disp_image = pygame.image.load(images[i])
disp_image = pygame.transform.scale(disp_image, (1980, 1080))
running = True

while running:
    surface.fill((255, 255, 255))
    surface.blit(disp_image, (0, 0))
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                os.rename(image[i], f"{image[i]}_bad")
                i += 1
                disp_image = pygame.image.load(images[i])
                disp_image = pygame.transform.scale(disp_image, (1980, 1080))
                print("bad")
            elif event.key == pygame.K_RIGHT:
                os.rename(image[i], f"{image[i]}_god")
                i += 1
                disp_image = pygame.image.load(images[i])
                disp_image = pygame.transform.scale(disp_image, (1980, 1080))
                print("good")
            elif event.key == pygame.K_q:
                running = False
    pygame.display.update()
