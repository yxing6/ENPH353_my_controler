import pathlib
import numpy as np
import os
import PIL
from PIL import Image, ImageEnhance
import tensorflow as tf
import csv
import random
from pathlib import Path
from math import sqrt
import matplotlib.pyplot as plt
import cv2

def filter_img(img):
    filtered = cv2.cvtColor(cv2.medianBlur(img, 71), cv2.COLOR_BGR2HSV)
    filtered = cv2.cvtColor(cv2.GaussianBlur(filtered, (45, 45), 0), cv2.COLOR_BGR2HSV)
    return filtered


darkline = filter_img(cv2.imread('/home/fizzer/Pictures/stage7darkline.png'))
outsidewall = cv2.imread('/home/fizzer/Pictures/darkoutertube.png')
hilltrack = filter_img(cv2.imread('/home/fizzer/Pictures/stage7hilltrack.png'))
topline = filter_img(cv2.imread('/home/fizzer/Pictures/stage7topline.png'))
toptrack = filter_img(cv2.imread('/home/fizzer/Pictures/stage7toptrack.png'))
track = filter_img(cv2.imread('/home/fizzer/Pictures/stage7track.png'))
wall = filter_img(cv2.imread('/home/fizzer/Pictures/stage7wall.png'))



hsv_darkline = cv2.cvtColor(darkline, cv2.COLOR_BGR2HSV)
hsv_outsidewall = cv2.cvtColor(outsidewall, cv2.COLOR_BGR2HSV)
hsv_hilltrack = cv2.cvtColor(hilltrack, cv2.COLOR_BGR2HSV)
hsv_topline = cv2.cvtColor(topline, cv2.COLOR_BGR2HSV)
hsv_toptrack = cv2.cvtColor(toptrack, cv2.COLOR_BGR2HSV)
hsv_track = cv2.cvtColor(track, cv2.COLOR_BGR2HSV)
hsv_wall = cv2.cvtColor(wall, cv2.COLOR_BGR2HSV)

hist_darkline_hue = cv2.calcHist([hsv_darkline], [0], None, [256], [0,256])
hist_darkline_sat = cv2.calcHist([hsv_darkline], [1], None, [256], [0,256])
hist_darkline_val = cv2.calcHist([hsv_darkline], [2], None, [256], [0,256])

hist_outsidewall_hue = cv2.calcHist([hsv_outsidewall], [0], None, [256], [0,256])
hist_outsidewall_sat = cv2.calcHist([hsv_outsidewall], [1], None, [256], [0,256])
hist_outsidewall_val = cv2.calcHist([hsv_outsidewall], [2], None, [256], [0,256])

hist_hilltrack_hue = cv2.calcHist([hilltrack], [0], None, [256], [0,256])
hist_hilltrack_sat = cv2.calcHist([hilltrack], [1], None, [256], [0,256])
hist_hilltrack_val = cv2.calcHist([hilltrack], [2], None, [256], [0,256])

hist_topline_hue = cv2.calcHist([topline], [0], None, [256], [0,256])
hist_topline_sat = cv2.calcHist([topline], [1], None, [256], [0,256])
hist_topline_val = cv2.calcHist([topline], [2], None, [256], [0,256])

hist_toptrack_hue = cv2.calcHist([toptrack], [0], None, [256], [0,256])
hist_toptrack_sat = cv2.calcHist([toptrack], [1], None, [256], [0,256])
hist_toptrack_val = cv2.calcHist([toptrack], [2], None, [256], [0,256])

hist_track_hue = cv2.calcHist([track], [0], None, [256], [0,256])
hist_track_sat = cv2.calcHist([track], [1], None, [256], [0,256])
hist_track_val = cv2.calcHist([track], [2], None, [256], [0,256])

hist_wall_hue = cv2.calcHist([wall], [0], None, [256], [0,256])
hist_wall_sat = cv2.calcHist([wall], [1], None, [256], [0,256])
hist_wall_val = cv2.calcHist([wall], [2], None, [256], [0,256])


# Normalize the histograms
hist_darkline_hue /= hist_darkline_hue.sum()
hist_darkline_sat /= hist_darkline_sat.sum()
hist_darkline_val /= hist_darkline_val.sum()

hist_outsidewall_hue /= hist_outsidewall_hue.sum()
hist_outsidewall_sat /= hist_outsidewall_sat.sum()
hist_outsidewall_val /= hist_outsidewall_val.sum()

hist_hilltrack_hue /= hist_hilltrack_hue.sum()
hist_hilltrack_sat /= hist_hilltrack_sat.sum()
hist_hilltrack_val /= hist_hilltrack_val.sum()

hist_topline_hue /= hist_topline_hue.sum()
hist_topline_sat /= hist_topline_sat.sum()
hist_topline_val /= hist_topline_val.sum()

hist_toptrack_hue /= hist_toptrack_hue.sum()
hist_toptrack_sat /= hist_toptrack_sat.sum()
hist_toptrack_val /= hist_toptrack_val.sum()

hist_track_hue /= hist_track_hue.sum()
hist_track_sat /= hist_track_sat.sum()
hist_track_val /= hist_track_val.sum()

hist_wall_hue /= hist_wall_hue.sum()
hist_wall_sat /= hist_wall_sat.sum()
hist_wall_val /= hist_wall_val.sum()

plt.title("Darkline Hue")
plt.xlabel("Hue value")
plt.ylabel("Ratio")
plt.plot(hist_darkline_hue)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/darklinehue.png')
plt.show()

plt.title("Darkline Sat")
plt.xlabel("Sat value")
plt.ylabel("Ratio")
plt.plot(hist_darkline_sat)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/darklinesat.png')
plt.show()

plt.title("Darkline Val")
plt.xlabel("Val value")
plt.ylabel("Ratio")
plt.plot(hist_darkline_val)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/darklineval.png')
plt.show()

plt.title("Outsidewall Hue")
plt.xlabel("Hue value")
plt.ylabel("Ratio")
plt.plot(hist_outsidewall_hue)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/outsidewallhue.png')
plt.show()

plt.title("Outsidewall Sat")
plt.xlabel("Sat value")
plt.ylabel("Ratio")
plt.plot(hist_outsidewall_sat)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/outsidewallsat.png')
plt.show()

plt.title("Outsidewall Val")
plt.xlabel("Val value")
plt.ylabel("Ratio")
plt.plot(hist_outsidewall_val)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/outsidewallval.png')
plt.show()

plt.title("Hilltrack Hue")
plt.xlabel("Hue value")
plt.ylabel("Ratio")
plt.plot(hist_hilltrack_hue)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/hilltrackhue.png')
plt.show()

plt.title("Hilltrack Sat")
plt.xlabel("Sat value")
plt.ylabel("Ratio")
plt.plot(hist_hilltrack_sat)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/hilltracksat.png')
plt.show()

plt.title("Hilltrack Val")
plt.xlabel("Val value")
plt.ylabel("Ratio")
plt.plot(hist_hilltrack_val)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/hilltrackval.png')
plt.show()

plt.title("Topline Hue")
plt.xlabel("Hue value")
plt.ylabel("Ratio")
plt.plot(hist_topline_hue)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/toplinehue.png')
plt.show()

plt.title("Topline Sat")
plt.xlabel("Sat value")
plt.ylabel("Ratio")
plt.plot(hist_topline_sat)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/toplinesat.png')
plt.show()

plt.title("Topline Val")
plt.xlabel("Val value")
plt.ylabel("Ratio")
plt.plot(hist_topline_val)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/toplineval.png')
plt.show()

plt.title("Toptrack Hue")
plt.xlabel("Hue value")
plt.ylabel("Ratio")
plt.plot(hist_toptrack_hue)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/toptrackhue.png')
plt.show()

plt.title("Toptrack Sat")
plt.xlabel("Sat value")
plt.ylabel("Ratio")
plt.plot(hist_toptrack_sat)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/toptracksat.png')
plt.show()

plt.title("Toptrack Val")
plt.xlabel("Val value")
plt.ylabel("Ratio")
plt.plot(hist_toptrack_val)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/toptrackval.png')
plt.show()

plt.title("Track Hue")
plt.xlabel("Hue value")
plt.ylabel("Ratio")
plt.plot(hist_track_hue)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/trackhue.png')
plt.show()

plt.title("Track Sat")
plt.xlabel("Sat value")
plt.ylabel("Ratio")
plt.plot(hist_track_sat)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/tracksat.png')
plt.show()

plt.title("Track Val")
plt.xlabel("Val value")
plt.ylabel("Ratio")
plt.plot(hist_track_val)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/trackval.png')
plt.show()

plt.title("Wall Hue")
plt.xlabel("Hue value")
plt.ylabel("Ratio")
plt.plot(hist_wall_hue)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/wallhue.png')
plt.show()

plt.title("Wall Sat")
plt.xlabel("Sat value")
plt.ylabel("Ratio")
plt.plot(hist_wall_sat)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/wallsat.png')
plt.show()

plt.title("Wall Val")
plt.xlabel("Val value")
plt.ylabel("Ratio")
plt.plot(hist_wall_val)
plt.xlim([0, 256])
plt.savefig('/home/fizzer/neuralNetworks/wallval.png')
plt.show()