# FrontKamera
#FrontKamera

## main.py
frontcam_image_shape = (240, 320, 3) # small resolution to reduce bandwidth and latency.Höhe #Frontkamera   = 240 Pixel Breite = 320 Pixel Kanäle = 3

rezize -> CPU optimiert beliebige Pixelgrößen
pooling-> Kernal glatt teilbar durch pixel größe

cv.rezize (img, w,H,interpolation= cv.Inter_AREA) = interpolation wie : Gitter draf und macht mittelwert: Pixelfehler /Artevakte vermeiden
weiter operation 
internierest: einfach nächsten auswählen
bilineare inter-.: intepolation mit vier Stützpunkten den Mittelpunkt finden, gute Qualitäte, etwas schneller INter_linear

pixel größe überlegen für neues bild, damit nihct unnätig aber genung


