import cv2

feed = cv2.VideoCapture(1)

ae = feed.get(cv2.CAP_PROP_AUTO_EXPOSURE)
ex = feed.get(cv2.CAP_PROP_EXPOSURE)

ae_new = ae
ex_new = ex

print(ae, ex)

while True:
    ret, frame = feed.read()

    cv2.imshow('frame', frame)
    c = cv2.waitKey(1)
    if c == 27:
        print("EXIT...")
        break
    if c == 49:
        feed.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        print("AE OFF... {}".format(feed.get(cv2.CAP_PROP_AUTO_EXPOSURE)))
    if c == 50:
        feed.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        ex = feed.set(cv2.CAP_PROP_EXPOSURE, ex)
        print("AE ON... {}".format(feed.get(cv2.CAP_PROP_AUTO_EXPOSURE)))
    
    if c == 51:
        ex_new -= 1
        ex = feed.set(cv2.CAP_PROP_EXPOSURE, ex_new)
        print("EXP... {}".format(feed.get(cv2.CAP_PROP_EXPOSURE)))
    if c == 52:
        ex_new += 1
        ex = feed.set(cv2.CAP_PROP_EXPOSURE, ex_new)
        print("EXP... {}".format(feed.get(cv2.CAP_PROP_EXPOSURE)))

feed.release()
cv2.destroyAllWindows()

# while True:
#     cv2.imshow('img', img)
#     c = cv2.waitKey(1)
#     if c==27:
#         print("Exiting...")
#         cv2.destroyAllWindows()
#         exit()
#     elif c > 0:
#         print("{} {}".format(c, chr(c&255)))